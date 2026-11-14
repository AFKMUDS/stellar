"""Persistent document streams, gradient accumulation, validation and exact resume."""
import json
import math
from pathlib import Path
import signal
import time
import torch
from .config import as_dict, from_dict
from .data import Corpus, StreamBatcher
from .evaluation import evaluate
from .losses import training_loss
from .model import RAP, State, parameter_count
from .runtime import (seed_everything, choose_device, choose_precision, autocast,
                      save_checkpoint, load_payload, restore_rng, Session)

def learning_rate(cfg, step):
    if step < cfg.warmup_steps:
        return cfg.learning_rate*(step+1)/max(cfg.warmup_steps, 1)
    progress = (step-cfg.warmup_steps)/max(cfg.steps-cfg.warmup_steps-1, 1)
    cosine = 0.5*(1+math.cos(math.pi*min(max(progress, 0), 1)))
    return cfg.minimum_learning_rate+(cfg.learning_rate-cfg.minimum_learning_rate)*cosine

def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")

def train(config, prepared, run_dir, *, resume=None, initialize=None, stop_at=None, install_signal=True):
    cfg = config.train
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir/"latest.pt"
    if resume == "auto":
        resume = str(latest) if latest.exists() else None
    if resume and initialize:
        raise ValueError('Choose resume or initialize, not both.')
    if latest.exists() and not resume:
        raise ValueError("This run already has a checkpoint. Use --resume auto or choose a fresh --run directory.")
    torch.set_num_threads(cfg.cpu_threads)
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    precision = choose_precision(cfg.precision, device)
    train_corpus, val_corpus = Corpus(prepared, "train"), Corpus(prepared, "validation")
    validation_sources = {
        name: Corpus(prepared, "validation", source=name)
        for name in train_corpus.manifest.get("source_names", [])
    }
    model = RAP(config.model).to(device)
    if initialize:
        source = load_payload(initialize)
        if as_dict(from_dict(source['config']))['model'] != as_dict(config)['model']:
            raise ValueError('Initialization requires the same new-model architecture.')
        model.load_state_dict(source['model'], strict=True)
        (run_dir/'initialization.json').write_text(json.dumps(dict(source=str(Path(initialize).resolve()),
            source_step=source['step'], optimizer_and_stream_reset=True), indent=2), encoding='utf-8')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    stream = StreamBatcher(train_corpus, cfg.batch_size, cfg.chunk_length, cfg.seed+100)
    state = model.initial_state(cfg.batch_size, device)
    step, target_tokens, best_loss = 0, 0, None
    target_raw_bytes = 0
    if resume:
        raw = load_payload(resume)
        previous = from_dict(raw["config"])
        if as_dict(previous)["model"] != as_dict(config)["model"]:
            raise ValueError("Model config differs from checkpoint; exact resume cannot change the architecture.")
        if as_dict(previous)["data"] != as_dict(config)["data"]:
            raise ValueError("Data config differs from checkpoint; exact resume cannot change source weights or splits.")
        operational = {'steps', 'device', 'precision', 'cpu_threads', 'log_every', 'save_every',
                       'evaluate_every', 'validation_target_tokens', 'validation_input_limit',
                       'snapshot_every', 'sample_every', 'sample_max_bytes', 'sample_prompt'}
        for field, value in vars(previous.train).items():
            if field not in operational and value != getattr(cfg, field):
                raise ValueError(f'Resume requires unchanged train.{field}; use the saved run config.')
        if raw["corpus_fingerprint"] != train_corpus.fingerprint:
            raise ValueError("Checkpoint and prepared dataset fingerprints differ.")
        model.load_state_dict(raw["model"], strict=True)
        optimizer.load_state_dict(raw["optimizer"])
        scaler.load_state_dict(raw["scaler"])
        stream.load_state_dict(raw["stream"])
        state = State.unpack(raw["recurrent_state"], device)
        step, target_tokens, best_loss = raw["step"], raw.get("target_tokens", 0), raw.get("best_loss")
        target_raw_bytes = raw.get("metrics", {}).get("target_raw_bytes", 0)
        restore_rng(raw["rng"])
        if previous.train.steps != cfg.steps:
            print("Extended/changed target steps: future learning rates follow the new configured horizon.", flush=True)
    (run_dir/"config.json").write_text(json.dumps(as_dict(config), indent=2), encoding="utf-8")
    environment = dict(device=str(device), precision=precision, torch=str(torch.__version__),
        parameters=parameter_count(model), state_size=config.model.state_size,
        byte_memory_slots=256, train_records=len(train_corpus.indices),
        validation_records=len(val_corpus.indices), corpus_fingerprint=train_corpus.fingerprint)
    if validation_sources:
        environment["source_stats"] = train_corpus.manifest["source_stats"]
    (run_dir/"environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment, indent=2), flush=True)
    stop_requested = [False]
    old_handler = None
    if install_signal:
        def request_stop(signum, frame):
            stop_requested[0] = True
            print("\nStop requested: finishing the current optimizer update, then saving.", flush=True)
        old_handler = signal.signal(signal.SIGINT, request_stop)
    def save(path, metrics):
        save_checkpoint(path, model=model, config=config, optimizer=optimizer,
            scaler=scaler, step=step, stream=stream, state=state,
            metrics=metrics, best_loss=best_loss, target_tokens=target_tokens)
    metrics = {}
    initial_target_raw_bytes = target_raw_bytes
    beginning = time.monotonic()
    limit = min(cfg.steps, stop_at) if stop_at is not None else cfg.steps
    model.train()
    try:
        while step < limit and not stop_requested[0]:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(cfg, step)
            optimizer.zero_grad(set_to_none=True)
            tokens = 0
            raw_bytes = 0
            supervised_chunks = 0
            prefill_chunks = 0
            sums = {}
            while supervised_chunks < cfg.gradient_accumulation:
                batch = {k: v.to(device) for k, v in stream.next().items()}
                has_targets = bool(batch["mask"].any().item())
                with torch.set_grad_enabled(has_targets):
                    with autocast(device, precision):
                        output = model(batch["input"], state)
                        if has_targets:
                            loss_sum, count, terms = training_loss(model, output, batch, cfg)
                state = output["state"].detached()
                if not has_targets:
                    prefill_chunks += 1
                    if prefill_chunks % 100 == 0:
                        print(f"Prefilling long records: {prefill_chunks} chunks, state preserved.", flush=True)
                    if stop_requested[0] and tokens == 0:
                        break
                    continue
                if not torch.isfinite(loss_sum).item():
                    raise FloatingPointError("Non-finite loss before optimizer update. Last completed checkpoint remains unchanged.")
                scaler.scale(loss_sum).backward()
                count = int(count.item())
                tokens += count; supervised_chunks += 1
                raw_bytes += int((batch["mask"] & (batch["target"] < 256)).sum().item())
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0)+float(value.detach().item())*count
            if tokens == 0:
                save(latest, metrics)
                break
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(tokens)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            if not torch.isfinite(norm).item():
                if not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradients; no optimizer update or new checkpoint was committed.")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step += 1; target_tokens += tokens; target_raw_bytes += raw_bytes
            metrics = {name: value/tokens for name, value in sums.items()}
            metrics.update(step=step, target_tokens=target_tokens,
                target_raw_bytes=target_raw_bytes,
                input_symbols=stream.input_tokens, completed_records=stream.completed_records,
                learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm=float(norm.item()) if torch.isfinite(norm).item() else None,
                optimizer_update_skipped=(scaler.get_scale() < previous_scale),
                elapsed_seconds=time.monotonic()-beginning)
            if stream.source_names:
                metrics["source_supervised_bytes"] = dict(zip(
                    stream.source_names, map(int, stream.source_supervised_bytes)))
            metrics["supervised_bytes_per_second"] = (target_raw_bytes - initial_target_raw_bytes) / max(
                metrics["elapsed_seconds"], 1e-9)
            if device.type == "cuda":
                metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
            if step == 1 or step % cfg.log_every == 0:
                append_json(run_dir/"metrics.jsonl", metrics)
                print(f"step {step}/{cfg.steps}  LM {metrics['language_nats']:.4f} nats  "
                      f"targets {target_tokens:,}  lr {metrics['learning_rate']:.3g}", flush=True)
            if not stop_requested[0] and (step % cfg.evaluate_every == 0 or step == limit):
                validation = evaluate(model, val_corpus, cfg, device, precision)
                if validation_sources:
                    validation["by_source"] = {
                        name: evaluate(model, corpus, cfg, device, precision)
                        for name, corpus in validation_sources.items()
                    }
                append_json(run_dir/"validation.jsonl", dict(step=step, **validation))
                print(f"validation step {step}: {json.dumps(validation)}", flush=True)
                if validation["language_nats"] is not None and (best_loss is None or validation["language_nats"] < best_loss):
                    best_loss = validation["language_nats"]
                    save(run_dir/"best.pt", validation)
                model.train()
            if not stop_requested[0] and cfg.sample_every and step % cfg.sample_every == 0:
                model.eval()
                session = Session(model, device, precision=precision)
                session.prompt(cfg.sample_prompt)
                raw_bytes, info = session.generate(max_bytes=cfg.sample_max_bytes, seed=cfg.seed)
                append_json(run_dir/"samples.jsonl", dict(step=step, prompt=cfg.sample_prompt,
                    text=raw_bytes.decode("utf-8", errors="replace"), bytes_hex=raw_bytes.hex(),
                    ended_on_eos=info["ended_on_eos"]))
                print("sample:", raw_bytes.decode("utf-8", errors="replace"), flush=True)
                model.train()
            if step == 1 or step % cfg.save_every == 0 or step == limit or stop_requested[0]:
                save(latest, metrics)
            if cfg.snapshot_every and step % cfg.snapshot_every == 0:
                save(run_dir/'snapshots'/f'step_{step:07d}.pt', metrics)
        if not latest.exists() or stop_requested[0] or step >= limit:
            save(latest, metrics)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    return latest, metrics
