"""Operational checks for this fresh pipeline; no old experiments or weights.

python verify.py             # CPU integration checks, including learning/resume
python verify.py --quick     # faster installation checks
python verify.py --report docs/VERIFIED.json
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
from rap.config import Config, ModelConfig, TrainConfig, DataConfig
from rap.tokens import BOS, EOS, USER, ASSISTANT, TURN_END, VOCAB_SIZE, encode_text, serialize_record
from rap.data import normalize, encode_record, prepare_dataset, Corpus, StreamBatcher
from rap.model import RAP
from rap.losses import training_loss
from rap.runtime import load_payload, load_model, Session
from rap.engine import train
from rap.evaluation import evaluate

QUICK = False
EVIDENCE = {}

def small_config(steps=6):
    return Config(model=ModelConfig(embedding_size=8, state_size=16,
        controller_size=16, readout_size=16, future_horizons=[4], past_lags=[1, 4]),
        train=TrainConfig(steps=steps, batch_size=2, chunk_length=16,
        gradient_accumulation=1, learning_rate=0.003, minimum_learning_rate=0.003,
        warmup_steps=0, device="cpu", precision="float32", seed=23,
        log_every=1000, save_every=1000, evaluate_every=1000, snapshot_every=2,
        validation_target_tokens=48, validation_input_limit=8192, sample_every=0))

def dataset(directory, count=60, answer="hi"):
    raw = Path(directory)/"source.jsonl"
    raw.write_text("\n".join(json.dumps({"question": f"Item {i}?", "answer": answer})
                            for i in range(count)), encoding="utf-8")
    cfg = small_config()
    with redirect_stdout(io.StringIO()):
        prepared = prepare_dataset(raw, Path(directory)/"prepared", cfg.data)
    return prepared

class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_all_byte_maps_are_invertible_and_specials_stay_atomic(self):
        model = RAP(small_config().model)
        values = torch.arange(VOCAB_SIZE).expand(256, VOCAB_SIZE)
        last = torch.arange(256)
        lag2 = torch.randperm(256, generator=torch.Generator().manual_seed(81))
        indices = model.map_indices(last, lag2)
        for k in range(8):
            self.assertTrue(torch.equal(indices[:, k].sort(-1).values, values))
            inverse = indices[:, k].argsort(-1)
            self.assertTrue(torch.equal(inverse.gather(-1, indices[:, k]), values))
            self.assertTrue(torch.equal(indices[:, k, 256:], values[:, 256:]))
        EVIDENCE["byte_map_context_cases"] = 256*8

    def test_normalized_causal_predictions_and_backward(self):
        cfg = small_config()
        model = RAP(cfg.model)
        x = torch.tensor([[BOS, USER, 195, 169, TURN_END, ASSISTANT, 104, 105, EOS, USER, 65]])
        modified = x.clone(); modified[:, 8:] = torch.tensor([[99, 100, 101]])
        a, b = model(x), model(modified)
        self.assertTrue(torch.allclose(a["log_probs"][:, :8], b["log_probs"][:, :8], atol=2e-6, rtol=0))
        self.assertLess(float((a["log_probs"].exp().sum(-1)-1).abs().max()), 1e-6)
        batch = dict(input=x, target=torch.roll(x, -1, 1), mask=torch.ones_like(x, dtype=torch.bool))
        loss, count, terms = training_loss(model, a, batch, cfg.train)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNotNone(model.controller[0].weight.grad)
        self.assertIsNotNone(model.expression_gate.weight.grad)

    def test_chunked_carry_matches_full_replay(self):
        model = RAP(small_config().model).eval()
        values = torch.tensor([[BOS]+list(range(64, 120))+[EOS, USER, 90, TURN_END, ASSISTANT]])
        with torch.no_grad():
            full = model(values)
            state = model.initial_state(1)
            pieces = []
            for start in range(0, values.shape[1], 7):
                out = model(values[:, start:start+7], state)
                pieces.append(out["log_probs"]); state = out["state"].detached()
        difference = float((full["log_probs"]-torch.cat(pieces, 1)).abs().max())
        self.assertLess(difference, 1e-5)
        for name in ("memory", "seen", "written_at", "position", "last_byte", "previous_byte", "recent"):
            self.assertTrue(torch.allclose(getattr(full["state"], name).float(), getattr(state, name).float(), atol=2e-6, rtol=0))
        EVIDENCE["full_vs_chunked_max_log_probability_error"] = difference

    def test_BOS_resets_all_document_state_but_turns_preserve_memory(self):
        model = RAP(small_config().model).eval()
        with torch.no_grad():
            prior = model(torch.tensor([[BOS, 200]]))["state"].detached()
            saved = prior.memory[:, 200].clone()
            carried = model(torch.tensor([[USER, 65, TURN_END, ASSISTANT, 66, EOS]]), prior)["state"]
            self.assertTrue(torch.equal(saved, carried.memory[:, 200]))
            suffix = torch.tensor([[BOS, USER, 195, 169, TURN_END, ASSISTANT]])
            existing = model(suffix, carried)
            fresh = model(suffix)
        self.assertTrue(torch.allclose(existing["log_probs"], fresh["log_probs"], atol=2e-6, rtol=0))
        self.assertTrue(torch.equal(existing["state"].seen, fresh["state"].seen))

    def test_memory_is_bounded_and_old_values_remain_addressable(self):
        model = RAP(small_config().model)
        state = model.initial_state(1)
        byte = torch.tensor([98])
        key = 98 ^ 255
        payload = torch.linspace(-1, 1, 16)
        state.memory[0, key] = payload
        state.seen[0, key] = True
        state.written_at[0, key] = 0
        state.position[:] = 90000
        reads, _, valid, age = model._retrieve(byte, state)
        self.assertTrue(torch.equal(reads[0, 0], payload))
        self.assertTrue(bool(valid[0, 0]))
        self.assertEqual(int(age[0, 0]), 90000)
        self.assertEqual(state.memory.numel(), 256*16)
        EVIDENCE["addressed_old_snapshot_age_symbols"] = 90000

    def test_ordered_memory_channels_avoid_mean_swap_ambiguity(self):
        model = RAP(small_config().model)
        state = model.initial_state(1)
        idx = torch.tensor([98])
        a, b = 98 ^ 255, int(model.bit_reverse[98])
        state.seen[0, [a, b]] = True
        state.memory[0, a] = 1; state.memory[0, b] = 3
        first = model._retrieve(idx, state)[0]
        state.memory[0, a] = 3; state.memory[0, b] = 1
        swapped = model._retrieve(idx, state)[0]
        self.assertFalse(torch.equal(first, swapped))
        model.cfg.retrieval = "mean"
        mean_a = model._retrieve(idx, state)[0]
        state.memory[0, a] = 1; state.memory[0, b] = 3
        mean_b = model._retrieve(idx, state)[0]
        self.assertTrue(torch.equal(mean_a, mean_b))

    def test_data_schemas_unicode_masks_and_censored_endings(self):
        records = [
            {"question": "é", "answer": "ok"},
            {"prompt": "é", "completion": "ok"},
            {"instruction": "é", "output": "ok"},
            {"messages": [{"role": "user", "content": "é"}, {"role": "assistant", "content": "ok"}]},
            {"conversations": [{"from": "human", "value": "é"}, {"from": "gpt", "value": "ok"}]}
        ]
        encoded = [encode_record(normalize(r)) for r in records]
        self.assertTrue(all(value == encoded[0] for value in encoded))
        ids, mask = encoded[0]
        self.assertIn(195, ids); self.assertIn(169, ids)
        self.assertEqual([v for v, keep in zip(ids, mask) if keep], [111, 107, EOS])
        first = mask.index(1)
        self.assertEqual(ids[first-1], ASSISTANT)
        self.assertTrue(mask[first])
        censored = encode_record(normalize(dict(question="q", answer="ok", truncated=True)))
        self.assertNotEqual(censored[0][-1], EOS)
        text_ids, _ = serialize_record(text="\0é")
        self.assertIn(0, text_ids)
        self.assertGreater(BOS, 255)

    def test_content_split_and_dedup_and_batch_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/"source.jsonl"
            original = [dict(question=f"q{i}", answer=f"a{i}") for i in range(40)]
            raw.write_text("\n".join(json.dumps(x) for x in original+original[:10]+[{"unsupported": "skip"}]), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                path = prepare_dataset(raw, Path(directory)/"data", DataConfig())
            train_c, val = Corpus(path, "train"), Corpus(path, "validation")
            self.assertEqual(train_c.manifest["stats"]["duplicates"], 10)
            self.assertEqual(train_c.manifest["stats"]["invalid"], 1)
            hashes = np.memmap(path/"hashes.bin", dtype=np.uint8, mode="r", shape=(40, 32))
            a = {bytes(hashes[i]) for i in train_c.indices}
            b = {bytes(hashes[i]) for i in val.indices}
            self.assertFalse(a & b)
            stream = StreamBatcher(train_c, 2, 7, 9)
            stream.next(); saved = stream.state_dict()
            expected = stream.next()
            restored = StreamBatcher(train_c, 2, 7, 99)
            restored.load_state_dict(saved)
            actual = restored.next()
            self.assertTrue(all(torch.equal(expected[k], actual[k]) for k in expected))
            # Windows cannot remove the temporary corpus while memmaps are open.
            del hashes, stream, restored, train_c, val

    def test_empty_dataset_fails_promptly_and_atomic_prepare_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            good = dataset(directory)
            old = (good/"manifest.json").read_bytes()
            bad = Path(directory)/"bad.jsonl"; bad.write_text("{}\n")
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                prepare_dataset(bad, good, DataConfig(), force=True)
            self.assertEqual((good/"manifest.json").read_bytes(), old)

    def test_checkpoint_resume_matches_uninterrupted_optimizer_and_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            data = dataset(directory)
            cfg = small_config(steps=6)
            with redirect_stdout(io.StringIO()):
                one, _ = train(cfg, data, Path(directory)/"one", install_signal=False)
                two, _ = train(cfg, data, Path(directory)/"two", stop_at=2, install_signal=False)
                two, _ = train(cfg, data, Path(directory)/"two", resume=two, install_signal=False)
            a, b = load_payload(one), load_payload(two)
            maximum = max(float((a["model"][key]-b["model"][key]).abs().max())
                          for key in a["model"] if a["model"][key].dtype.is_floating_point)
            self.assertEqual(a["step"], 6); self.assertEqual(b["step"], 6)
            self.assertEqual(load_payload(Path(directory)/'one/snapshots/step_0000002.pt')['step'], 2)
            self.assertLess(maximum, 1e-7)
            self.assertEqual(a["stream"], b["stream"])
            for key in a["recurrent_state"]:
                self.assertTrue(torch.equal(a["recurrent_state"][key], b["recurrent_state"][key]))
            EVIDENCE["resumed_vs_uninterrupted_max_parameter_error"] = maximum

    def test_explicit_split_limits_and_non_text_blocks(self):
        with self.assertRaises(ValueError):
            normalize({"messages": [{"role": "assistant", "content": ["unsupported"]}]})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            a = directory/"train.jsonl"; b = directory/"val.jsonl"
            a.write_text("\n".join(json.dumps({"question": f"train {i}", "answer": "a"}) for i in range(10)))
            b.write_text("\n".join(json.dumps({"question": f"val {i}", "answer": "b"}) for i in range(10)))
            cfg = DataConfig(max_records=3)
            with redirect_stdout(io.StringIO()):
                prepared = prepare_dataset(a, directory/"prepared", cfg, validation_path=b)
            self.assertEqual(len(Corpus(prepared, "train").indices), 3)
            self.assertEqual(len(Corpus(prepared, "validation").indices), 3)

    def test_new_stage_initialization_resets_optimizer_stream_and_carrier(self):
        with tempfile.TemporaryDirectory() as directory:
            data = dataset(directory)
            cfg = small_config(steps=2)
            with redirect_stdout(io.StringIO()):
                old, _ = train(cfg, data, Path(directory)/"old", install_signal=False)
                new, _ = train(cfg, data, Path(directory)/"new", initialize=old,
                               stop_at=0, install_signal=False)
            a, b = load_payload(old), load_payload(new)
            self.assertEqual(b["step"], 0)
            self.assertEqual(b["stream"]["input_tokens"], 0)
            self.assertFalse(bool(b["recurrent_state"]["seen"].any()))
            self.assertFalse(bool(b["recurrent_state"]["h"].any()))
            self.assertTrue(all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]))

    def test_training_and_free_generation_can_learn_a_tiny_task(self):
        if QUICK:
            self.skipTest("full learning check omitted by --quick")
        with tempfile.TemporaryDirectory() as directory:
            data = dataset(directory, count=100, answer="hi")
            cfg = small_config(steps=90)
            cfg.train.learning_rate = cfg.train.minimum_learning_rate = 0.01
            cfg.train.future_weight = cfg.train.past_weight = cfg.train.boundary_weight = 0
            cfg.train.stop_weight = 0.01
            with redirect_stdout(io.StringIO()):
                checkpoint, _ = train(cfg, data, Path(directory)/"learn", install_signal=False)
            model, _, _, device = load_model(checkpoint, "cpu")
            value = evaluate(model, Corpus(data, "validation"), cfg.train, device, "float32",
                             target_budget=48, collect_diagnostics=True)
            self.assertLess(value["language_nats"], 0.2)
            session = Session(model, device); session.prompt("Item 999?")
            generated, info = session.generate(max_bytes=12, temperature=0)
            self.assertEqual(generated, b"hi")
            self.assertTrue(info["ended_on_eos"])
            self.assertEqual(len(info["trace"]), 3)
            self.assertLess(value["expression_intervention"]["native_nats"], 0.2)
            EVIDENCE["tiny_task_validation_nats"] = value["language_nats"]
            EVIDENCE["tiny_task_greedy_bytes"] = list(generated)
            EVIDENCE["tiny_task_emitted_EOS"] = info["ended_on_eos"]

def main():
    global QUICK
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--report")
    args = p.parse_args()
    QUICK = args.quick
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        seconds=time.monotonic()-started, torch=str(torch.__version__), device="cpu",
        cuda_available=torch.cuda.is_available(), evidence=EVIDENCE,
        scope="Operational/numerical checks and a tiny task; not language-quality validation or measured RTX 4070 performance.")
    if args.report:
        path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
