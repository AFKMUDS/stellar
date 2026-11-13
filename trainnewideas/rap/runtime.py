"""Device selection, checkpoint compatibility, and autoregressive sessions."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .config import as_dict, from_dict
from .model import RAP, State
from .tokens import BOS, EOS, USER, ASSISTANT, TURN_END, encode_text

CHECKPOINT_FORMAT = "rap-fresh-recurrent-expression-v1"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device(request="auto"):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Activate your CUDA PyTorch environment, or use --device cpu.")
    return device

def choose_precision(request, device):
    if device.type != "cuda":
        return "float32"
    if request == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if request == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; choose fp16 or float32.")
    return request

def autocast(device, precision):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)

def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def save_checkpoint(path, *, model, config, optimizer, scaler, step, stream, state,
                    metrics=None, best_loss=None, target_tokens=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = dict(format=CHECKPOINT_FORMAT, format_version=1, config=as_dict(config),
        model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        step=int(step), stream=stream.state_dict(), recurrent_state=state.packed(),
        rng=rng_state(), metrics=metrics or {}, best_loss=best_loss,
        target_tokens=int(target_tokens), corpus_fingerprint=stream.corpus.fingerprint,
        torch_version=str(torch.__version__))
    temporary = path.with_name(path.name+".writing")
    torch.save(raw, temporary)
    os.replace(temporary, path)

def load_payload(path):
    # New checkpoints contain tensors and ordinary Python containers only.
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("This is not a checkpoint from RAP_Fresh_Training. Start fresh; old B/C weights have different architecture/token IDs.")
    return raw

def load_model(path, device="auto"):
    device = choose_device(device)
    raw = load_payload(path)
    cfg = from_dict(raw["config"])
    torch.set_num_threads(cfg.train.cpu_threads)
    model = RAP(cfg.model).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, cfg, raw, device

class Session:
    def __init__(self, model, device, *, precision="float32"):
        self.model, self.device, self.precision = model, device, precision
        self.state = model.initial_state(1, device)
        self.last = None
        self.started = False

    def reset(self):
        self.state = self.model.initial_state(1, self.device)
        self.last = None
        self.started = False

    @torch.no_grad()
    def feed(self, tokens):
        for start in range(0, len(tokens), 128):
            ids = torch.tensor(tokens[start:start+128], device=self.device, dtype=torch.long)[None]
            with autocast(self.device, self.precision):
                self.last = self.model(ids, self.state)
            self.state = self.last["state"].detached()
        self.started = True

    def prompt(self, text, *, raw=False):
        prefix = [] if self.started else [BOS]
        if raw:
            self.feed(prefix+encode_text(text))
        else:
            self.feed(prefix+[USER]+encode_text(text)+[TURN_END, ASSISTANT])

    @torch.no_grad()
    def generate(self, *, max_bytes=512, temperature=0.8, top_k=40, seed=23, on_byte=None):
        if self.last is None:
            raise ValueError("Feed a prompt before generating.")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        generated, trace = bytearray(), []
        ended = False
        for _ in range(max_bytes):
            logp = self.last["log_probs"][0, -1].float().clone()
            # Ordinary inference emits raw bytes or the learned response-end action.
            allowed = torch.zeros_like(logp, dtype=torch.bool)
            allowed[:256] = True; allowed[EOS] = True
            logp[~allowed] = -torch.inf
            if temperature <= 0:
                token = int(logp.argmax().item())
            else:
                scores = logp/temperature
                if top_k > 0:
                    threshold = torch.topk(scores, min(top_k, 257)).values[-1]
                    scores[scores < threshold] = -torch.inf
                token = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator).item())
            trace.append(dict(token=token,
                primitive=int(self.last["transition_probs"][0, -1].argmax().item()),
                expression_path=int(self.last["expression_probs"][0, -1].argmax().item()),
                stop_probability=float(torch.sigmoid(self.last["stop_logits"][0, -1]).item())))
            # The selected output, never a supplied answer, is committed to recurrence.
            self.feed([token])
            if token == EOS:
                ended = True; break
            generated.append(token)
            if on_byte is not None:
                on_byte(bytes([token]))
        return bytes(generated), dict(ended_on_eos=ended, generated_bytes=len(generated), trace=trace)
