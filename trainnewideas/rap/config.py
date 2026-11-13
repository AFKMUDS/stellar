from dataclasses import asdict, dataclass, field, fields
import copy
import json
from pathlib import Path

@dataclass
class ModelConfig:
    embedding_size: int = 256
    state_size: int = 512
    primitives: int = 8
    controller_size: int = 512
    readout_size: int = 256
    numeric_features: bool = True
    byte_operator_seeds: bool = True
    retrieval: str = "tagged"  # tagged, mean, none, random
    expression_head: str = "reversible"  # reversible, direct, random
    future_horizons: list = field(default_factory=lambda: [4, 16])
    past_lags: list = field(default_factory=lambda: [1, 4])
    permutation_seed: int = 701

@dataclass
class TrainConfig:
    steps: int = 100000
    batch_size: int = 2
    chunk_length: int = 128
    gradient_accumulation: int = 4
    learning_rate: float = 0.0002
    minimum_learning_rate: float = 0.00002
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    device: str = "auto"
    precision: str = "auto"
    seed: int = 23
    cpu_threads: int = 1
    log_every: int = 10
    save_every: int = 250
    evaluate_every: int = 500
    validation_target_tokens: int = 4096
    validation_input_limit: int = 262144
    snapshot_every: int = 10000
    sample_every: int = 1000
    sample_max_bytes: int = 256
    sample_prompt: str = "What is gravity?"
    future_weight: float = 0.05
    past_weight: float = 0.02
    boundary_weight: float = 0.05
    stop_weight: float = 0.02
    stop_positive_weight: float = 16.0

@dataclass
class DataConfig:
    validation_fraction: float = 0.02
    split_seed: int = 2718
    deduplicate: bool = True
    max_record_bytes: int = 1048576
    max_records: int = 0

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

def from_dict(raw):
    result = Config()
    unknown = set(raw) - {"model", "train", "data"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section in ("model", "train", "data"):
        target = getattr(result, section)
        valid = {f.name for f in fields(target)}
        for key, value in raw.get(section, {}).items():
            if key not in valid:
                raise ValueError(f"Unknown config key: {section}.{key}")
            setattr(target, key, copy.deepcopy(value))
    validate(result)
    return result

def load_config(path=None, overrides=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".", 1)
        raw.setdefault(section, {})[key] = value
    return from_dict(raw)

def validate(cfg):
    m, t, d = cfg.model, cfg.train, cfg.data
    if m.primitives != 8:
        raise ValueError("This architecture has exactly eight learned transition primitives.")
    for value in (m.embedding_size, m.state_size, m.controller_size, m.readout_size,
                  t.steps, t.batch_size, t.chunk_length, t.gradient_accumulation,
                  t.log_every, t.save_every, t.evaluate_every,
                  t.validation_target_tokens, t.validation_input_limit):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Model dimensions, budgets, batch and chunk sizes must be positive integers.")
    if m.retrieval not in {"tagged", "mean", "none", "random"}:
        raise ValueError("Unknown retrieval mode.")
    if m.expression_head not in {"reversible", "direct", "random"}:
        raise ValueError("Unknown expression head.")
    if not 0 < d.validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if t.precision not in {"auto", "float32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, float32, bf16 or fp16.")
    if t.learning_rate <= 0 or t.minimum_learning_rate < 0 or t.warmup_steps < 0:
        raise ValueError("Invalid learning-rate schedule.")
    if any(not isinstance(x, int) or x <= 0 for x in m.future_horizons + m.past_lags):
        raise ValueError("Auxiliary lags and horizons must be positive integers.")
    if t.snapshot_every < 0:
        raise ValueError('snapshot_every cannot be negative.')
    if any(getattr(t, k) < 0 for k in ('future_weight', 'past_weight', 'boundary_weight', 'stop_weight')):
        raise ValueError('Auxiliary loss weights cannot be negative.')
    if t.gradient_clip <= 0 or t.cpu_threads <= 0 or d.max_records < 0 or d.max_record_bytes < 0:
        raise ValueError('Invalid gradient clipping, CPU threads, or data limits.')
    if t.sample_every < 0 or t.sample_max_bytes <= 0:
        raise ValueError("Invalid sampling configuration.")

def as_dict(config):
    return asdict(config)
