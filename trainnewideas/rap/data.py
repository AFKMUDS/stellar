"""Streaming dataset preparation and document-aware recurrent batching."""
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile
import uuid
import numpy as np
import torch
from .tokens import serialize_record

FORMAT = "rap-fresh-corpus-v1"
SUFFIXES = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz", ".json", ".txt", ".parquet")

def source_files(path):
    if isinstance(path, (list, tuple)):
        return sorted(set(p for item in path for p in source_files(item)))
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if not str(path).lower().endswith(SUFFIXES):
            raise ValueError(f"Unsupported dataset type: {path.name}")
        return [path]
    if path.is_dir():
        result = sorted(p for p in path.rglob("*") if p.is_file()
                        and str(p).lower().endswith(SUFFIXES))
        if result:
            return result
    raise FileNotFoundError(f"No supported dataset files at {path}")

def _content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if any(not isinstance(x, dict) or x.get("type", "text") not in
               {"text", "input_text", "output_text"} for x in value):
            return None
        text = [x.get("text") for x in value]
        if all(isinstance(x, str) for x in text):
            return "".join(text)
    return None

def normalize(obj):
    if isinstance(obj, str):
        return {"text": obj, "censored": False}
    if not isinstance(obj, dict):
        raise ValueError("record must be an object or text string")
    censored = bool(obj.get("censored") or obj.get("truncated")
                    or obj.get("finish_reason") == "length")
    messages = obj.get("messages", obj.get("conversations"))
    if isinstance(messages, list):
        result = []
        aliases = {"human": "user", "gpt": "assistant", "developer": "system"}
        for row in messages:
            if not isinstance(row, dict):
                raise ValueError("message is not an object")
            role = aliases.get(row.get("role", row.get("from")), row.get("role", row.get("from")))
            text = _content(row.get("content", row.get("value")))
            if role in {"tool", "function"}:
                role, text = "user", "Tool result:\n" + (text or "")
            if role not in {"system", "user", "assistant"} or text is None:
                raise ValueError("unsupported role or non-text message")
            result.append({"role": role, "content": text})
        if not result:
            raise ValueError("empty messages")
        return {"messages": result, "censored": censored}
    for left, right in [("question", "answer"), ("question", "response"),
                        ("prompt", "completion"), ("prompt", "response"),
                        ("problem", "solution"), ("instruction", "output"),
                        ("instruction", "completion"), ("input", "output")]:
        question, answer = _content(obj.get(left)), _content(obj.get(right))
        if question is not None and answer is not None:
            if left == "instruction" and isinstance(obj.get("input"), str) and obj["input"]:
                question += ("\n" if question else "") + obj["input"]
            return {"messages": [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}],
                    "censored": censored}
    if isinstance(obj.get("text"), str):
        return {"text": obj["text"], "censored": censored}
    for key in ("data", "record", "example"):
        if isinstance(obj.get(key), dict):
            return normalize(obj[key])
    raise ValueError("no supported text, message list, or prompt/answer fields")

def encode_record(record):
    ids, mask = serialize_record(record.get("messages"), record.get("text"))
    if record.get("censored"):
        # Never label an unfinished final answer or text cutoff as a true EOS.
        from .tokens import EOS
        if ids[-1] == EOS:
            ids.pop(); mask.pop()
    return ids, mask

def iter_objects(path, error):
    name = str(path).lower()
    if name.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    error(path, line_number, f"invalid JSON: {exc}")
                    continue
                yield line_number, value
    elif name.endswith(".json"):
        if path.stat().st_size > 128*1024*1024:
            try:
                import ijson
            except ImportError as exc:
                raise RuntimeError("Large JSON arrays need optional 'ijson'; install it or use JSONL.") from exc
            with path.open("rb") as f:
                first = f.read(4096).lstrip()
            if not first.startswith(b"["):
                raise ValueError("Large JSON must be a top-level array. Use JSONL for nested containers.")
            with path.open("rb") as f:
                for i, value in enumerate(ijson.items(f, "item"), 1):
                    yield i, value
        else:
            with path.open("r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                values = obj
            elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
                values = obj["data"]
            else:
                values = [obj]
            yield from enumerate(values, 1)
    elif name.endswith(".txt"):
        # Bounded paragraph buffering, including files with very long lines.
        buffer, number = "", 0
        with path.open("r", encoding="utf-8-sig") as f:
            for block in iter(lambda: f.read(65536), ""):
                buffer += block
                while "\n\n" in buffer:
                    text, buffer = buffer.split("\n\n", 1)
                    if text.strip():
                        number += 1
                        yield number, text
                while len(buffer) > 262144:
                    cut = buffer.rfind("\n", 0, 262144)
                    cut = cut if cut > 0 else 262144
                    number += 1
                    yield number, {"text": buffer[:cut], "censored": True}
                    buffer = buffer[cut:]
            if buffer.strip():
                number += 1
                yield number, buffer
    elif name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input needs optional 'pyarrow': pip install pyarrow") from exc
        number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for obj in batch.to_pylist():
                number += 1
                yield number, obj

def _signature(files, validation, cfg):
    entries = []
    for split, paths in [("main", files), ("validation", validation)]:
        for p in paths:
            stat = p.stat()
            entries.append([split, str(p), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps([entries, asdict(cfg)], sort_keys=True).encode()).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def prepare_dataset(dataset, output, cfg, *, validation_path=None, force=False):
    main = source_files(dataset)
    validation = source_files(validation_path) if validation_path else []
    output = Path(output).resolve()
    signature = _signature(main, validation, cfg)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("source_signature") != signature:
            raise ValueError("Prepared data belongs to a different source/settings. Choose a new --prepared directory or explicitly --force-prepare.")
        Corpus(output, "train")  # validate file sizes before reuse
        Corpus(output, "validation")
        return output
    if output.exists() and any(output.iterdir()) and not force:
        raise ValueError(f"Prepared destination is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+".preparing_", dir=output.parent))
    stats = dict(records_seen=0, records_written=0, invalid=0, no_supervision=0,
                 oversized=0, duplicates=0, cross_split_duplicates=0,
                 train_records=0, validation_records=0, tokens=0, target_tokens=0,
                 censored_records=0, first_errors=[])
    connection = None
    handles = {}
    try:
        connection = sqlite3.connect(staging/"dedup.sqlite")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE docs (digest BLOB PRIMARY KEY, split INTEGER, first_index INTEGER)")
        handles = {name: (staging/name).open("wb", buffering=1024*1024)
                   for name in ["tokens.bin", "loss_mask.bin", "records.bin", "splits.bin", "hashes.bin"]}
        def error(path, line, reason):
            stats["invalid"] += 1
            if len(stats["first_errors"]) < 12:
                stats["first_errors"].append(dict(file=str(path), line=line, reason=reason))
        stop = False
        for forced, paths in [(0 if validation else None, main), (1, validation)]:
            stop = False
            split_written = 0
            for path in paths:
                print(f"Preparing {path.name}", flush=True)
                for line, obj in iter_objects(path, error):
                    stats["records_seen"] += 1
                    try:
                        record = normalize(obj)
                        canonical = {k: v for k, v in record.items() if k != "censored"}
                        canonical_bytes = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    except (ValueError, TypeError, UnicodeError) as exc:
                        error(path, line, str(exc)); continue
                    if cfg.max_record_bytes and len(canonical_bytes) > cfg.max_record_bytes:
                        stats["oversized"] += 1; continue
                    ids, mask = encode_record(record)
                    if len(ids) < 2 or not any(mask[1:]):
                        stats["no_supervision"] += 1; continue
                    digest = hashlib.sha256(canonical_bytes).digest()
                    score = hashlib.sha256(str(cfg.split_seed).encode()+digest).digest()
                    split = forced if forced is not None else int(int.from_bytes(score[:8], "big")/2**64 < cfg.validation_fraction)
                    previous = connection.execute("SELECT split FROM docs WHERE digest=?", (digest,)).fetchone()
                    if previous is not None:
                        if forced is not None and previous[0] != split:
                            stats["cross_split_duplicates"] += 1; continue
                        split = previous[0]
                        if cfg.deduplicate:
                            stats["duplicates"] += 1; continue
                    else:
                        connection.execute("INSERT INTO docs VALUES (?, ?, ?)",
                                           (digest, split, stats["records_written"]))
                    token_bytes = np.asarray(ids, dtype="<u2").tobytes()
                    handles["tokens.bin"].write(token_bytes)
                    handles["loss_mask.bin"].write(bytes(mask))
                    handles["records.bin"].write(struct.pack("<QQ", stats["tokens"], len(ids)))
                    handles["splits.bin"].write(bytes([split]))
                    handles["hashes.bin"].write(digest)
                    stats["tokens"] += len(ids)
                    stats["target_tokens"] += sum(mask[1:])
                    stats["records_written"] += 1
                    split_written += 1
                    stats["censored_records"] += int(record.get("censored", False))
                    if stats["records_written"] % 10000 == 0:
                        connection.commit()
                        print(f"  {stats['records_written']:,} records; {stats['tokens']:,} symbols", flush=True)
                    if cfg.max_records and (split_written if validation else stats['records_written']) >= cfg.max_records:
                        stop = True; break
                if stop: break
            if stop and not validation: break
        for f in handles.values():
            f.flush(); f.close()
        handles = {}
        connection.commit()
        count = stats["records_written"]
        unique = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if unique < 2:
            raise ValueError("Need at least two distinct supervised records for disjoint training and validation.")
        partition = np.memmap(staging/"splits.bin", dtype=np.uint8, mode="r+", shape=(count,))
        hashes = np.memmap(staging/"hashes.bin", dtype=np.uint8, mode="r", shape=(count, 32))
        # Deterministic fallback for tiny corpora whose hash split is empty.
        for missing in (0, 1):
            if not np.any(partition == missing):
                if validation:
                    raise ValueError("Explicit training/validation files must each contain distinct supervised records.")
                candidate = connection.execute("SELECT digest FROM docs ORDER BY digest LIMIT 1 OFFSET ?", (missing,)).fetchone()[0]
                needle = np.frombuffer(candidate, dtype=np.uint8)
                for start in range(0, count, 65536):
                    matches = np.all(hashes[start:start+65536] == needle, axis=1)
                    local = partition[start:start+65536]
                    local[matches] = missing
                    del local  # release memmap view before atomic directory rename on Windows
                print("Tiny-corpus fallback assigned a whole content-hash group to the missing split.", flush=True)
        partition.flush()
        stats["train_records"] = int(np.sum(partition == 0))
        stats["validation_records"] = int(np.sum(partition == 1))
        del partition, hashes
        connection.close(); connection = None
        (staging/"dedup.sqlite").unlink()
        file_hashes = {name: sha256_file(staging/name) for name in
                       ["tokens.bin", "loss_mask.bin", "records.bin", "splits.bin", "hashes.bin"]}
        fingerprint = hashlib.sha256(json.dumps([FORMAT, file_hashes], sort_keys=True).encode()).hexdigest()
        manifest = dict(format=FORMAT, source_signature=signature, fingerprint=fingerprint,
                        tokenizer="raw-utf8-bytes-plus-structural-v1",
                        sources=[str(p) for p in main], explicit_validation=[str(p) for p in validation],
                        data_config=asdict(cfg), stats=stats, file_sha256=file_hashes)
        (staging/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        backup = None
        if output.exists():
            backup = output.with_name(output.name+".old_"+uuid.uuid4().hex[:8])
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup: backup.rename(output)
            raise
        if backup: shutil.rmtree(backup)
        print(json.dumps(stats, indent=2), flush=True)
        return output
    except BaseException:
        for f in handles.values():
            f.close()
        if connection is not None: connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise

class Corpus:
    def __init__(self, directory, split):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory/"manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Incompatible prepared dataset format.")
        n, tokens = self.manifest["stats"]["records_written"], self.manifest["stats"]["tokens"]
        expected = {"tokens.bin": tokens*2, "loss_mask.bin": tokens, "records.bin": n*16,
                    "splits.bin": n, "hashes.bin": n*32}
        for name, size in expected.items():
            if (self.directory/name).stat().st_size != size:
                raise ValueError(f"Prepared file size mismatch: {name}; prepare again.")
        self.tokens = np.memmap(self.directory/"tokens.bin", dtype="<u2", mode="r", shape=(tokens,))
        self.mask = np.memmap(self.directory/"loss_mask.bin", dtype=np.uint8, mode="r", shape=(tokens,))
        self.records = np.memmap(self.directory/"records.bin", dtype="<u8", mode="r", shape=(n, 2))
        self.partition = np.memmap(self.directory/"splits.bin", dtype=np.uint8, mode="r", shape=(n,))
        code = 0 if split == "train" else 1 if split == "validation" else None
        if code is None: raise ValueError("split must be train or validation")
        self.indices = np.flatnonzero(self.partition == code)
        self.split = split
        if not len(self.indices):
            raise ValueError(f"No records in {split} split.")

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]

class StreamBatcher:
    """Keep each lane on a complete document; TBPTT cuts gradients, not its state."""
    def __init__(self, corpus, batch_size, chunk_length, seed, *, shuffle=True):
        self.corpus, self.batch_size, self.chunk_length = corpus, batch_size, chunk_length
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.next_record = 0
        self.completed_records = 0
        self.input_tokens = 0
        self.lanes = [self._new_lane() for _ in range(batch_size)]

    def _new_lane(self):
        if self.shuffle:
            record = int(self.rng.choice(self.corpus.indices))
        else:
            record = int(self.corpus.indices[self.next_record % len(self.corpus.indices)])
            self.next_record += 1
        return [record, 0]

    def next(self):
        x = np.empty((self.batch_size, self.chunk_length), dtype=np.int64)
        y = np.empty_like(x); mask = np.empty_like(x, dtype=np.bool_)
        for b in range(self.batch_size):
            for t in range(self.chunk_length):
                record, offset = self.lanes[b]
                start, length = map(int, self.corpus.records[record])
                if offset >= length-1:
                    self.completed_records += 1
                    self.lanes[b] = self._new_lane()
                    record, offset = self.lanes[b]
                    start, length = map(int, self.corpus.records[record])
                at = start + offset
                x[b, t] = self.corpus.tokens[at]
                y[b, t] = self.corpus.tokens[at+1]
                mask[b, t] = self.corpus.mask[at+1]
                self.lanes[b][1] += 1
        self.input_tokens += x.size
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(y),
                "mask": torch.from_numpy(mask)}

    def state_dict(self):
        return dict(lanes=[list(x) for x in self.lanes], rng=self.rng.bit_generator.state,
                    next_record=self.next_record, completed_records=self.completed_records,
                    input_tokens=self.input_tokens, fingerprint=self.corpus.fingerprint,
                    batch_size=self.batch_size, chunk_length=self.chunk_length,
                    shuffle=self.shuffle)

    def load_state_dict(self, raw):
        for key, value in [("fingerprint", self.corpus.fingerprint), ("batch_size", self.batch_size),
                           ("chunk_length", self.chunk_length), ("shuffle", self.shuffle)]:
            if raw[key] != value:
                raise ValueError(f"Resume batch stream mismatch: {key}")
        self.lanes = [list(x) for x in raw["lanes"]]
        self.rng.bit_generator.state = raw["rng"]
        self.next_record = raw["next_record"]
        self.completed_records = raw["completed_records"]
        self.input_tokens = raw["input_tokens"]
