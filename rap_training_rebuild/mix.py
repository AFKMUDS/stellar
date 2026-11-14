"""Bounded, deterministic preparation of source-weighted RAP training records.

Network datasets are opened only when --allow-network is explicitly provided.
Weights are applied by the fresh StreamBatcher's supervised-byte accounting,
not by duplicating short records.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import random
import re
import sqlite3
import tempfile
from pathlib import Path

from trainnewideas.rap.data import encode_record, iter_objects, normalize, sha256_file


KINDS = {"natural", "fineweb", "smoltalk", "openthoughts", "synthetic"}
DEFAULT_WEIGHTS = {
    "natural": 0.35, "fineweb": 0.25, "smoltalk": 0.20,
    "openthoughts": 0.10, "synthetic": 0.10,
}


def _prefix(text, limit):
    if not isinstance(text, str):
        raise ValueError("Missing text")
    if len(text.encode("utf-8")) <= limit:
        return text, False
    truncated = text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    return truncated, True


def adapt(source, row, *, max_bytes):
    """Use explicit fields; never treat an unlabelled reasoning trace as a final answer."""
    kind = source["kind"]
    if kind == "fineweb":
        text, censored = _prefix(row["text"], max_bytes - 256)
        return normalize({"text": text, "censored": censored})
    if kind == "natural":
        question = row["question"]
        answer = row.get("reference_answer") or row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("Natural Reasoning needs question and reference_answer")
        answer, censored = _prefix(answer, source.get("max_target_bytes", max_bytes))
        return normalize({"question": question, "answer": answer, "censored": censored})
    if kind == "smoltalk":
        messages = row.get("messages") or row.get("conversation") or row.get("conversations")
        if messages is None:
            raise ValueError("SmolTalk requires messages")
        record = normalize({"messages": messages})
        cleaned, censored = [], False
        for index, message in enumerate(record["messages"]):
            text = message["content"]
            if message["role"] == "assistant":
                if not source.get("include_reasoning", False):
                    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                    if "<think>" in text or "</think>" in text:
                        raise ValueError("Unclosed reasoning span")
                text, cut = _prefix(text, source.get("max_target_bytes", max_bytes))
                if cut and index != len(record["messages"]) - 1:
                    raise ValueError("Only a final assistant response can be truncated")
                censored |= cut
            cleaned.append({"role": message["role"], "content": text})
        if not any(message["role"] == "assistant" and message["content"] for message in cleaned):
            raise ValueError("No supervised assistant text")
        record["messages"], record["censored"] = cleaned, censored
        return record
    if kind == "openthoughts":
        question = row.get("problem") or row.get("question") or row.get("prompt")
        if source.get("include_reasoning", False):
            answer = row.get("generated_solution") or row.get("solution") or row.get("reasoning")
        else:
            answer = row.get("final_answer") or row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("OpenThoughts requires a question and explicit selected answer field")
        if not source.get("include_reasoning", False):
            answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
            if "<think>" in answer or "</think>" in answer:
                raise ValueError("Unclosed reasoning span")
        answer, censored = _prefix(answer, source.get("max_target_bytes", max_bytes))
        return normalize({"question": question, "answer": answer, "censored": censored})
    if kind == "synthetic":
        return normalize(row)
    raise ValueError(f"Unknown adapter: {kind}")


def validate_plan(plan):
    sources = plan.get("sources", [])
    if not sources or not isinstance(sources, list):
        raise ValueError("sources must be a nonempty list")
    if not isinstance(plan.get("seed"), int):
        raise ValueError("Specify an integer seed for a stable validation split")
    if not isinstance(plan.get("target_supervised_bytes"), int) or plan["target_supervised_bytes"] < 100:
        raise ValueError("target_supervised_bytes must be at least 100")
    if not isinstance(plan.get("validation_bytes_per_source"), int) or plan["validation_bytes_per_source"] < 1:
        raise ValueError("validation_bytes_per_source must be positive")
    if not isinstance(plan.get("max_record_bytes", 8192), int) or plan.get("max_record_bytes", 8192) < 512:
        raise ValueError("max_record_bytes must be at least 512")
    if not 0 < plan.get("validation_fraction", 0.02) < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    weights = {}
    for spec in sources:
        name = spec.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Source names must be lowercase identifiers")
        if name in weights or spec.get("kind") not in KINDS:
            raise ValueError(f"Duplicate source or unsupported adapter: {name}")
        weight = spec.get("weight")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Each source needs a positive weight")
        weights[name] = weight
        if not spec.get("license") or not isinstance(spec["license"], str):
            raise ValueError(f"{name}: explicit license metadata is required (use 'unverified' if unknown)")
        if not isinstance(spec.get("max_scanned"), int) or spec["max_scanned"] < 1:
            raise ValueError(f"{name}: max_scanned bounds the stream")
        if (not isinstance(spec.get("max_target_bytes", 512), int)
                or spec.get("max_target_bytes", 512) < 1):
            raise ValueError(f"{name}: max_target_bytes must be positive")
        if spec["kind"] != "synthetic" and bool(spec.get("path")) == bool(spec.get("repo")):
            raise ValueError(f"{name}: specify exactly one of path or repo")
        if spec.get("repo") and (
            not isinstance(spec.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"])
            or not spec.get("split")
        ):
            raise ValueError(f"{name}: HF sources require an immutable 40-character revision and split")
        if spec.get("repo") and spec["kind"] == "smoltalk" and not spec.get("config"):
            raise ValueError("SmolTalk requires a selected SFT config; never load the whole collection implicitly")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Source weights must sum to 1")
    return weights


def iter_source(spec, allow_network, on_error, *, split="train"):
    if spec["kind"] == "synthetic":
        from .data import Example, memory_example, operation_example

        for index in range(spec["max_scanned"]):
            subset = "train" if split == "train" else "val"
            if index % 3 == 0:
                example = memory_example(subset, index, spec.get("max_delay", 256))
            elif index % 3 == 1:
                example = operation_example(subset, index)
            else:
                rng = random.Random(index * 2309 + (0 if subset == "train" else 1))
                value = rng.randrange(0, 128) if subset == "train" else rng.randrange(128, 256)
                example = Example(
                    f"User: Bitwise invert the 8-bit hex byte {value:02x}. Reply with two hex digits.\nAssistant: ",
                    f"{value ^ 255:02x}", "operations",
                )
            yield {"question": example.prompt, "answer": example.answer}
        return
    if spec.get("path"):
        path = Path(spec["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        for _, row in iter_objects(path, on_error):
            yield row
        return
    if not allow_network:
        raise ValueError(f"{spec['name']}: HF streaming needs explicit --allow-network on RunPod")
    from datasets import load_dataset

    options = {"split": spec["split"], "revision": spec["revision"], "streaming": True}
    if spec.get("config"):
        options["name"] = spec["config"]
    yield from load_dataset(spec["repo"], **options)


def _digest(record):
    normalized = {key: value for key, value in record.items() if key != "censored"}
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).digest()


def prepare_mix(plan, output, *, allow_network=False):
    """Cap each source by supervised bytes; fail rather than silently changing ratios."""
    from trainnewideas.rap.data import sha256_file

    weights = validate_plan(plan)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Mix directory already exists: {output}; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_bytes = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix=output.name + ".preparing_", dir=output.parent) as tmp:
        staging = Path(tmp)
        connection = sqlite3.connect(staging / "dedup.sqlite")
        connection.execute("CREATE TABLE hashes (digest BLOB PRIMARY KEY)")
        reports = {}
        try:
            with ExitStack() as stack:
                for spec in plan["sources"]:
                    name = spec["name"]
                    quota = round(weights[name] * plan["target_supervised_bytes"])
                    val_quota = plan["validation_bytes_per_source"]
                    stats = {"train_records": 0, "validation_records": 0,
                             "train_supervised_bytes": 0, "validation_supervised_bytes": 0,
                             "train_raw_bytes": 0, "validation_raw_bytes": 0,
                             "scanned": 0, "validation_scanned": 0,
                             "duplicates": 0, "invalid": 0}
                    train_path = staging / f"{name}.train.jsonl"
                    val_path = staging / f"{name}.validation.jsonl"
                    streams = {
                        "train": stack.enter_context(train_path.open("w", encoding="utf-8")),
                        "validation": stack.enter_context(val_path.open("w", encoding="utf-8")),
                    }

                    def error(_path, _line, _reason):
                        stats["invalid"] += 1

                    def process(row, forced=None):
                        try:
                            record = adapt(spec, row, max_bytes=plan.get("max_record_bytes", 8192))
                            ids, mask = encode_record(record)
                            raw = sum(keep and byte < 256 for byte, keep in zip(ids[1:], mask[1:]))
                            if not raw or len(json.dumps(record, ensure_ascii=False).encode("utf-8")) > plan.get("max_record_bytes", 8192):
                                stats["invalid"] += 1
                                return
                            digest = _digest(record)
                            split = forced or ("validation" if int.from_bytes(
                                hashlib.sha256(str(plan["seed"]).encode() + digest).digest()[:8], "big"
                            ) / 2**64 < plan.get("validation_fraction", 0.02) else "train")
                            goal = val_quota if split == "validation" else quota
                            if stats[f"{split}_supervised_bytes"] >= goal:
                                return
                            try:
                                connection.execute("INSERT INTO hashes VALUES (?)", (digest,))
                            except sqlite3.IntegrityError:
                                stats["duplicates"] += 1
                                return
                            streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                            stats[f"{split}_records"] += 1
                            stats[f"{split}_supervised_bytes"] += raw
                            stats[f"{split}_raw_bytes"] += raw
                        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                            stats["invalid"] += 1
                            if stats["invalid"] <= 3:
                                print(f"{name}: skipped invalid record: {exc}", flush=True)

                    for row in iter_source(spec, allow_network, error):
                        stats["scanned"] += 1
                        if stats["scanned"] > spec["max_scanned"]:
                            break
                        process(row, "train" if spec["kind"] == "synthetic" else None)
                        if (stats["train_supervised_bytes"] >= quota
                                and stats["validation_supervised_bytes"] >= val_quota):
                            break
                    if spec["kind"] == "synthetic":
                        for row in iter_source(spec, allow_network, error, split="validation"):
                            stats["validation_scanned"] += 1
                            process(row, "validation")
                            if stats["validation_supervised_bytes"] >= val_quota:
                                break
                    if (stats["train_supervised_bytes"] < quota
                            or stats["validation_supervised_bytes"] < val_quota):
                        raise ValueError(f"{name}: insufficient unique supervised bytes after "
                                         f"{stats['scanned']} records: {stats}; increase max_scanned "
                                         "or choose a smaller byte budget")
                    reports[name] = dict(
                        **stats,
                        weight=weights[name], adapter=spec["kind"], repo=spec.get("repo"),
                        config=spec.get("config"), split=spec.get("split"),
                        revision=spec.get("revision"),
                        license=spec["license"], license_reviewed=spec.get("license_reviewed", False),
                        source_sha256=sha256_file(Path(spec["path"])) if spec.get("path") else None,
                    )
                for stream in streams.values():
                    stream.flush()
            connection.close()
            connection = None
            (staging / "dedup.sqlite").unlink()
            checksums = {p.name: sha256_file(p) for p in staging.glob("*.jsonl")}
            manifest = dict(format="rap-weighted-mix-v1", plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                            weights=weights, seed=plan["seed"], target_supervised_bytes=plan["target_supervised_bytes"],
                            validation_bytes_per_source=plan["validation_bytes_per_source"],
                            sources=reports, files_sha256=checksums)
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            staging.rename(output)
            print(json.dumps(manifest, indent=2), flush=True)
            return output
        finally:
            if connection is not None:
                connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON source configuration with pinned revisions")
    parser.add_argument("--out", required=True, help="New directory for canonical, bounded source JSONL")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly permit streaming HF datasets")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    prepare_mix(plan, args.out, allow_network=args.allow_network)


if __name__ == "__main__":
    main()
