import json
import tempfile
import unittest
from pathlib import Path

import torch

from rap_training_rebuild.mix import adapt, iter_source, prepare_mix, validate_plan
from trainnewideas.rap.config import Config, DataConfig, ModelConfig, TrainConfig, as_dict
from trainnewideas.rap.data import Corpus, StreamBatcher, encode_record, prepare_dataset
from trainnewideas.rap.engine import train as train_fresh
from trainnewideas.rap.runtime import load_payload
from trainnewideas.rap.tokens import EOS


class MixTests(unittest.TestCase):
    def test_adapters_and_censored_endings(self):
        text = adapt({"kind": "fineweb"}, {"text": "é" * 200}, max_bytes=300)
        self.assertTrue(text["censored"])
        self.assertNotEqual(encode_record(text)[0][-1], EOS)
        q = adapt({"kind": "natural"}, {"question": "Why?", "reference_answer": "yes"}, max_bytes=400)
        self.assertEqual(sum(encode_record(q)[1]), 4)
        spec = {"kind": "openthoughts", "max_target_bytes": 12}
        with self.assertRaisesRegex(ValueError, "explicit"):
            adapt(spec, {"problem": "Explain", "generated_solution": "trace only"}, max_bytes=400)
        answer = adapt(spec, {"problem": "Explain", "final_answer": "this is a long answer"}, max_bytes=400)
        self.assertTrue(answer["censored"])
        self.assertNotEqual(encode_record(answer)[0][-1], EOS)
        smol = adapt({"kind": "smoltalk"},
                     {"messages": [{"role": "user", "content": "Hi"},
                                   {"role": "assistant", "content": "<think>secret</think> Hello"}]},
                     max_bytes=400)
        self.assertEqual(smol["messages"][-1]["content"], "Hello")

    def test_network_requires_opt_in_and_pinned_revision(self):
        spec = dict(name="fineweb", kind="fineweb", repo="example/repo",
                    split="train", revision="bad", weight=1, license="unverified", max_scanned=20)
        plan = dict(seed=7, target_supervised_bytes=100, validation_bytes_per_source=2,
                    sources=[spec])
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_plan(plan)
        spec["revision"] = "a" * 40
        validate_plan(plan)
        with self.assertRaisesRegex(ValueError, "allow-network"):
            next(iter_source(spec, False, lambda *_: None))

    def test_mixture_is_disjoint_byte_weighted_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            natural = root / "natural.jsonl"
            natural.write_text("".join(json.dumps({"question": f"item {i}?", "reference_answer": f"value {i}"}) + "\n"
                                       for i in range(160)), encoding="utf-8")
            fineweb = root / "fineweb.jsonl"
            fineweb.write_text("".join(json.dumps({"text": f"Lesson {i}: " + "clear prose " * 7}) + "\n"
                                       for i in range(60)), encoding="utf-8")
            plan = {
                "seed": 2718, "validation_fraction": 0.4,
                "target_supervised_bytes": 500, "validation_bytes_per_source": 16,
                "max_record_bytes": 1200,
                "sources": [
                    dict(name="natural", kind="natural", path=str(natural),
                         weight=0.4, license="unverified", max_scanned=160),
                    dict(name="fineweb", kind="fineweb", path=str(fineweb),
                         weight=0.3, license="unverified", max_scanned=60),
                    dict(name="synthetic", kind="synthetic", weight=0.3,
                         license="generated-here", max_scanned=300, max_delay=64),
                ],
            }
            mixed = prepare_mix(plan, root / "mixed")
            manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["weights"]), {"natural", "fineweb", "synthetic"})
            cfg = DataConfig(source_weights=manifest["weights"])
            names = list(manifest["weights"])
            prepared = prepare_dataset([mixed / f"{n}.train.jsonl" for n in names],
                                       root / "prepared", cfg,
                                       validation_path=[mixed / f"{n}.validation.jsonl" for n in names])
            train, val = Corpus(prepared, "train"), Corpus(prepared, "validation")
            self.assertFalse(set(train.indices) & set(val.indices))
            for name in names:
                self.assertTrue(len(Corpus(prepared, "validation", source=name).indices))
            stream = StreamBatcher(train, 4, 32, 19)
            for _ in range(500):
                stream.next()
            counts = stream.source_supervised_bytes
            ratios = counts / counts.sum()
            for actual, expected in zip(ratios, (0.4, 0.3, 0.3)):
                self.assertLess(abs(actual - expected), 0.12)
            saved = stream.state_dict()
            expected = stream.next()
            restored = StreamBatcher(train, 4, 32, 20)
            restored.load_state_dict(saved)
            actual = restored.next()
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]))
            del train, val, stream, restored
            model_cfg = ModelConfig(embedding_size=8, state_size=16, controller_size=16,
                                    readout_size=16, future_horizons=[4], past_lags=[1, 4])
            train_cfg = TrainConfig(steps=2, batch_size=2, chunk_length=16,
                                    gradient_accumulation=1, device="cpu", precision="float32",
                                    evaluate_every=1, validation_target_tokens=8,
                                    validation_input_limit=1000, log_every=1,
                                    save_every=1, sample_every=0, snapshot_every=0,
                                    future_weight=0, past_weight=0,
                                    boundary_weight=0, stop_weight=0)
            cfg = Config(model_cfg, train_cfg, cfg)
            checkpoint, _ = train_fresh(cfg, prepared, root / "run", install_signal=False)
            latest = json.loads((root / "run" / "validation.jsonl").read_text().splitlines()[-1])
            self.assertEqual(set(latest["by_source"]), set(names))
            self.assertTrue(all("transition_entropy_bits" in row for row in latest["by_source"].values()))
            self.assertTrue(load_payload(checkpoint)["metrics"]["source_supervised_bytes"])
            cfg.train.steps = 3
            checkpoint, _ = train_fresh(cfg, prepared, root / "run",
                                        resume="auto", install_signal=False)
            self.assertEqual(load_payload(checkpoint)["step"], 3)
            config_path = root / "tiny.json"
            config_path.write_text(json.dumps(as_dict(cfg)), encoding="utf-8")
            from rap_training_rebuild.vnext import main as launch_stage
            flags = ["--core", "fresh", "--primitive-mode", "fresh",
                     "--mix", str(mixed), "--config", str(config_path),
                     "--prepared", str(root / "stage_prepared"),
                     "--run", str(root / "stage"), "--device", "cpu",
                     "--batch-size", "2", "--chunk-length", "16",
                     "--gradient-accumulation", "1", "--evaluate-every", "1",
                     "--validation-target-tokens", "8"]
            launch_stage(flags + ["--initialize-from", str(checkpoint), "--steps", "1"])
            self.assertEqual(load_payload(root / "stage" / "latest.pt")["step"], 1)
            initialization = json.loads((root / "stage" / "initialization.json").read_text())
            self.assertEqual(initialization["source_step"], 3)
            self.assertTrue(initialization["optimizer_and_stream_reset"])
            launch_stage(flags + ["--resume", "--steps", "2"])
            self.assertEqual(load_payload(root / "stage" / "latest.pt")["step"], 2)
            from rap_training_rebuild.compare_stage import compare
            from rap_training_rebuild.probes import evaluate_probes
            scores = compare([checkpoint, root / "stage" / "latest.pt"],
                             root / "stage_prepared", budget=8, device="cpu")
            self.assertEqual(len(scores), 2)
            self.assertIn("natural", scores[str(checkpoint)]["source_scores"])
            probes = evaluate_probes(root / "stage" / "latest.pt",
                                     mixed / "synthetic.validation.jsonl",
                                     device="cpu", count=1, max_new_bytes=3)
            self.assertIn("memory", probes)
            self.assertIn("byte_inverse", probes)
            with self.assertRaisesRegex(ValueError, "must differ"):
                launch_stage(flags + ["--initialize-from", str(root / "stage" / "latest.pt"),
                                      "--steps", "3"])
            if torch.cuda.is_available():
                gpu_flags = [
                    flag for flag in flags
                ]
                gpu_flags[gpu_flags.index("--run") + 1] = str(root / "gpu_stage")
                gpu_flags[gpu_flags.index("--device") + 1] = "cuda"
                gpu_flags[gpu_flags.index("--batch-size") + 1] = "8"
                launch_stage(gpu_flags + ["--initialize-from", str(checkpoint), "--steps", "1"])
                launch_stage(gpu_flags + ["--resume", "--steps", "2"])
                gpu_payload = load_payload(root / "gpu_stage" / "latest.pt")
                self.assertEqual(gpu_payload["step"], 2)
                self.assertGreater(gpu_payload["metrics"]["peak_vram_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
