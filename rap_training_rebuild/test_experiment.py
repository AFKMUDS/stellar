import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from rap_training_rebuild.data import Example, batch, load_examples, memory_example
from rap_training_rebuild.experiment import Readout, forward, load_base, show_dialogue
from rap_training_rebuild.flags import CHOICES, validate_flags
from rap_training_rebuild.online_teacher import (
    QwenTeacher, acceptable, append_teacher_example, load_teacher_examples,
    non_personal_answer, objective_question, target_train_delay,
)


ROOT = Path(__file__).resolve().parent.parent


class ExperimentTests(unittest.TestCase):
    def test_vnext_flags_reject_unimplemented_paths(self):
        args = Namespace(variant="A", **{name: None for name in CHOICES})
        self.assertEqual(validate_flags(args)["core"], "original")
        args.core = "fresh"
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_flags(args)
        args.core = "original"
        args.expression_loss = "byte-delta"
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_flags(args)
        args.expression_loss = None
        args.variant = "B"
        args.head = "original"
        with self.assertRaisesRegex(ValueError, "only joint"):
            validate_flags(args)

    def test_qa_and_instruction_records_have_assistant_only_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            rows = [
                {"question": "What is 2+2?", "reference_answer": "four"},
                {"instruction": "Say hi", "output": "hello"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            samples = load_examples(str(path), "train", "conversation")
            self.assertTrue(samples)
            for example in samples:
                self.assertTrue(example.prompt.endswith(("Answer:\n", "Assistant: ")))
                _, _, mask = batch([example], 100, torch.device("cpu"))
                self.assertEqual(int(mask.sum()), len(example.answer.encode("utf-8")))

    def test_teacher_dialogue_prints_training_example_and_rap_reply(self):
        example = Example("User: Say hello.\nAssistant: ", "Hello.", "conversation")
        output = io.StringIO()
        with patch("rap_training_rebuild.experiment.generate", return_value=b"\thello"), redirect_stdout(output):
            show_dialogue(None, None, example, 16, "cpu")
        self.assertIn("Qwen answer: 'Hello.'", output.getvalue())
        self.assertIn("RAP greedy reply (16 bytes): '\\thello'", output.getvalue())

    def test_qwen_teacher_accepts_tensor_and_batch_encoding(self):
        class Encoding(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 2

            def __init__(self, encoded):
                self.encoded = encoded

            def apply_chat_template(self, *args, **kwargs):
                return self.encoded

            def decode(self, generated, skip_special_tokens):
                if generated.tolist() != [7]:
                    raise AssertionError("Decoder received prompt bytes instead of new output")
                return " answer "

        class Model:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.cat([kwargs["input_ids"], torch.tensor([[7]])], dim=1)

        ids = torch.tensor([[4, 5]])
        for encoded, expected in (
            (ids, torch.ones_like(ids)),
            (Encoding(input_ids=ids), torch.ones_like(ids)),
            (Encoding(input_ids=ids, attention_mask=torch.tensor([[0, 1]])), torch.tensor([[0, 1]])),
        ):
            teacher = object.__new__(QwenTeacher)
            teacher.device = "cpu"
            teacher.max_tokens = 13
            teacher.tokenizer = Tokenizer(encoded)
            teacher.model = Model()
            self.assertEqual(teacher.ask("hello"), "answer")
            torch.testing.assert_close(teacher.model.kwargs["attention_mask"], expected)
            self.assertEqual(teacher.model.kwargs["max_new_tokens"], 13)
            self.assertFalse(teacher.model.kwargs["do_sample"])

    def test_disjoint_memory_and_mask(self):
        train = memory_example("train", 1, 64)
        val = memory_example("val", 1, 96)
        self.assertNotEqual(train.answer, val.answer)
        x, y, mask = batch([train], 400, torch.device("cpu"))
        self.assertEqual(int(mask.sum()), len(train.answer.encode("utf-8")))
        self.assertEqual(y.shape, x.shape)
        self.assertNotIn(memory_example("dev", 0, 80).answer, (train.answer, val.answer))

    def test_stable_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            rows = [{"prompt": f"User: prompt {i}\nAssistant: ", "answer": "okay"} for i in range(40)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            train = load_examples(str(path), "train", "conversation")
            val = load_examples(str(path), "val", "conversation")
            self.assertEqual(len(train) + len(val), 40)
            self.assertFalse(set(train) & set(val))

    def test_online_teacher_does_not_use_validation_values(self):
        self.assertFalse(acceptable("A violin sounds lovely under the evening sky."))
        self.assertFalse(acceptable("User: Hello, Assistant: goodbye."))
        self.assertFalse(objective_question("What's the best song you've heard recently?"))
        self.assertFalse(objective_question("What is the genre of the song you like most?"))
        self.assertFalse(non_personal_answer("The song I like most is pop."))
        self.assertEqual(target_train_delay(160, 256), 256)
        self.assertEqual(target_train_delay(384, 1024), 1024)
        teacher = object.__new__(QwenTeacher)
        teacher.ask = lambda _: "Clouds drift slowly above a quiet distant coastline."
        with tempfile.TemporaryDirectory() as folder:
            seen = set()
            example = teacher.challenge(6, "memory", 256, 600, seen, focus_delay=160)
            self.assertIsNotNone(example)
            self.assertEqual(example.delay, 256)
            self.assertIn("Context:  Clouds drift", example.prompt)
            self.assertNotIn("Clouds", example.answer)
            path = Path(folder) / "teacher.jsonl"
            append_teacher_example(path, example)
            self.assertEqual(load_teacher_examples(path), [example])
            repeated = teacher.challenge(6, "memory", 256, 600, seen, focus_delay=160)
            self.assertTrue(repeated is None or repeated.prompt != example.prompt)
        teacher.ask = lambda instruction: (
            "How does sunlight help a tomato plant grow?"
            if "Write one short" in instruction else
            "Sunlight provides energy for photosynthesis, helping the plant grow."
        )
        conversation = teacher.challenge(1, "conversation", 256, 600, set())
        self.assertEqual(conversation.answer,
                         "Sunlight provides energy for photosynthesis, helping the plant grow.")

    @unittest.skipUnless((ROOT / "rap_B_latest (1).pt").exists(), "checkpoint not present")
    def test_stream_matches_native_forward_and_frozen_readout(self):
        torch.manual_seed(17)
        model = load_base(ROOT / "rap_B_latest (1).pt", "cpu")
        model.eval()
        x = torch.tensor([[ord(c) for c in "otter otter "], [ord(c) for c in "value value "]])
        with torch.no_grad():
            native = model(x)[0]
            actual = forward(model, None, x)
        torch.testing.assert_close(actual, native, atol=1e-5, rtol=1e-5)
        for param in model.parameters():
            param.requires_grad_(False)
        head = Readout(model.cfg)
        logits = forward(model, head, x)
        logits.sum().backward()
        self.assertIsNotNone(head.net[-1].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
