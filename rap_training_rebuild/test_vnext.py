import unittest

import torch

from rap_training_rebuild.vnext import parse_args, validate
from trainnewideas.rap.config import ModelConfig
from trainnewideas.rap.model import RAP
from trainnewideas.rap.tokens import ASSISTANT, BOS, TURN_END, USER, VOCAB_SIZE


class VNextFlagsTests(unittest.TestCase):
    def test_fresh_head_and_retrieval_are_independent(self):
        flags = validate(parse_args(["--core", "fresh", "--head", "direct",
                                     "--retrieval", "random"]))
        self.assertEqual((flags["head"], flags["retrieval"]), ("direct", "random"))
        flags = validate(parse_args(["--core", "fresh", "--head", "expression",
                                     "--retrieval", "none", "--memory-readout", "off"]))
        self.assertEqual(flags["memory_readout"], "off")

    def test_original_baseline_is_separate(self):
        args = parse_args(["--core", "original", "--base", "rap_B_latest (1).pt",
                           "--head", "original"])
        self.assertEqual(validate(args)["primitive_mode"], "original")
        args = parse_args(["--core", "original", "--base", "rap_B_latest (1).pt",
                           "--head", "joint", "--memory-readout", "joint"])
        self.assertEqual(validate(args)["head"], "joint")

    def test_unavailable_crosses_are_rejected_before_training(self):
        for values in (
            ["--core", "fresh", "--primitive-mode", "original"],
            ["--core", "fresh", "--head", "original"],
            ["--core", "fresh", "--phase-clock", "learned"],
            ["--core", "fresh", "--expression-loss", "byte-delta"],
            ["--core", "original", "--base", "rap_B_latest (1).pt",
             "--retrieval", "ordered"],
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "unimplemented"):
                validate(parse_args(values))

    def test_fresh_direct_head_gradients_and_disabled_retrieval(self):
        cfg = ModelConfig(embedding_size=8, state_size=16, controller_size=16,
                          readout_size=16, expression_head="direct", retrieval="none",
                          future_horizons=[4], past_lags=[1, 4])
        model = RAP(cfg)
        ids = torch.tensor([[BOS, USER, 72, TURN_END, ASSISTANT, 104, 105]])
        output = model(ids)
        self.assertEqual(tuple(output["log_probs"].shape), (1, len(ids[0]), VOCAB_SIZE))
        self.assertIsNone(model.expression_gate)
        self.assertEqual(int(output["retrieval_hits"].sum()), 0)
        (-output["log_probs"][0, -1, 105]).backward()
        self.assertIsNotNone(model.readout[0].weight.grad)
        self.assertIsNotNone(model.prim_A.grad)


if __name__ == "__main__":
    unittest.main()
