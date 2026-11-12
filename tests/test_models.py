"""Architecture checks with random weights and synthetic tensors only."""
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.extra_modules.head import LQE
from ultralytics.utils import DEFAULT_CFG_DICT

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_lqe_channel_grouping(self):
        estimator = LQE(4, 64, 2, 16)
        # Capture the actual statistics entering the unchanged learned MLP.
        observed = []
        hook = estimator.reg_conf.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        logits = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1) / 10
        scores = torch.randn(1, 3, 1, 1)
        result = estimator(scores, logits)
        hook.remove()
        probabilities = logits.reshape(1, 16, 4, 1, 1).softmax(1)
        top = probabilities.topk(4, dim=1).values
        expected = torch.cat((top, top.mean(1, keepdim=True)), dim=1).reshape(1, 20, 1, 1)
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(result, scores, rtol=0, atol=0)
        corrected = logits.reshape(1, 4, 16, 1, 1).softmax(2).topk(4, dim=2).values
        self.assertFalse(torch.allclose(top.transpose(1, 2), corrected))

    def test_eight_ablations_build_and_infer_at_640(self):
        configs = sorted((ROOT / "configs/ablations").glob("*.yaml"))
        self.assertEqual(len(configs), 8)
        for config in configs:
            with self.subTest(config=config.name):
                model = DetectionModel(str(config), nc=10, verbose=False).eval()
                torch.testing.assert_close(model.stride, torch.tensor([8., 16., 32.]))
                with torch.inference_mode():
                    predictions, raw = model(torch.zeros(1, 3, 640, 640))
                self.assertEqual(tuple(predictions.shape), (1, 14, 8400))
                self.assertEqual([tuple(item.shape[-2:]) for item in raw], [(80, 80), (40, 40), (20, 20)])
                self.assertTrue(torch.isfinite(predictions).all())

    def test_full_model_loss_backward_and_serialization(self):
        torch.manual_seed(7)
        model = DetectionModel(str(ROOT / "configs/models/rdl-yolo.yaml"), nc=10, verbose=False)
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        model.train()
        batch = {"img": torch.rand(1, 3, 640, 640), "batch_idx": torch.tensor([0.]),
                 "cls": torch.tensor([[0.]]), "bboxes": torch.tensor([[.5, .5, .2, .2]])}
        loss, components = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(components).all())
        loss.sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        model.eval()
        sample = torch.rand(1, 3, 640, 640)
        with torch.inference_mode():
            expected = model(sample)[0]
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        # Only this locally generated, trusted object is unpickled.
        restored = torch.load(buffer, weights_only=False).eval()
        with torch.inference_mode():
            actual = restored(sample)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
