import unittest

import torch
from src.scene_classifier.models import build_model


class ModelShapeTests(unittest.TestCase):
    def test_tnet_output_shape(self):
        model = build_model({"name": "tnet", "pretrained": False}, num_classes=16)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 16))

    def test_small_cnn_output_shape(self):
        model = build_model(
            {"name": "small_cnn", "pretrained": False, "dropout": 0.35},
            num_classes=16,
        )
        self.assertEqual(model(torch.randn(2, 3, 128, 128)).shape, (2, 16))

    def test_resnet_output_shape_without_download(self):
        model = build_model(
            {"name": "resnet18", "pretrained": True, "dropout": 0.2},
            num_classes=16,
            load_pretrained=False,
        )
        model.eval()
        self.assertEqual(model(torch.randn(1, 3, 224, 224)).shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
