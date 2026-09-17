import unittest

import torch
from src.scene_classifier.models import build_model


class ModelShapeTests(unittest.TestCase):
    def test_starter_cnn_output_shape(self):
        model = build_model(
            {"name": "starter_cnn", "pretrained": False}, num_classes=16
        )
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
