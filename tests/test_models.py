import unittest

import torch
from src.scene_classifier.models import build_model


class ModelShapeTests(unittest.TestCase):
    def test_starter_cnn_output_shape(self):
        model = build_model(
            {"name": "starter_cnn", "pretrained": False}, num_classes=16
        )
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 16))

    def test_resnet18_scratch_output_shape(self):
        model = build_model(
            {
                "name": "resnet18_scratch",
                "pretrained": False,
                "dropout": 0.3,
            },
            num_classes=16,
        )
        model.eval()
        self.assertEqual(model(torch.randn(1, 3, 128, 128)).shape, (1, 16))

    def test_checkpoint3_models_have_16_class_output(self):
        names = [
            "densenet121_scratch",
            "efficientnet_b0_scratch",
            "mobilenet_v3_large_scratch",
            "resnext50_32x4d_scratch",
            "convnext_tiny_scratch",
        ]
        for name in names:
            with self.subTest(name=name):
                model = build_model(
                    {"name": name, "pretrained": False, "dropout": 0.3},
                    num_classes=16,
                )
                linear_layers = [
                    layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)
                ]
                self.assertEqual(linear_layers[-1].out_features, 16)


if __name__ == "__main__":
    unittest.main()
