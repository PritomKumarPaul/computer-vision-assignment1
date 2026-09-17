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

    def test_resnext_transfer_scopes(self):
        expected_trainable = {
            "all": {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"},
            "layer4_and_head": {"layer4", "fc"},
            "head_only": {"fc"},
        }
        for scope, expected_groups in expected_trainable.items():
            with self.subTest(scope=scope):
                model = build_model(
                    {
                        "name": "resnext50_32x4d_transfer",
                        "pretrained": True,
                        "weights": "IMAGENET1K_V2",
                        "trainable_scope": scope,
                        "dropout": 0.3,
                    },
                    num_classes=16,
                    load_pretrained=False,
                )
                actual_groups = {
                    name.split(".", maxsplit=1)[0]
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(actual_groups, expected_groups)
                self.assertEqual(model.fc[-1].out_features, 16)

    def test_checkpoint5_partial_and_head_only_scopes(self):
        cases = [
            (
                "resnet18_transfer", "IMAGENET1K_V1", "layer1", "layer4",
                {"fc.1.weight", "fc.1.bias"},
            ),
            (
                "densenet121_transfer", "IMAGENET1K_V1",
                "features.denseblock1", "features.denseblock4",
                {"classifier.1.weight", "classifier.1.bias"},
            ),
            (
                "efficientnet_b0_transfer", "IMAGENET1K_V1", "features.1",
                "features.7", {"classifier.1.weight", "classifier.1.bias"},
            ),
            (
                "convnext_tiny_transfer", "IMAGENET1K_V1", "features.1",
                "features.7", {"classifier.2.1.weight", "classifier.2.1.bias"},
            ),
        ]
        for name, weights, early_module, final_stage, head_parameters in cases:
            with self.subTest(name=name, scope="last_stage_and_head"):
                model = build_model(
                    {
                        "name": name,
                        "pretrained": True,
                        "weights": weights,
                        "trainable_scope": "last_stage_and_head",
                        "dropout": 0.3,
                    },
                    num_classes=16,
                    load_pretrained=False,
                )
                self.assertFalse(
                    next(model.get_submodule(early_module).parameters()).requires_grad
                )
                self.assertTrue(
                    next(model.get_submodule(final_stage).parameters()).requires_grad
                )

            with self.subTest(name=name, scope="head_only"):
                model = build_model(
                    {
                        "name": name,
                        "pretrained": True,
                        "weights": weights,
                        "trainable_scope": "head_only",
                        "dropout": 0.3,
                    },
                    num_classes=16,
                    load_pretrained=False,
                )
                actual = {
                    parameter_name
                    for parameter_name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(actual, head_parameters)


if __name__ == "__main__":
    unittest.main()
