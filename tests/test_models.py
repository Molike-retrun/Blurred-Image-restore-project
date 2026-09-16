import unittest

import torch

from models import build_model


class ModelSmokeTests(unittest.TestCase):
    def test_supported_models_preserve_shape(self):
        x = torch.rand(2, 3, 64, 64)
        for name in ("simple", "residual", "unet"):
            with self.subTest(model=name):
                y = build_model(name)(x)
                self.assertEqual(y.shape, x.shape)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            build_model("unknown")


if __name__ == "__main__":
    unittest.main()
