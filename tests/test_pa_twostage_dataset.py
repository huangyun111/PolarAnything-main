from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from datasets.pa_twostage_dataset import PATwostageDataset, read_polar_encoding


class PATwostageDatasetTest(unittest.TestCase):
    def test_read_polar_encoding_uses_pil_rgb_order_dolp_cos_sin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            encoded = np.zeros((2, 2, 3), dtype=np.uint8)
            encoded[..., 0] = 128
            encoded[..., 1] = 255
            encoded[..., 2] = 0
            iio.imwrite(path, encoded)

            polar = read_polar_encoding(path)

        self.assertAlmostEqual(float(polar[0, 0, 0]), 128 / 255, places=5)
        self.assertAlmostEqual(float(polar[1, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(polar[2, 0, 0]), -1.0, places=5)

    def test_dataset_returns_training_target_and_physical_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rgb_dir = root / "S0"
            gt_dir = root / "Polarization_Encoding"
            rgb_dir.mkdir()
            gt_dir.mkdir()

            rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
            gt = np.zeros((4, 4, 3), dtype=np.uint8)
            gt[..., 0] = 255
            gt[..., 1] = 128
            gt[..., 2] = 255
            iio.imwrite(rgb_dir / "a.png", rgb)
            iio.imwrite(gt_dir / "a.png", gt)

            dataset = PATwostageDataset(root_dir=root, tokenizer=None, image_size=8)
            item = dataset[0]

        self.assertEqual(item["name"], "a")
        self.assertEqual(tuple(item["rgb"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["polarization"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["polar_gt"].shape), (3, 8, 8))
        self.assertAlmostEqual(float(item["polar_gt"][0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(item["polarization"][0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(item["polar_gt"][2, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(item["polarization"][2, 0, 0]), 1.0, places=4)

    def test_dataset_refuses_test_split_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "test"
            rgb_dir = root / "S0"
            gt_dir = root / "Polarization_Encoding"
            rgb_dir.mkdir(parents=True)
            gt_dir.mkdir()

            rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
            gt = np.zeros((8, 8, 3), dtype=np.uint8)
            gt[..., 0] = 255
            gt[..., 1] = 128
            gt[..., 2] = 255
            iio.imwrite(rgb_dir / "a.png", rgb)
            iio.imwrite(gt_dir / "a.png", gt)

            with self.assertRaisesRegex(ValueError, "test split"):
                PATwostageDataset(root_dir=root, tokenizer=None, image_size=8)


if __name__ == "__main__":
    unittest.main()
