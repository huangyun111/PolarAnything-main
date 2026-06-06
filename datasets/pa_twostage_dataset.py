from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PROMPT = "denoised polarized images"


class PATwostageDataset(Dataset):
    """Read RGB/S0 inputs and Polarization_Encoding GT for PA clean training.

    The physical GT tensor is always [DoLP, cos(2AoLP), sin(2AoLP)]. The
    training target passed to the VAE is [2*DoLP-1, cos(2AoLP), sin(2AoLP)].
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        rgb_dir: str | Path | None = None,
        gt_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        tokenizer: Any | None = None,
        image_size: int | None = 256,
        resize_mode: str = "resize",
        random_crop: bool = False,
        normalize_mode: str = "fixed255",
        seed: int = 42,
        forbid_test_split: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser() if root_dir is not None else None
        self.rgb_dir = Path(rgb_dir).expanduser() if rgb_dir is not None else None
        self.gt_dir = Path(gt_dir).expanduser() if gt_dir is not None else None
        self.manifest_path = (
            Path(manifest_path).expanduser() if manifest_path is not None else None
        )
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.random_crop = random_crop
        self.normalize_mode = normalize_mode
        self.seed = seed
        self.forbid_test_split = forbid_test_split

        if self.image_size is not None:
            if self.image_size <= 0:
                raise ValueError("image_size must be positive or None.")
            if self.image_size % 8 != 0:
                raise ValueError("image_size must be divisible by 8 for Stable Diffusion.")
        if self.resize_mode not in {"resize", "center_crop", "none"}:
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")
        if self.random_crop and self.resize_mode != "center_crop":
            raise ValueError("random_crop is only supported with resize_mode='center_crop'.")
        if self.normalize_mode not in {"fixed255", "image_max", "dtype"}:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")

        if self.manifest_path is None:
            if self.root_dir is None and (self.rgb_dir is None or self.gt_dir is None):
                raise ValueError("Provide root_dir, rgb_dir/gt_dir, or manifest_path.")
            if self.rgb_dir is None:
                self.rgb_dir = resolve_subdir(self.root_dir, ("S0", "s0", "RGB", "rgb"))
            if self.gt_dir is None:
                self.gt_dir = resolve_subdir(self.root_dir, ("Polarization_Encoding",))

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError("No matched RGB/S0 and Polarization_Encoding samples found.")
        if self.forbid_test_split:
            reject_test_split_samples(self.samples)
        self.input_ids = make_input_ids(self.tokenizer)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rgb_path, gt_path, name = self.samples[index]
        rng = random.Random(self.seed + index)

        rgb = read_rgb_image(rgb_path, normalize_mode=self.normalize_mode)
        polar_gt = read_polar_encoding(gt_path)
        rgb, polar_gt = transform_pair(
            rgb=rgb,
            polar=polar_gt,
            image_size=self.image_size,
            resize_mode=self.resize_mode,
            random_crop=self.random_crop,
            rng=rng,
        )
        polar_gt = normalize_polar_tensor(polar_gt)
        polarization = physical_polar_to_vae_target(polar_gt)

        return {
            "rgb": rgb.contiguous(),
            "polarization": polarization.contiguous(),
            "polar_gt": polar_gt.contiguous(),
            "input_ids": self.input_ids.clone(),
            "name": name,
            "rgb_path": str(rgb_path),
            "gt_path": str(gt_path),
        }

    def _collect_samples(self) -> list[tuple[Path, Path, str]]:
        if self.manifest_path is not None:
            return read_manifest(self.manifest_path)

        assert self.rgb_dir is not None
        assert self.gt_dir is not None
        rgb_files = index_images_by_stem(self.rgb_dir)
        gt_files = index_images_by_stem(self.gt_dir)
        stems = sorted(set(rgb_files) & set(gt_files))
        return [(rgb_files[stem], gt_files[stem], stem) for stem in stems]


def resolve_subdir(root_dir: Path | None, candidates: tuple[str, ...]) -> Path:
    if root_dir is None:
        raise ValueError("root_dir is required to resolve dataset subdirectories.")
    for candidate in candidates:
        direct = root_dir / candidate
        if direct.is_dir():
            return direct
    lower_candidates = {candidate.lower() for candidate in candidates}
    for child in root_dir.iterdir():
        if child.is_dir() and child.name.lower() in lower_candidates:
            return child
    raise FileNotFoundError(f"Could not find any of {candidates} under {root_dir}.")


def index_images_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.setdefault(path.stem, path)
    return files


def read_manifest(path: Path) -> list[tuple[Path, Path, str]]:
    samples: list[tuple[Path, Path, str]] = []
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in {2, 3}:
                raise ValueError(
                    f"{path}:{line_number} must contain rgb_path gt_path [name]."
                )
            rgb_path = resolve_manifest_path(parts[0], base_dir)
            gt_path = resolve_manifest_path(parts[1], base_dir)
            name = parts[2] if len(parts) == 3 else rgb_path.stem
            samples.append((rgb_path, gt_path, name))
    return samples


def reject_test_split_samples(samples: list[tuple[Path, Path, str]]) -> None:
    for rgb_path, gt_path, _name in samples:
        for path in (rgb_path, gt_path):
            if "test" in {part.lower() for part in path.parts}:
                raise ValueError(
                    f"Refusing to use test split sample for clean PA training: {path}"
                )


def resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_rgb_image(path: str | Path, normalize_mode: str = "fixed255") -> torch.Tensor:
    array = iio.imread(path)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB/S0 image with at least one channel: {path}")
    array = array[..., :3].astype(np.float32)

    if normalize_mode == "image_max":
        max_value = float(np.nanmax(array))
        denom = max(max_value, 1.0)
    elif normalize_mode == "dtype":
        denom = dtype_max(iio.imread(path).dtype)
    else:
        denom = 255.0

    array = np.clip(array / denom, 0.0, 1.0)
    array = array * 2.0 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).float()


def read_polar_encoding(path: str | Path) -> torch.Tensor:
    encoded = iio.imread(path)
    if encoded.ndim != 3 or encoded.shape[-1] < 3:
        raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
    encoded = encoded[..., :3]
    unit = encoded.astype(np.float32) / dtype_max(encoded.dtype)
    dolp = unit[..., 0].clip(0.0, 1.0)
    cos2 = (unit[..., 1] * 2.0 - 1.0).clip(-1.0, 1.0)
    sin2 = (unit[..., 2] * 2.0 - 1.0).clip(-1.0, 1.0)
    return torch.from_numpy(np.stack((dolp, cos2, sin2), axis=0)).float()


def dtype_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def transform_pair(
    rgb: torch.Tensor,
    polar: torch.Tensor,
    image_size: int | None,
    resize_mode: str,
    random_crop: bool,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_size is None or resize_mode == "none":
        return rgb, polar

    if resize_mode == "resize":
        return resize_tensor(rgb, image_size, is_rgb=True), resize_tensor(
            polar, image_size, is_rgb=False
        )

    height, width = rgb.shape[-2:]
    if height < image_size or width < image_size:
        rgb = resize_min_side(rgb, image_size, is_rgb=True)
        polar = resize_min_side(polar, image_size, is_rgb=False)
        height, width = rgb.shape[-2:]
    if random_crop:
        top = rng.randint(0, height - image_size)
        left = rng.randint(0, width - image_size)
    else:
        top = (height - image_size) // 2
        left = (width - image_size) // 2
    return (
        rgb[:, top : top + image_size, left : left + image_size],
        polar[:, top : top + image_size, left : left + image_size],
    )


def resize_min_side(tensor: torch.Tensor, image_size: int, is_rgb: bool) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    scale = image_size / min(height, width)
    target_height = int(round(height * scale))
    target_width = int(round(width * scale))
    return resize_tensor_hw(tensor, target_height, target_width, is_rgb=is_rgb)


def resize_tensor(tensor: torch.Tensor, image_size: int, is_rgb: bool) -> torch.Tensor:
    return resize_tensor_hw(tensor, image_size, image_size, is_rgb=is_rgb)


def resize_tensor_hw(
    tensor: torch.Tensor,
    height: int,
    width: int,
    is_rgb: bool,
) -> torch.Tensor:
    channels = tensor.permute(1, 2, 0).numpy()
    if is_rgb:
        unit = ((channels + 1.0) * 0.5).clip(0.0, 1.0)
        image = Image.fromarray((unit * 255.0).round().astype(np.uint8), mode="RGB")
        resized = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = array * 2.0 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).float()

    resized_channels = []
    for channel in tensor:
        image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
        resized = image.resize((width, height), Image.BILINEAR)
        resized_channels.append(np.asarray(resized, dtype=np.float32))
    return torch.from_numpy(np.stack(resized_channels, axis=0)).float()


def normalize_polar_tensor(polar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dolp = polar[0:1].clamp(0.0, 1.0)
    vector = polar[1:3].clamp(-1.0, 1.0)
    norm = torch.sqrt((vector * vector).sum(dim=0, keepdim=True) + eps)
    vector = vector / norm
    return torch.cat((dolp, vector), dim=0)


def physical_polar_to_vae_target(polar: torch.Tensor) -> torch.Tensor:
    dolp = polar[0:1].clamp(0.0, 1.0) * 2.0 - 1.0
    vector = polar[1:3].clamp(-1.0, 1.0)
    return torch.cat((dolp, vector), dim=0)


def make_input_ids(tokenizer: Any | None) -> torch.Tensor:
    if tokenizer is None:
        return torch.zeros(77, dtype=torch.long)
    return tokenizer.batch_encode_plus(
        [PROMPT],
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.squeeze(0)
