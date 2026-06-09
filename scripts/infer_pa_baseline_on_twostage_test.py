"""Run PolarAnything author checkpoint on the twostagenet test split."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PretrainedConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.PolarControlnet import PolarControl  # noqa: E402
from model.utils import load_params, remove_module_prefix  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METRIC_NAMES = (
    "dolp_mae",
    "dolp_rmse",
    "cos_mae",
    "sin_mae",
    "cos_sin_vector_error",
    "weighted_aolp_error_deg",
    "high_dolp_aolp_error_deg",
    "dop_mae",
    "dop_rmse",
    "aop_mae_deg",
    "weighted_aop_mae_deg",
    "high_dop_aop_mae_deg",
)
CHANNEL_ORDER_CANDIDATES = {
    "A_[DoLP,cos2,sin2]": (0, 1, 2),
    "B_[DoLP,sin2,cos2]": (0, 2, 1),
    "C_[cos2,DoLP,sin2]": (1, 0, 2),
    "D_[sin2,DoLP,cos2]": (1, 2, 0),
    "E_[cos2,sin2,DoLP]": (2, 0, 1),
    "F_[sin2,cos2,DoLP]": (2, 1, 0),
}


class PolarControlTest(ControlNetModel):
    """Author inference wrapper from PolarAnything-main/infer.py."""

    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__(cross_attention_dim=768)
        self.controlnet = PolarControl(PretrainedConfig())
        load_params(self.controlnet, unet)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict | None = None,
        return_dict: bool = True,
        guess_mode: bool | None = None,
    ):
        timestep = timestep.reshape(1)
        out_down, out_mid = self.controlnet(
            out_vae_noise=sample,
            noise_step=timestep,
            out_encoder=encoder_hidden_states,
            condition=controlnet_cond,
        )
        if return_dict:
            return {"down_block_res_samples": out_down, "mid_block_res_sample": out_mid}
        return out_down, out_mid


class PATestSplitDataset(Dataset):
    """Load S0/RGB inputs and GT polarization in [DoLP, cos2, sin2] order."""

    def __init__(
        self,
        root_dir: str | Path | None,
        rgb_dir: str | Path | None,
        gt_dir: str | Path | None,
        image_size: int | None,
        preprocess_mode: str = "aligned256",
        normalize_mode: str = "fixed255",
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.rgb_dir = Path(rgb_dir) if rgb_dir is not None else self._resolve_rgb_dir()
        self.gt_dir = Path(gt_dir) if gt_dir is not None else self._resolve_gt_dir()
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.normalize_mode = normalize_mode

        if image_size is not None:
            if image_size <= 0:
                raise ValueError("image_size must be positive or None.")
            if image_size % 8 != 0:
                raise ValueError("image_size must be divisible by 8 for Stable Diffusion.")
        if preprocess_mode not in {"aligned256", "official"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")
        if normalize_mode not in {"fixed255", "image_max"}:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No matched samples found under {self.rgb_dir} and {self.gt_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rgb_path, gt_path, stem = self.samples[index]
        rgb, native_size, input_size = self._read_rgb_for_mode(rgb_path)
        gt = self._read_gt(gt_path)

        if self.preprocess_mode == "aligned256" and self.image_size is not None:
            rgb = self._resize_rgb(rgb, self.image_size)
            gt = self._resize_polar(gt, self.image_size)
            input_size = f"{self.image_size}x{self.image_size}"
        elif self.preprocess_mode == "aligned256":
            rgb, gt = self._resize_to_multiple_of_8(rgb, gt)
            input_size = f"{rgb.shape[-2]}x{rgb.shape[-1]}"
        elif self.image_size is not None:
            gt = self._resize_polar(gt, self.image_size)

        return {
            "rgb": rgb,
            "polar_gt": gt,
            "name": stem,
            "rgb_path": str(rgb_path),
            "gt_path": str(gt_path),
            "input_native_size": native_size,
            "input_size": input_size,
        }

    def _resolve_rgb_dir(self) -> Path:
        if self.root_dir is None:
            raise ValueError("Either root_dir or rgb_dir must be provided.")
        return self._resolve_subdir(self.root_dir, ("S0", "s0", "RGB", "rgb"))

    def _resolve_gt_dir(self) -> Path:
        if self.root_dir is None:
            raise ValueError("Either root_dir or gt_dir must be provided.")
        return self._resolve_subdir(self.root_dir, ("Polarization_Encoding",))

    @staticmethod
    def _resolve_subdir(root_dir: Path, candidates: tuple[str, ...]) -> Path:
        for candidate in candidates:
            direct = root_dir / candidate
            if direct.is_dir():
                return direct
        lower_candidates = {candidate.lower() for candidate in candidates}
        if root_dir.is_dir():
            for child in root_dir.iterdir():
                if child.is_dir() and child.name.lower() in lower_candidates:
                    return child
        raise FileNotFoundError(f"Could not find any of {candidates} under {root_dir}.")

    def _collect_samples(self) -> list[tuple[Path, Path, str]]:
        rgb_files = self._index_images_by_stem(self.rgb_dir)
        gt_files = self._index_images_by_stem(self.gt_dir)
        stems = sorted(set(rgb_files) & set(gt_files))
        return [(rgb_files[stem], gt_files[stem], stem) for stem in stems]

    @staticmethod
    def _index_images_by_stem(directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.setdefault(path.stem, path)
        return files

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array * 2.0 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _read_rgb_for_mode(self, path: Path) -> tuple[torch.Tensor, str, str]:
        if self.preprocess_mode == "official":
            rgb, native_size, input_size = official_preprocess_image_with_sizes(
                path,
                normalize_mode=self.normalize_mode,
            )
            return rgb, native_size, input_size

        if self.normalize_mode == "image_max":
            rgb = read_rgb_image_max(path)
        else:
            rgb = self._read_rgb(path)
        height, width = rgb.shape[-2:]
        size = f"{height}x{width}"
        return rgb, size, size

    @classmethod
    def _read_gt(cls, path: Path) -> torch.Tensor:
        encoded = iio.imread(path)
        if encoded.ndim == 2 or encoded.shape[-1] < 3:
            raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
        encoded_float = cls._to_unit_range(encoded[..., :3])
        dolp = encoded_float[..., 0]
        cos2 = encoded_float[..., 1] * 2.0 - 1.0
        sin2 = encoded_float[..., 2] * 2.0 - 1.0
        polar = np.stack((dolp, cos2, sin2), axis=0).astype(np.float32)
        return normalize_polar_tensor(torch.from_numpy(polar))

    @staticmethod
    def _to_unit_range(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint16:
            return array.astype(np.float32) / 65535.0
        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0

        array_float = array.astype(np.float32)
        finite = array_float[np.isfinite(array_float)]
        if finite.size == 0:
            return np.zeros_like(array_float, dtype=np.float32)
        min_value = float(finite.min())
        max_value = float(finite.max())
        if min_value >= 0.0 and max_value <= 1.0:
            return array_float
        if min_value >= 0.0 and max_value <= 255.0:
            return array_float / 255.0
        if min_value >= 0.0 and max_value <= 65535.0:
            return array_float / 65535.0
        return np.clip(array_float, 0.0, 1.0)

    @staticmethod
    def _resize_rgb(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
        array = rgb.permute(1, 2, 0).numpy()
        array = ((array + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        resized = resized * 2.0 - 1.0
        return torch.from_numpy(resized).permute(2, 0, 1).contiguous()

    @staticmethod
    def _resize_polar(polar: torch.Tensor, image_size: int) -> torch.Tensor:
        channels = []
        for channel in polar:
            image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            channels.append(np.asarray(image, dtype=np.float32))
        resized = torch.from_numpy(np.stack(channels, axis=0)).contiguous()
        return normalize_polar_tensor(resized)

    @classmethod
    def _resize_to_multiple_of_8(
        cls,
        rgb: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        target_height = (height // 8) * 8
        target_width = (width // 8) * 8
        if target_height <= 0 or target_width <= 0:
            raise ValueError(f"Image is too small for Stable Diffusion: {height}x{width}")
        if target_height == height and target_width == width:
            return rgb, gt

        rgb = cls._resize_rgb_hw(rgb, target_height, target_width)
        gt = cls._resize_polar_hw(gt, target_height, target_width)
        return rgb, gt

    @staticmethod
    def _resize_rgb_hw(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
        array = rgb.permute(1, 2, 0).numpy()
        array = ((array + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image = image.resize((width, height), Image.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        resized = resized * 2.0 - 1.0
        return torch.from_numpy(resized).permute(2, 0, 1).contiguous()

    @staticmethod
    def _resize_polar_hw(polar: torch.Tensor, height: int, width: int) -> torch.Tensor:
        channels = []
        for channel in polar:
            image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
            image = image.resize((width, height), Image.BILINEAR)
            channels.append(np.asarray(image, dtype=np.float32))
        resized = torch.from_numpy(np.stack(channels, axis=0)).contiguous()
        return normalize_polar_tensor(resized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer PolarAnything baseline on the twostagenet test split."
    )
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--rgb_dir", "--s0_dir", dest="rgb_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--output_dir", type=str, default="./pa_baseline_test_outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("aligned256", "official"),
        default="aligned256",
    )
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default=None,
        help="Defaults to fixed255 for aligned256 and image_max for official.",
    )
    parser.add_argument("--resize_output_to_gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vis_every", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--save_pred_png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_raw_outputs", action="store_true")
    parser.add_argument("--diagnose_channel_order", action="store_true")
    parser.add_argument("--fairness_check", action="store_true")
    parser.add_argument("--official_input_folder", type=str, default=None)
    parser.add_argument("--official_results_folder", type=str, default=None)
    parser.add_argument("--report_path", type=str, default="reports/pa_baseline_fairness_check_runtime.md")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_normalize_mode(preprocess_mode: str, normalize_mode: str | None) -> str:
    if normalize_mode is not None:
        return normalize_mode
    if preprocess_mode == "official":
        return "image_max"
    return "fixed255"


def build_dataloader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, int]:
    if args.preprocess_mode == "official" and args.batch_size != 1:
        raise ValueError("official preprocess_mode keeps native sizes; please use --batch_size 1.")
    dataset = PATestSplitDataset(
        root_dir=args.root_dir,
        rgb_dir=args.rgb_dir,
        gt_dir=args.gt_dir,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        normalize_mode=args.resolved_normalize_mode,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataloader, len(dataset)


def build_pipeline(args: argparse.Namespace, device: torch.device) -> StableDiffusionControlNetPipeline:
    checkpoint = resolve_pretrained_model_path(
        args.pretrained_model_name_or_path,
        args.hf_cache_dir,
    )
    from_pretrained_kwargs = {}
    if args.hf_cache_dir and not Path(checkpoint).expanduser().exists():
        from_pretrained_kwargs["cache_dir"] = args.hf_cache_dir

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint,
        subfolder="unet",
        **from_pretrained_kwargs,
    )
    controlnet = PolarControlTest(unet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        checkpoint,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        **from_pretrained_kwargs,
    )
    pipeline.unet.requires_grad_(False)
    pipeline.controlnet.requires_grad_(False)

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    validate_checkpoint_keys(checkpoint_data, args.checkpoint)
    unet_result = pipeline.unet.load_state_dict(
        remove_module_prefix(checkpoint_data["unet_state_dict"])
    )
    controlnet_result = pipeline.controlnet.controlnet.load_state_dict(
        remove_module_prefix(checkpoint_data["controlnet_state_dict"])
    )
    print_load_result("UNet", unet_result)
    print_load_result("ControlNet", controlnet_result)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def validate_checkpoint_keys(checkpoint_data: dict, checkpoint_path: str) -> None:
    required_keys = ("unet_state_dict", "controlnet_state_dict")
    missing_keys = [key for key in required_keys if key not in checkpoint_data]
    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing required keys: {', '.join(missing_keys)}"
        )
    print(
        "Loaded PA checkpoint keys: "
        + ", ".join(sorted(str(key) for key in checkpoint_data.keys())),
        flush=True,
    )


def print_load_result(name: str, result) -> None:
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    print(
        f"{name} load_state_dict: missing_keys={len(missing)}, "
        f"unexpected_keys={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"{name} missing keys: {missing}", flush=True)
    if unexpected:
        print(f"{name} unexpected keys: {unexpected}", flush=True)


def resolve_pretrained_model_path(
    model_name_or_path: str,
    hf_cache_dir: str | None,
) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)

    snapshot = find_local_hf_snapshot(model_name_or_path, hf_cache_dir)
    if snapshot is not None:
        print(f"Using local Hugging Face snapshot: {snapshot}", flush=True)
        return str(snapshot)
    return model_name_or_path


def find_local_hf_snapshot(repo_id: str, hf_cache_dir: str | None) -> Path | None:
    if "/" not in repo_id:
        return None

    repo_cache_name = "models--" + repo_id.replace("/", "--")
    candidates: list[Path] = []
    if hf_cache_dir:
        cache_dir = Path(hf_cache_dir).expanduser()
        candidates.append(cache_dir / "hub" / repo_cache_name)
        candidates.append(cache_dir / repo_cache_name)
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        candidates.append(Path(hf_hub_cache).expanduser() / repo_cache_name)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub" / repo_cache_name)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / repo_cache_name)

    for repo_cache_dir in candidates:
        snapshot_dir = choose_complete_snapshot(repo_cache_dir / "snapshots")
        if snapshot_dir is not None:
            return snapshot_dir
    return None


def choose_complete_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        if is_complete_sd_snapshot(snapshot):
            return snapshot
    return None


def is_complete_sd_snapshot(snapshot: Path) -> bool:
    required_files = (
        "unet/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    )
    return all((snapshot / path).is_file() for path in required_files)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def infer_one(
    pipeline: StableDiffusionControlNetPipeline,
    rgb: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
    num_inference_steps: int,
) -> torch.Tensor:
    height, width = rgb.shape[-2:]
    result = pipeline(
        "denoised polarized images",
        rgb.unsqueeze(0).to(device),
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        output_type="np",
        generator=generator,
    )
    image = result.images[0].astype(np.float32)
    raw = pipeline_output_to_raw_chw(image)
    return raw.cpu()


def run_pipeline_raw(
    pipeline: StableDiffusionControlNetPipeline,
    rgb: torch.Tensor,
    device: torch.device,
    seed: int,
    num_inference_steps: int,
) -> torch.Tensor:
    generator = make_generator(device, seed)
    return infer_one(
        pipeline=pipeline,
        rgb=rgb,
        device=device,
        generator=generator,
        num_inference_steps=num_inference_steps,
    )


def read_rgb_image_max(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    max_value = float(np.max(array))
    if max_value <= 0.0:
        raise ValueError(f"Input image has non-positive max value: {image_path}")
    array = array / max_value
    array = array * 2.0 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def official_preprocess_image_with_sizes(
    image_path: str | Path,
    normalize_mode: str = "image_max",
) -> tuple[torch.Tensor, str, str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read official input image: {image_path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    native_height, native_width = image.shape[:2]
    height, width = (native_height // 8) * 8, (native_width // 8) * 8
    if height <= 0 or width <= 0:
        raise ValueError(f"Official input image is too small: {image_path}")
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32)
    if normalize_mode == "image_max":
        max_value = float(np.max(image))
        if max_value <= 0.0:
            raise ValueError(f"Official input image has non-positive max value: {image_path}")
        image = image / max_value
    elif normalize_mode == "fixed255":
        image = image / 255.0
    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor_image * 2.0 - 1.0, f"{native_height}x{native_width}", f"{height}x{width}"


def official_preprocess_image(image_path: str | Path) -> torch.Tensor:
    tensor_image, _, _ = official_preprocess_image_with_sizes(image_path, normalize_mode="image_max")
    return tensor_image


def current_preprocess_image(
    image_path: str | Path,
    image_size: int | None,
    preprocess_mode: str = "aligned256",
    normalize_mode: str = "fixed255",
) -> torch.Tensor:
    if preprocess_mode == "official":
        rgb, _, _ = official_preprocess_image_with_sizes(
            image_path,
            normalize_mode=normalize_mode,
        )
        return rgb
    if normalize_mode == "image_max":
        rgb = read_rgb_image_max(image_path)
    else:
        rgb = PATestSplitDataset._read_rgb(Path(image_path))
    if image_size is not None:
        return PATestSplitDataset._resize_rgb(rgb, image_size)
    height, width = rgb.shape[-2:]
    target_height = (height // 8) * 8
    target_width = (width // 8) * 8
    if target_height == height and target_width == width:
        return rgb
    return PATestSplitDataset._resize_rgb_hw(rgb, target_height, target_width)


def pipeline_output_to_raw_chw(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected pipeline RGB output with at least 3 channels, got {image.shape}.")
    return torch.from_numpy(image[..., :3].transpose(2, 0, 1).copy()).float()


def pipeline_output_to_polar(image: np.ndarray) -> torch.Tensor:
    raw = pipeline_output_to_raw_chw(image)
    return interpret_raw_output(raw, CHANNEL_ORDER_CANDIDATES["A_[DoLP,cos2,sin2]"])


def interpret_raw_output(raw: torch.Tensor, order: tuple[int, int, int]) -> torch.Tensor:
    dolp = map_dolp_channel(raw[order[0]])
    cos2 = map_angle_channel(raw[order[1]])
    sin2 = map_angle_channel(raw[order[2]])
    polar = torch.stack((dolp, cos2, sin2), dim=0).float()
    return normalize_polar_tensor(polar)


def map_dolp_channel(channel: torch.Tensor) -> torch.Tensor:
    min_value = float(channel.detach().min())
    max_value = float(channel.detach().max())
    if min_value >= -1.1 and max_value <= 1.1 and min_value < -0.05:
        return ((channel + 1.0) * 0.5).clamp(0.0, 1.0)
    return channel.clamp(0.0, 1.0)


def map_angle_channel(channel: torch.Tensor) -> torch.Tensor:
    min_value = float(channel.detach().min())
    max_value = float(channel.detach().max())
    if min_value >= -1.1 and max_value <= 1.1 and min_value < -0.05:
        return channel.clamp(-1.0, 1.0)
    return (channel * 2.0 - 1.0).clamp(-1.0, 1.0)


def normalize_polar_tensor(polar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dolp = polar[0:1].clamp(0.0, 1.0)
    cos_sin = polar[1:3]
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=0, keepdim=True) + eps)
    cos_sin = cos_sin / norm
    return torch.cat((dolp, cos_sin), dim=0).contiguous()


def resize_prediction_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred, False
    resized = torch.nn.functional.interpolate(
        pred.unsqueeze(0),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return normalize_polar_tensor(resized), True


def resize_prediction_to_gt_if_requested(
    pred: torch.Tensor,
    gt: torch.Tensor,
    resize_output_to_gt: bool,
) -> tuple[torch.Tensor, bool]:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred, False
    if not resize_output_to_gt:
        raise ValueError(
            "Prediction and GT sizes differ. Use --resize_output_to_gt for metrics "
            f"or align sizes manually. pred={tuple(pred.shape[-2:])}, gt={tuple(gt.shape[-2:])}"
        )
    return resize_prediction_to_gt(pred, gt)


def save_polar_encoding_png(path: Path, polar: torch.Tensor) -> None:
    """Save uint16 PA-style RGB semantics [DoLP, cos2, sin2]."""
    polar = polar.detach().float().cpu()
    dolp = polar[0].clamp(0.0, 1.0)
    cos2 = ((polar[1].clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    sin2 = ((polar[2].clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    encoded_rgb = torch.stack((dolp, cos2, sin2), dim=-1).numpy()
    encoded_uint16 = (encoded_rgb * 65535.0).round().clip(0, 65535).astype(np.uint16)
    encoded_bgr = cv2.cvtColor(encoded_uint16, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), encoded_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write polarization encoding PNG: {path}")


def save_pred_png(path: Path, pred: torch.Tensor) -> None:
    save_polar_encoding_png(path, pred)


def compute_metric_dict(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    dolp_error = (pred[0] - gt[0]).abs()
    cos_error = (pred[1] - gt[1]).abs()
    sin_error = (pred[2] - gt[2]).abs()
    vector_error = torch.sqrt((pred[1] - gt[1]) ** 2 + (pred[2] - gt[2]) ** 2)

    dot = pred[1] * gt[1] + pred[2] * gt[2]
    cross = pred[2] * gt[1] - pred[1] * gt[2]
    err_2aolp = torch.atan2(cross.abs(), dot)
    err_aolp_deg = err_2aolp * 0.5 * (180.0 / math.pi)
    reliability = torch.clamp((gt[0] - 0.03) / (0.15 - 0.03), 0.0, 1.0)
    high_mask = gt[0] > 0.15
    dolp_mae = float(dolp_error.mean())
    dolp_rmse = float(torch.sqrt(torch.mean((pred[0] - gt[0]) ** 2)))
    weighted_aop_mae_deg = weighted_mean(err_aolp_deg, reliability)
    high_dop_aop_mae_deg = masked_mean(err_aolp_deg, high_mask)

    return {
        "dolp_mae": dolp_mae,
        "dolp_rmse": dolp_rmse,
        "cos_mae": float(cos_error.mean()),
        "sin_mae": float(sin_error.mean()),
        "cos_sin_vector_error": float(vector_error.mean()),
        "weighted_aolp_error_deg": weighted_aop_mae_deg,
        "high_dolp_aolp_error_deg": high_dop_aop_mae_deg,
        "dop_mae": dolp_mae,
        "dop_rmse": dolp_rmse,
        "aop_mae_deg": float(err_aolp_deg.mean()),
        "weighted_aop_mae_deg": weighted_aop_mae_deg,
        "high_dop_aop_mae_deg": high_dop_aop_mae_deg,
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        return float("nan")
    return float((values * weights).sum() / weight_sum)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(values[mask].mean())


def tensor_to_rgb_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image + 1.0) * 0.5, 0.0, 1.0)


def to_display_map(tensor: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    min_value, max_value = value_range
    return np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)


def polar_to_aolp(polar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.atan2(polar[2], polar[1])


def polar_to_aolp_deg(polar: torch.Tensor | np.ndarray) -> np.ndarray:
    array = polar.detach().float().cpu().numpy() if isinstance(polar, torch.Tensor) else np.asarray(polar)
    if array.ndim != 3:
        raise ValueError(f"Expected polar shape [3,H,W] or [H,W,3], got {array.shape}.")
    if array.shape[0] == 3:
        polar_chw = array
    elif array.shape[-1] == 3:
        polar_chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected polar shape [3,H,W] or [H,W,3], got {array.shape}.")
    aolp_rad = 0.5 * np.arctan2(polar_chw[2], polar_chw[1])
    return np.clip(aolp_rad * (180.0 / math.pi), -90.0, 90.0)


def aolp_to_display(aolp: torch.Tensor) -> np.ndarray:
    return np.clip((aolp.detach().cpu().numpy() + math.pi / 2.0) / math.pi, 0.0, 1.0)


def compute_aolp_error_deg_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dot = pred[1] * gt[1] + pred[2] * gt[2]
    cross = pred[2] * gt[1] - pred[1] * gt[2]
    return torch.atan2(cross.abs(), dot) * 0.5 * (180.0 / math.pi)


def polar_encoding_display(polar: torch.Tensor) -> np.ndarray:
    polar = polar.detach().cpu()
    red = polar[0].clamp(0.0, 1.0).numpy()
    green = (polar[1].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5)
    blue = (polar[2].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5)
    return np.stack((red, green, blue), axis=-1)


def save_paper_aop_dop_vis(
    rgb_or_none: torch.Tensor | np.ndarray | None,
    gt_polar: torch.Tensor | np.ndarray,
    pred_polar: torch.Tensor | np.ndarray,
    save_path: Path,
) -> None:
    del rgb_or_none
    gt_array = gt_polar.detach().float().cpu().numpy() if isinstance(gt_polar, torch.Tensor) else np.asarray(gt_polar)
    pred_array = pred_polar.detach().float().cpu().numpy() if isinstance(pred_polar, torch.Tensor) else np.asarray(pred_polar)
    if gt_array.ndim == 3 and gt_array.shape[-1] == 3:
        gt_array = np.moveaxis(gt_array, -1, 0)
    if pred_array.ndim == 3 and pred_array.shape[-1] == 3:
        pred_array = np.moveaxis(pred_array, -1, 0)

    panels = [
        ("Captured AoLP", polar_to_aolp_deg(gt_array), "hsv", -90.0, 90.0, [-90.0, 0.0, 90.0]),
        ("Captured DoLP", np.clip(gt_array[0], 0.0, 1.0), "GnBu", 0.0, 1.0, [0.0, 1.0]),
        ("Gen. AoLP", polar_to_aolp_deg(pred_array), "hsv", -90.0, 90.0, [-90.0, 0.0, 90.0]),
        ("Gen. DoLP", np.clip(pred_array[0], 0.0, 1.0), "GnBu", 0.0, 1.0, [0.0, 1.0]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.patches.extend(
        [
            plt.Rectangle((0.0, 0.5), 1.0, 0.5, transform=fig.transFigure, color="#fff0f4", zorder=-1),
            plt.Rectangle((0.0, 0.0), 1.0, 0.5, transform=fig.transFigure, color="#eefcff", zorder=-1),
        ]
    )
    for axis, (title, image, cmap, vmin, vmax, ticks) in zip(axes.flat, panels):
        axis.set_facecolor("#fff0f4" if "Captured" in title else "#eefcff")
        im = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
        colorbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_ticks(ticks)
        if "AoLP" in title:
            colorbar.set_ticklabels([r"$-90^\circ$", r"$0^\circ$", r"$90^\circ$"])
        else:
            colorbar.set_ticklabels(["0", "1"])
        colorbar.ax.tick_params(labelsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_visualization(
    rgb: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    path: Path,
) -> None:
    rgb_image = tensor_to_rgb_image(rgb)
    dolp_error = (pred[0] - gt[0]).abs()
    aolp_error = compute_aolp_error_deg_map(pred, gt)

    panels = [
        ("RGB / S0", rgb_image, None),
        ("GT DoLP", gt[0].numpy(), (0.0, 1.0)),
        ("PA pred DoLP", pred[0].numpy(), (0.0, 1.0)),
        ("DoLP error", dolp_error.numpy(), (0.0, 1.0)),
        ("GT AoLP", aolp_to_display(polar_to_aolp(gt)), (0.0, 1.0)),
        ("PA pred AoLP", aolp_to_display(polar_to_aolp(pred)), (0.0, 1.0)),
        ("AoLP error", aolp_error.numpy(), (0.0, 90.0)),
        ("GT Polarization Encoding", polar_encoding_display(gt), None),
        ("PA pred Polarization Encoding", polar_encoding_display(pred), None),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, (title, image, value_range) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            vmin, vmax = value_range if value_range is not None else (None, None)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["name"]
    fieldnames.extend(METRIC_NAMES)
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        values = np.array([float(row[metric_name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[metric_name] = float(finite.mean()) if finite.size else float("nan")
    return summary


def summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        values = np.array([float(row[metric_name]) for row in metric_dicts], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[metric_name] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_channel_order_diagnosis(
    path: Path,
    diagnosis_metrics: dict[str, list[dict[str, float]]],
) -> None:
    summaries = {
        name: summarize_metric_dicts(rows)
        for name, rows in diagnosis_metrics.items()
        if rows
    }
    if not summaries:
        path.write_text("No channel order diagnosis samples were processed.\n", encoding="utf-8")
        return

    best_dolp = min(summaries, key=lambda name: summaries[name]["dolp_mae"])
    best_aolp = min(summaries, key=lambda name: summaries[name]["weighted_aolp_error_deg"])
    best_vector = min(summaries, key=lambda name: summaries[name]["cos_sin_vector_error"])

    lines = [
        "PolarAnything raw output channel-order diagnosis",
        "",
        f"Best DoLP MAE: {best_dolp}",
        f"Best weighted AoLP error: {best_aolp}",
        f"Best vector error: {best_vector}",
        "",
        "Mean metrics by interpretation:",
    ]
    for name in CHANNEL_ORDER_CANDIDATES:
        if name not in summaries:
            continue
        lines.append("")
        lines.append(name)
        for metric_name in METRIC_NAMES:
            lines.append(f"  {metric_name}: {format_float(summaries[name][metric_name])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fairness_check(
    args: argparse.Namespace,
    pipeline: StableDiffusionControlNetPipeline,
    device: torch.device,
) -> None:
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.official_input_folder) if args.official_input_folder else None
    if input_dir is None:
        if args.rgb_dir:
            input_dir = Path(args.rgb_dir)
        elif args.root_dir:
            input_dir = PATestSplitDataset(
                root_dir=args.root_dir,
                rgb_dir=None,
                gt_dir=args.gt_dir,
                image_size=args.image_size,
                preprocess_mode=args.preprocess_mode,
                normalize_mode=args.resolved_normalize_mode,
            ).rgb_dir
        else:
            raise ValueError("fairness_check needs root_dir, rgb_dir, or official_input_folder.")

    image_files = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:3]
    if not image_files:
        raise RuntimeError(f"No input images found for fairness_check under {input_dir}.")

    results_dir = (
        Path(args.official_results_folder)
        if args.official_results_folder
        else report_path.parent / "pa_fairness_check"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PA baseline runtime fairness check",
        "",
        f"checkpoint: {args.checkpoint}",
        f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}",
        f"official_input_folder: {input_dir}",
        f"preprocess_mode: {args.preprocess_mode}",
        f"normalize_mode: {args.resolved_normalize_mode}",
        f"num_inference_steps: {args.num_inference_steps}",
        f"seed: {args.seed}",
        "",
        "| sample | current_shape | official_shape | raw_mean_abs_diff | raw_max_abs_diff | final_mean_abs_diff | final_max_abs_diff | nearly_identical |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for index, image_path in enumerate(image_files):
        sample_seed = args.seed + index
        current_rgb = current_preprocess_image(
            image_path,
            args.image_size,
            preprocess_mode=args.preprocess_mode,
            normalize_mode=args.resolved_normalize_mode,
        )
        official_rgb = official_preprocess_image(image_path)

        current_raw = run_pipeline_raw(
            pipeline,
            current_rgb,
            device,
            sample_seed,
            args.num_inference_steps,
        )
        official_raw = run_pipeline_raw(
            pipeline,
            official_rgb,
            device,
            sample_seed,
            args.num_inference_steps,
        )

        current_pred = interpret_raw_output(
            current_raw,
            CHANNEL_ORDER_CANDIDATES["A_[DoLP,cos2,sin2]"],
        )
        official_pred = interpret_raw_output(
            official_raw,
            CHANNEL_ORDER_CANDIDATES["A_[DoLP,cos2,sin2]"],
        )

        raw_a, raw_b = align_pair(current_raw, official_raw)
        pred_a, pred_b = align_pair(current_pred, official_pred)
        raw_diff = (raw_a - raw_b).abs()
        final_diff = (pred_a - pred_b).abs()
        raw_mean = float(raw_diff.mean())
        raw_max = float(raw_diff.max())
        final_mean = float(final_diff.mean())
        final_max = float(final_diff.max())
        nearly_identical = raw_max < 1e-5 and final_max < 1e-5

        np.save(results_dir / f"{image_path.stem}_current_raw.npy", current_raw.numpy().astype(np.float32))
        np.save(results_dir / f"{image_path.stem}_official_raw.npy", official_raw.numpy().astype(np.float32))
        np.save(results_dir / f"{image_path.stem}_current_pred.npy", current_pred.numpy().astype(np.float32))
        np.save(results_dir / f"{image_path.stem}_official_pred.npy", official_pred.numpy().astype(np.float32))

        lines.append(
            f"| {image_path.name} | {tuple(current_raw.shape)} | {tuple(official_raw.shape)} | "
            f"{raw_mean:.8f} | {raw_max:.8f} | {final_mean:.8f} | {final_max:.8f} | "
            f"{nearly_identical} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- current_shape uses this script's selected preprocessing mode.",
            "- official_shape uses PolarAnything infer.py preprocessing: cv2 BGR->RGB, floor to multiple of 8, divide by image max, then [-1,1].",
            "- Raw/final arrays are saved under the fairness-check output folder for inspection.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote fairness check report to {report_path}", flush=True)


def align_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-2:] == b.shape[-2:]:
        return a, b
    b_resized = torch.nn.functional.interpolate(
        b.unsqueeze(0),
        size=a.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return a, b_resized


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    sample_count: int,
    summary: dict[str, float],
    resized_output_count: int,
    input_native_sizes: set[str],
    input_sizes: set[str],
) -> None:
    lines = [
        "PolarAnything author baseline test summary",
        f"checkpoint: {args.checkpoint}",
        f"test_root: {args.root_dir}",
        f"rgb_dir: {args.rgb_dir}",
        f"gt_dir: {args.gt_dir}",
        f"samples: {sample_count}",
        f"image_size: {args.image_size}",
        f"preprocess_mode: {args.preprocess_mode}",
        f"normalize_mode: {args.resolved_normalize_mode}",
        f"input_native_size: {format_size_set(input_native_sizes)}",
        f"input_size: {format_size_set(input_sizes)}",
        f"num_inference_steps: {args.num_inference_steps}",
        f"seed: {args.seed}",
        f"resize_output_to_gt: {args.resize_output_to_gt}",
        f"resized_outputs_to_gt: {resized_output_count}",
        f"output_resized_to_gt_count: {resized_output_count}",
        "saved_pred_encoding_png: True",
        "saved_gt_encoding_png: True",
        "encoding_png_dtype: uint16",
        "encoding_png_channel_order: [DoLP, cos2, sin2]",
        "aolp_visualization_range: [-90deg, 90deg]",
        "aolp_error_range_deg: [0, 90]",
        "",
        "Mean metrics:",
    ]
    for key in sorted(summary):
        lines.append(f"{key}: {format_float(summary[key])}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_size_set(values: set[str]) -> str:
    if not values:
        return "unknown"
    ordered = sorted(values)
    if len(ordered) <= 8:
        return ", ".join(ordered)
    return ", ".join(ordered[:8]) + f", ... ({len(ordered)} unique)"


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def run(args: argparse.Namespace) -> None:
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    args.resolved_normalize_mode = resolve_normalize_mode(
        args.preprocess_mode,
        args.normalize_mode,
    )
    if args.hf_cache_dir:
        os.environ.setdefault("HF_HOME", args.hf_cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(args.hf_cache_dir) / "hub"))

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    generator = make_generator(device, args.seed)

    output_dir = Path(args.output_dir)
    pred_npy_dir = output_dir / "pred_npy"
    pred_png_dir = output_dir / "pred_png"
    pred_encoding_png_dir = output_dir / "pred_encoding_png"
    gt_encoding_png_dir = output_dir / "gt_encoding_png"
    raw_npy_dir = output_dir / "raw_npy"
    vis_dir = output_dir / "vis"
    paper_vis_dir = output_dir / "paper_vis"
    pred_npy_dir.mkdir(parents=True, exist_ok=True)
    pred_encoding_png_dir.mkdir(parents=True, exist_ok=True)
    gt_encoding_png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred_png:
        pred_png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_outputs:
        raw_npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    paper_vis_dir.mkdir(parents=True, exist_ok=True)

    dataloader, dataset_size = build_dataloader(args, device)
    pipeline = build_pipeline(args, device)
    if args.fairness_check:
        run_fairness_check(args, pipeline, device)

    rows: list[dict[str, float | str]] = []
    diagnosis_metrics = {name: [] for name in CHANNEL_ORDER_CANDIDATES}
    processed = 0
    resized_output_count = 0
    input_native_sizes: set[str] = set()
    input_sizes: set[str] = set()
    with torch.no_grad():
        for batch in dataloader:
            names = batch["name"]
            rgb_batch = batch["rgb"]
            gt_batch = batch["polar_gt"]
            input_native_sizes.update(str(value) for value in batch["input_native_size"])
            input_sizes.update(str(value) for value in batch["input_size"])

            for offset, name in enumerate(names):
                sample_index = processed + offset
                rgb = rgb_batch[offset].cpu()
                gt = gt_batch[offset].cpu()
                raw = infer_one(
                    pipeline=pipeline,
                    rgb=rgb,
                    device=device,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                )
                if args.save_raw_outputs:
                    np.save(raw_npy_dir / f"{name}.npy", raw.numpy().astype(np.float32))

                if args.diagnose_channel_order:
                    for candidate_name, order in CHANNEL_ORDER_CANDIDATES.items():
                        candidate_pred = interpret_raw_output(raw, order)
                        candidate_pred, _ = resize_prediction_to_gt_if_requested(
                            candidate_pred,
                            gt,
                            args.resize_output_to_gt,
                        )
                        diagnosis_metrics[candidate_name].append(
                            compute_metric_dict(candidate_pred, gt)
                        )

                pred = interpret_raw_output(raw, CHANNEL_ORDER_CANDIDATES["A_[DoLP,cos2,sin2]"])
                pred, resized = resize_prediction_to_gt_if_requested(
                    pred,
                    gt,
                    args.resize_output_to_gt,
                )
                if resized:
                    resized_output_count += 1

                np.save(pred_npy_dir / f"{name}.npy", pred.numpy().astype(np.float32))
                save_polar_encoding_png(pred_encoding_png_dir / f"{name}.png", pred)
                save_polar_encoding_png(gt_encoding_png_dir / f"{name}.png", gt)
                if args.save_pred_png:
                    save_pred_png(pred_png_dir / f"{name}.png", pred)

                row: dict[str, float | str] = {"name": name}
                row.update(compute_metric_dict(pred, gt))
                rows.append(row)

                save_paper_aop_dop_vis(
                    rgb_or_none=None,
                    gt_polar=gt,
                    pred_polar=pred,
                    save_path=paper_vis_dir / f"{name}_aop_dop_compare.png",
                )

                if args.vis_every > 0 and sample_index % args.vis_every == 0:
                    save_visualization(rgb, gt, pred, vis_dir / f"{name}.png")

            processed += len(names)
            print(f"processed {processed}/{dataset_size}", flush=True)

    write_metrics_csv(output_dir / "metrics.csv", rows)
    if args.diagnose_channel_order:
        write_channel_order_diagnosis(
            output_dir / "channel_order_diagnosis.txt",
            diagnosis_metrics,
        )
    summary = summarize_rows(rows)
    write_summary(
        summary_path=output_dir / "summary.txt",
        args=args,
        sample_count=len(rows),
        summary=summary,
        resized_output_count=resized_output_count,
        input_native_sizes=input_native_sizes,
        input_sizes=input_sizes,
    )
    print(f"Done. Wrote outputs to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
