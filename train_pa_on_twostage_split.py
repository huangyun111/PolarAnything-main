from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import chain
from pathlib import Path

import matplotlib
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pa_twostage_dataset import PATwostageDataset  # noqa: E402
from model.PolarControlnet import PolarControl  # noqa: E402
from model.utils import load_params, print_model_size, remove_module_prefix  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a clean PolarAnything baseline on twostagenet train/val splits."
    )
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--train_root_dir", type=str, default=None)
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("aligned256", "official_train"),
        default="aligned256",
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--save_last_freq", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument(
        "--resize_mode",
        choices=("resize", "center_crop", "none"),
        default="resize",
    )
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max", "dtype"),
        default=None,
        help="Defaults to fixed255 for aligned256 and image_max for official_train.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--save_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save optimizer state in checkpoints. Disabled by default to avoid very large PA checkpoints.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_normalize_mode(preprocess_mode: str, normalize_mode: str | None) -> str:
    if normalize_mode is not None:
        return normalize_mode
    if preprocess_mode == "official_train":
        return "image_max"
    return "fixed255"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def apply_hf_cache_env(hf_cache_dir: str | None) -> None:
    if not hf_cache_dir:
        return
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_cache_dir).expanduser() / "hub"))


def build_models(
    pretrained_model_name_or_path: str,
    hf_cache_dir: str | None,
    enable_xformers: bool,
    device: torch.device,
) -> tuple[CLIPTextModel, CLIPTokenizer, AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PolarControl]:
    checkpoint = resolve_pretrained_model_path(pretrained_model_name_or_path, hf_cache_dir)
    from_pretrained_kwargs = {}
    if hf_cache_dir and not Path(checkpoint).expanduser().exists():
        from_pretrained_kwargs["cache_dir"] = hf_cache_dir

    encoder = CLIPTextModel.from_pretrained(
        checkpoint,
        subfolder="text_encoder",
        **from_pretrained_kwargs,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        checkpoint,
        subfolder="tokenizer",
        **from_pretrained_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(
        checkpoint,
        subfolder="vae",
        **from_pretrained_kwargs,
    )
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint,
        subfolder="unet",
        **from_pretrained_kwargs,
    )
    scheduler = DDPMScheduler.from_pretrained(
        checkpoint,
        subfolder="scheduler",
        **from_pretrained_kwargs,
    )
    controlnet = PolarControl(PretrainedConfig())
    load_params(controlnet, unet)

    if enable_xformers:
        enable_xformers_if_available(unet)
        for module in controlnet.modules():
            enable_xformers_if_available(module)

    vae.requires_grad_(False)
    encoder.requires_grad_(False)
    unet.requires_grad_(True)
    controlnet.requires_grad_(True)

    encoder.eval()
    vae.eval()
    unet.train()
    controlnet.train()

    encoder.to(device)
    vae.to(device)
    unet.to(device)
    controlnet.to(device)

    for name, module in (
        ("encoder", encoder),
        ("vae", vae),
        ("unet", unet),
        ("controlnet", controlnet),
    ):
        print_model_size(name, module)

    return encoder, tokenizer, vae, scheduler, unet, controlnet


def enable_xformers_if_available(module: torch.nn.Module) -> None:
    if hasattr(module, "enable_xformers_memory_efficient_attention"):
        try:
            module.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # xformers is optional in PA environments.
            print(f"xformers not enabled for {module.__class__.__name__}: {exc}", flush=True)


def resolve_pretrained_model_path(model_name_or_path: str, hf_cache_dir: str | None) -> str:
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
        snapshots_dir = repo_cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        snapshots = sorted(
            [path for path in snapshots_dir.iterdir() if is_complete_sd_snapshot(path)],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]
    return None


def is_complete_sd_snapshot(snapshot: Path) -> bool:
    required = (
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "unet/config.json",
        "scheduler/scheduler_config.json",
    )
    return snapshot.is_dir() and all((snapshot / item).exists() for item in required)


def build_dataset(
    root_dir: str | None,
    manifest_path: str | None,
    tokenizer: CLIPTokenizer,
    args: argparse.Namespace,
    split_name: str,
) -> PATwostageDataset:
    if root_dir is None and manifest_path is None:
        raise ValueError(f"Provide {split_name}_root_dir or {split_name}_manifest.")
    return PATwostageDataset(
        root_dir=root_dir,
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        resize_mode=args.resize_mode,
        random_crop=(args.preprocess_mode == "official_train" or args.random_crop)
        and split_name == "train",
        normalize_mode=args.resolved_normalize_mode,
        seed=args.seed,
        forbid_test_split=split_name == "train",
    )


def limit_dataset(dataset: PATwostageDataset, max_samples: int | None) -> torch.utils.data.Dataset:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None.")
    return Subset(dataset, range(min(max_samples, len(dataset))))


def build_dataloaders(
    tokenizer: CLIPTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_root_dir = args.train_root_dir
    val_root_dir = args.val_root_dir
    if args.root_dir:
        root = Path(args.root_dir).expanduser()
        train_root_dir = train_root_dir or str(root / "train_clean")
        val_root_dir = val_root_dir or str(root / "val_clean")

    train_dataset = build_dataset(
        root_dir=train_root_dir,
        manifest_path=args.train_manifest,
        tokenizer=tokenizer,
        args=args,
        split_name="train",
    )
    val_dataset = build_dataset(
        root_dir=val_root_dir,
        manifest_path=args.val_manifest,
        tokenizer=tokenizer,
        args=args,
        split_name="val",
    )
    train_dataset_limited = limit_dataset(train_dataset, args.max_train_samples)
    val_dataset_limited = limit_dataset(val_dataset, args.max_val_samples)

    train_loader = DataLoader(
        train_dataset_limited,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset_limited,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    return train_loader, val_loader, len(train_dataset_limited), len(val_dataset_limited)


def compute_noise_prediction_loss(
    batch: dict[str, torch.Tensor],
    encoder: CLIPTextModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    controlnet: PolarControl,
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    polarization = batch["polarization"].to(device)
    rgb = batch["rgb"].to(device)

    with torch.no_grad():
        encoder_hidden_states = encoder(input_ids)[0]
        latents = vae.encode(polarization).latent_dist.sample()
        latents = latents * getattr(vae.config, "scaling_factor", 0.18215)

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
    ).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    control_down, control_mid = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        condition=rgb,
    )
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=control_down,
        mid_block_additional_residual=control_mid,
    ).sample
    return torch.nn.functional.mse_loss(noise_pred.float(), noise.float())


@torch.no_grad()
def validate(
    val_loader: DataLoader,
    encoder: CLIPTextModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    controlnet: PolarControl,
    device: torch.device,
) -> float:
    unet.eval()
    controlnet.eval()
    losses: list[float] = []
    for batch in val_loader:
        loss = compute_noise_prediction_loss(
            batch=batch,
            encoder=encoder,
            vae=vae,
            scheduler=scheduler,
            unet=unet,
            controlnet=controlnet,
            device=device,
        )
        losses.append(float(loss.detach().cpu()))
    unet.train()
    controlnet.train()
    return float(sum(losses) / max(len(losses), 1))


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    unet: UNet2DConditionModel,
    controlnet: PolarControl,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    args: argparse.Namespace,
    include_optimizer: bool,
) -> None:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "unet_state_dict": unet.state_dict(),
        "controlnet_state_dict": controlnet.state_dict(),
        "best_val_loss": best_val_loss,
        "config": vars(args),
        "clean_baseline": True,
        "initialized_from_pa_final_model": False,
    }
    if include_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_resume_checkpoint(
    resume: str | None,
    output_dir: Path,
    unet: UNet2DConditionModel,
    controlnet: PolarControl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    if resume is None:
        return 0, 0, float("inf")
    resume_path = output_dir / "last.pth" if resume == "last" else Path(resume).expanduser()
    if resume_path.name == "PA_Final_Model.pth":
        raise ValueError("PA_Final_Model.pth is not allowed for clean-baseline resume.")
    checkpoint = torch.load(resume_path, map_location=device)
    unet.load_state_dict(remove_module_prefix(checkpoint["unet_state_dict"]))
    controlnet.load_state_dict(remove_module_prefix(checkpoint["controlnet_state_dict"]))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Resume checkpoint has no optimizer_state_dict; optimizer starts fresh.", flush=True)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    print(f"Resumed clean PA checkpoint: {resume_path}", flush=True)
    return start_epoch, global_step, best_val_loss


def prune_epoch_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("epoch_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_path in checkpoints[save_total_limit:]:
        old_path.unlink()


def save_validation_visual(batch: dict[str, torch.Tensor], output_path: Path) -> None:
    rgb = batch["rgb"][0].detach().cpu()
    gt = batch["polar_gt"][0].detach().cpu()
    rgb_img = ((rgb.permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    panels = [
        ("RGB/S0", rgb_img, None),
        ("GT DoLP", gt[0].numpy(), (0.0, 1.0)),
        ("GT cos2", gt[1].numpy(), (-1.0, 1.0)),
        ("GT sin2", gt[2].numpy(), (-1.0, 1.0)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, (title, image, value_range) in zip(axes, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            vmin, vmax = value_range if value_range is not None else (None, None)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_config(path: Path, args: argparse.Namespace, train_count: int, val_count: int) -> None:
    uses_test_for_validation = path_looks_like_test_split(args.val_root_dir) or path_looks_like_test_split(
        args.val_manifest
    )
    payload = {
        **vars(args),
        "train_samples": train_count,
        "val_samples": val_count,
        "uses_pa_final_model": False,
        "uses_test_for_training": False,
        "uses_test_for_validation_or_checkpoint_selection": uses_test_for_validation,
        "preprocess_mode": args.preprocess_mode,
        "crop_size": args.crop_size,
        "normalize_mode": args.resolved_normalize_mode,
        "train_random_crop": args.preprocess_mode == "official_train" or args.random_crop,
        "crop_strategy": (
            "official_train resizes the short side to crop_size only when needed, "
            "then random-crops train samples and center-crops validation samples."
            if args.preprocess_mode == "official_train"
            else "aligned256 uses image_size with the selected resize_mode."
        ),
        "trainable_modules": ["unet", "controlnet"],
        "frozen_modules": ["vae", "text_encoder"],
        "gt_channel_order": "[DoLP, cos(2AoLP), sin(2AoLP)]",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_looks_like_test_split(value: str | None) -> bool:
    if value is None:
        return False
    return "test" in {part.lower() for part in Path(value).expanduser().parts}


def log_line(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def train(args: argparse.Namespace) -> None:
    if args.pretrained_model_name_or_path.endswith("PA_Final_Model.pth"):
        raise ValueError("Use SD1.5 or another base model path, not PA_Final_Model.pth.")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.val_freq <= 0:
        raise ValueError("val_freq must be positive.")
    if args.save_last_freq <= 0:
        raise ValueError("save_last_freq must be positive.")
    if args.crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if args.crop_size % 8 != 0:
        raise ValueError("crop_size must be divisible by 8 for Stable Diffusion.")

    apply_hf_cache_env(args.hf_cache_dir)
    args.resolved_normalize_mode = resolve_normalize_mode(
        args.preprocess_mode,
        args.normalize_mode,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")

    encoder, tokenizer, vae, scheduler, unet, controlnet = build_models(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        hf_cache_dir=args.hf_cache_dir,
        enable_xformers=args.enable_xformers_memory_efficient_attention,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        chain(unet.parameters(), controlnet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    train_loader, val_loader, train_count, val_count = build_dataloaders(
        tokenizer=tokenizer,
        args=args,
        device=device,
    )
    write_config(output_dir / "config.json", args, train_count, val_count)
    log_line(log_path, f"train_samples={train_count} val_samples={val_count}")
    log_line(log_path, "uses_PA_Final_Model=False")
    log_line(log_path, "uses_test_for_training=False")
    log_line(
        log_path,
        "uses_test_for_validation_or_checkpoint_selection="
        f"{path_looks_like_test_split(args.val_root_dir) or path_looks_like_test_split(args.val_manifest)}",
    )
    log_line(log_path, "trainable_modules=unet,controlnet frozen_modules=vae,text_encoder")
    log_line(
        log_path,
        f"preprocess_mode={args.preprocess_mode} crop_size={args.crop_size} "
        f"image_size={args.image_size} normalize_mode={args.resolved_normalize_mode}",
    )
    log_line(
        log_path,
        f"val_freq={args.val_freq} save_last_freq={args.save_last_freq} "
        f"save_optimizer={args.save_optimizer}",
    )

    start_epoch, global_step, best_val_loss = load_resume_checkpoint(
        resume=args.resume,
        output_dir=output_dir,
        unet=unet,
        controlnet=controlnet,
        optimizer=optimizer,
        device=device,
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        unet.train()
        controlnet.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            loss = compute_noise_prediction_loss(
                batch=batch,
                encoder=encoder,
                vae=vae,
                scheduler=scheduler,
                unet=unet,
                controlnet=controlnet,
                device=device,
            )
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    list(chain(unet.parameters(), controlnet.parameters())),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_losses.append(float(loss.detach().cpu()))
            if args.log_freq > 0 and (step + 1) % args.log_freq == 0:
                log_line(
                    log_path,
                    f"epoch={epoch + 1} step={step + 1}/{len(train_loader)} "
                    f"loss={epoch_losses[-1]:.6f}",
                )

        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        should_validate = (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.num_epochs
        val_loss = None
        if should_validate:
            val_loss = validate(
                val_loader=val_loader,
                encoder=encoder,
                vae=vae,
                scheduler=scheduler,
                unet=unet,
                controlnet=controlnet,
                device=device,
            )
        elapsed_min = (time.time() - start_time) / 60.0
        if val_loss is None:
            log_line(
                log_path,
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"val_loss=skipped elapsed_min={elapsed_min:.2f}",
            )
        else:
            log_line(
                log_path,
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} elapsed_min={elapsed_min:.2f}",
            )

        should_save_last = (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.num_epochs
        if should_save_last:
            save_checkpoint(
                output_dir / "last.pth",
                epoch=epoch,
                global_step=global_step,
                unet=unet,
                controlnet=controlnet,
                optimizer=optimizer,
                best_val_loss=best_val_loss if val_loss is None else min(best_val_loss, val_loss),
                args=args,
                include_optimizer=args.save_optimizer,
            )
            log_line(log_path, f"last_saved epoch={epoch + 1}")

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best_val.pth",
                epoch=epoch,
                global_step=global_step,
                unet=unet,
                controlnet=controlnet,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                args=args,
                include_optimizer=args.save_optimizer,
            )
            log_line(log_path, f"best_val_updated epoch={epoch + 1} val_loss={val_loss:.6f}")

        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:04d}.pth",
                epoch=epoch,
                global_step=global_step,
                unet=unet,
                controlnet=controlnet,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                args=args,
                include_optimizer=args.save_optimizer,
            )
            prune_epoch_checkpoints(output_dir, args.save_total_limit)

        if should_validate:
            first_val_batch = next(iter(val_loader))
            save_validation_visual(first_val_batch, output_dir / "vis" / f"epoch_{epoch + 1:04d}.png")

    log_line(log_path, f"training_finished best_val_loss={best_val_loss:.6f}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
