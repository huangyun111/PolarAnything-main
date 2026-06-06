from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed train_clean/val_clean split for PA clean training."
    )
    parser.add_argument("--split_root", type=str, default=os.environ.get("SPLIT_ROOT"))
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_count", type=int, default=None)
    parser.add_argument("--output_mode", choices=("symlink", "manifest", "both"), default="both")
    parser.add_argument("--manifest_dir", type=str, default="splits")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_train_root(args: argparse.Namespace) -> Path:
    if args.train_dir:
        return Path(args.train_dir).expanduser()
    if not args.split_root:
        raise ValueError("Provide --split_root or set SPLIT_ROOT.")
    return Path(args.split_root).expanduser() / "train"


def resolve_split_root(args: argparse.Namespace, train_root: Path) -> Path:
    if args.split_root:
        return Path(args.split_root).expanduser()
    return train_root.parent


def resolve_subdir(root_dir: Path, candidates: tuple[str, ...]) -> Path:
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


def collect_samples(train_root: Path) -> tuple[list[tuple[str, Path, Path]], str, str]:
    rgb_dir = resolve_subdir(train_root, ("S0", "s0", "RGB", "rgb"))
    gt_dir = resolve_subdir(train_root, ("Polarization_Encoding",))
    rgb_files = index_images_by_stem(rgb_dir)
    gt_files = index_images_by_stem(gt_dir)
    stems = sorted(set(rgb_files) & set(gt_files))
    samples = [(stem, rgb_files[stem], gt_files[stem]) for stem in stems]
    return samples, rgb_dir.name, gt_dir.name


def split_samples(
    samples: list[tuple[str, Path, Path]],
    seed: int,
    val_fraction: float,
    val_count: int | None,
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]]]:
    if not samples:
        raise RuntimeError("No matched samples found in train split.")
    if val_count is None:
        if val_fraction <= 0.0 or val_fraction >= 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        val_count = max(1, int(round(len(samples) * val_fraction)))
    if val_count <= 0 or val_count >= len(samples):
        raise ValueError(
            f"val_count must be between 1 and len(samples)-1; got {val_count} for {len(samples)} samples."
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_stems = {sample[0] for sample in shuffled[:val_count]}
    train_clean = [sample for sample in samples if sample[0] not in val_stems]
    val_clean = [sample for sample in samples if sample[0] in val_stems]
    return train_clean, val_clean


def write_manifest(path: Path, samples: list[tuple[str, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{rgb_path.resolve().as_posix()} {gt_path.resolve().as_posix()} {stem}"
        for stem, rgb_path, gt_path in samples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_symlink_tree(
    output_root: Path,
    samples: list[tuple[str, Path, Path]],
    rgb_subdir_name: str,
    gt_subdir_name: str,
    force: bool,
) -> bool:
    rgb_out = output_root / rgb_subdir_name
    gt_out = output_root / gt_subdir_name
    if output_root.exists() and not force:
        existing = [path for path in output_root.rglob("*") if path.is_file()]
        if existing:
            raise FileExistsError(f"{output_root} already contains files. Use --force.")
    rgb_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    try:
        for _stem, rgb_path, gt_path in samples:
            link_file(rgb_path, rgb_out / rgb_path.name, force=force)
            link_file(gt_path, gt_out / gt_path.name, force=force)
    except OSError as exc:
        print(f"Symlink creation failed for {output_root}: {exc}")
        return False
    return True


def link_file(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            return
        destination.unlink()
    destination.symlink_to(source.resolve())


def run(args: argparse.Namespace) -> None:
    train_root = resolve_train_root(args)
    split_root = resolve_split_root(args, train_root)
    samples, rgb_subdir_name, gt_subdir_name = collect_samples(train_root)
    train_clean, val_clean = split_samples(
        samples=samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_count=args.val_count,
    )

    manifest_dir = Path(args.manifest_dir).expanduser()
    if args.output_mode in {"manifest", "both"}:
        write_manifest(manifest_dir / "pa_train_clean.txt", train_clean)
        write_manifest(manifest_dir / "pa_val_clean.txt", val_clean)

    if args.output_mode in {"symlink", "both"}:
        train_ok = create_symlink_tree(
            split_root / "train_clean",
            train_clean,
            rgb_subdir_name=rgb_subdir_name,
            gt_subdir_name=gt_subdir_name,
            force=args.force,
        )
        val_ok = create_symlink_tree(
            split_root / "val_clean",
            val_clean,
            rgb_subdir_name=rgb_subdir_name,
            gt_subdir_name=gt_subdir_name,
            force=args.force,
        )
        if not train_ok or not val_ok:
            print("Symlink split incomplete. Use the generated manifest files instead.")

    print(f"source_train: {train_root}")
    print(f"train_clean_count: {len(train_clean)}")
    print(f"val_clean_count: {len(val_clean)}")
    print("test split was not read, modified, or referenced.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
