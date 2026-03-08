#!/usr/bin/env python3
"""
nn-chain-explorer CLI entrypoint.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
VIZ_DIR = ROOT / "viz"
NOTEBOOKS_DIR = ROOT / "notebooks"


def _ensure_dirs():
    for d in (DATA_DIR, EMBEDDINGS_DIR, RESULTS_DIR, VIZ_DIR, NOTEBOOKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── setup-data ────────────────────────────────────────────────────────────────

def cmd_setup_data(args):
    dataset_name = getattr(args, "dataset", "cifar10")
    if dataset_name == "imagenet":
        _setup_imagenet(args)
    else:
        _setup_cifar10(args)


def _setup_cifar10(args):
    import random
    import shutil

    import torchvision
    from rich.table import Table

    _ensure_dirs()

    # Check existing images
    existing = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.jpeg")) + \
               list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.webp"))
    if existing and not args.force:
        console.print(
            f"[yellow]data/ already contains {len(existing)} image(s).[/yellow]\n"
            "Pass [bold]--force[/bold] to overwrite."
        )
        return

    if args.force and existing:
        console.print(f"[yellow]--force: removing existing data/...[/yellow]")
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    console.print("Downloading CIFAR-10 test split...")
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=str(ROOT / ".cifar_cache"), train=False, download=True
        )
    except Exception as e:
        if args.debug:
            raise
        console.print(
            f"[red]Download failed: {e}[/red]\n"
            "Manually place images in ./data/ as flat JPEGs or subfolders and re-run embed."
        )
        sys.exit(1)

    classes = dataset.classes  # 10 class names
    n_per_class = getattr(args, "n_per_class", None) or 100
    rng = random.Random(42)

    # Group indices by class
    class_indices: dict[int, list[int]] = {c: [] for c in range(len(classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    saved = []
    for cls_idx, cls_name in enumerate(classes):
        idxs = class_indices[cls_idx]
        sampled = rng.sample(idxs, min(n_per_class, len(idxs)))
        cls_dir = DATA_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j, dataset_idx in enumerate(sampled):
            pil_img, _ = dataset[dataset_idx]
            out_path = cls_dir / f"{j:04d}.jpg"
            try:
                pil_img.save(str(out_path), format="JPEG")
                saved.append(out_path)
            except OSError as e:
                console.print(f"[red]Write failed for {out_path}: {e}[/red]")
                sys.exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Class")
    table.add_column("Count", justify="right")
    table.add_column("Sample")
    for cls_name in classes:
        cls_dir = DATA_DIR / cls_name
        imgs = list(cls_dir.glob("*.jpg"))
        sample = str(imgs[0]) if imgs else ""
        table.add_row(cls_name, str(len(imgs)), sample)
    console.print(table)
    console.print(f"\n[green]✓[/green] {len(saved)} images saved to data/")


def _setup_imagenet(args):
    """
    Download a large-scale ImageNet dataset via HuggingFace datasets and save as JPEGs.

    Dataset options (controlled by --hf-dataset):
      - ILSVRC/imagenet-1k   : full 1.2M-image ImageNet. GATED — requires HF login +
                               accepting terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
      - zh-plus/tiny-imagenet: 100K images, 200 classes, 64x64px. UNGATED — works without login.
                               Good for testing the pipeline at scale before using full ImageNet.
    """
    import shutil

    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    _ensure_dirs()

    # Check existing images
    existing = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.jpeg")) + \
               list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.webp"))
    if existing and not args.force:
        console.print(
            f"[yellow]data/ already contains {len(existing)} image(s).[/yellow]\n"
            "Pass [bold]--force[/bold] to overwrite."
        )
        return

    if args.force and existing:
        console.print(f"[yellow]--force: removing existing data/...[/yellow]")
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        console.print(
            "[red]datasets library not found.[/red] "
            "Install with: [bold]pip install datasets[/bold]"
        )
        sys.exit(1)

    hf_dataset = getattr(args, "hf_dataset", "zh-plus/tiny-imagenet")
    n_per_class = getattr(args, "n_per_class", None)
    split = "train"

    # Dataset-specific guidance
    GATED_DATASETS = {"ILSVRC/imagenet-1k", "imagenet-1k", "timm/imagenet-1k-wds"}
    is_gated = hf_dataset in GATED_DATASETS

    console.print(
        f"Loading [bold]{hf_dataset}[/bold] (streaming) from HuggingFace Hub...\n"
        f"  split: {split}  |  n_per_class: {'all' if not n_per_class else n_per_class}"
    )

    if is_gated:
        console.print(
            "[yellow]Note:[/yellow] This is a gated dataset. You must:\n"
            f"  1. Accept terms at [link]https://huggingface.co/datasets/{hf_dataset}[/link]\n"
            "  2. Run [bold]huggingface-cli login[/bold] (or set HF_TOKEN env var)\n"
            "\n"
            "  Alternatively, use the ungated Tiny ImageNet (200 classes, 100K images):\n"
            "  [bold]python main.py setup-data --dataset imagenet --hf-dataset zh-plus/tiny-imagenet[/bold]"
        )
    else:
        console.print(
            f"[dim]Ungated dataset — no login required.[/dim]"
        )

    try:
        dataset = load_dataset(hf_dataset, split=split, streaming=True)
    except Exception as e:
        err = str(e)
        if args.debug:
            raise
        if "gated" in err.lower() or "authentication" in err.lower() or "401" in err:
            console.print(
                f"[red]Auth required for {hf_dataset}.[/red]\n"
                f"  1. Accept terms: https://huggingface.co/datasets/{hf_dataset}\n"
                "  2. Run: [bold]huggingface-cli login[/bold]\n"
                "\n"
                "  Or use the ungated alternative:\n"
                "  [bold]python main.py setup-data --dataset imagenet --hf-dataset zh-plus/tiny-imagenet[/bold]"
            )
        else:
            console.print(f"[red]Failed to load {hf_dataset}: {err}[/red]")
        sys.exit(1)

    # Detect label field and class names
    label_field = "label"
    image_field = "image"
    # tiny-imagenet uses 'label' for int and may have different structure; probe first sample
    try:
        features = dataset.features
        if "label" not in features and "class" in features:
            label_field = "class"
        if "image" not in features and "img" in features:
            image_field = "img"
    except Exception:
        pass

    class_counts: dict[int, int] = {}
    class_names: dict[int, str] = {}
    saved_total = 0

    def _get_class_name(label: int) -> str:
        if label in class_names:
            return class_names[label]
        try:
            feat = dataset.features[label_field]
            name = feat.int2str(label)
        except Exception:
            name = f"class_{label:04d}"
        class_names[label] = name
        return name

    console.print("Streaming and saving images...")

    with Progress(
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Saving...", total=None)

        for sample in dataset:
            label = sample[label_field]
            if isinstance(label, str):
                # Some datasets use string labels directly
                cls_name = label
                label_key = label
            else:
                cls_name = _get_class_name(label)
                label_key = label

            if n_per_class and class_counts.get(label_key, 0) >= n_per_class:
                continue

            cls_dir = DATA_DIR / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            count = class_counts.get(label_key, 0)
            out_path = cls_dir / f"{count:06d}.jpg"

            try:
                img = sample[image_field]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(str(out_path), format="JPEG", quality=95)
                class_counts[label_key] = count + 1
                saved_total += 1
            except OSError as e:
                console.print(f"[red]Write failed for {out_path}: {e}[/red]")
                sys.exit(1)
            except Exception:
                continue  # skip corrupt images silently

            progress.update(
                task,
                description=f"Saved {saved_total} images ({len(class_counts)} classes)..."
            )

    console.print(f"\n[green]✓[/green] {saved_total} images saved to data/")
    console.print(f"  Classes: {len(class_counts)}")

    # Summary table (top 10 classes by count)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Class")
    table.add_column("Count", justify="right")
    top = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
    for lbl, cnt in top:
        table.add_row(str(class_names.get(lbl, lbl)), str(cnt))
    if len(class_counts) > 10:
        table.add_row(f"... ({len(class_counts) - 10} more classes)", "")
    console.print(table)


# ── embed ─────────────────────────────────────────────────────────────────────

def cmd_embed(args):
    from src.embed import extract_embeddings

    _ensure_dirs()
    models = ["dinov2", "clip"] if args.model == "all" else [args.model]
    for m in models:
        try:
            extract_embeddings(
                model_name=m,
                data_dir=DATA_DIR,
                embeddings_dir=EMBEDDINGS_DIR,
                batch_size=args.batch_size,
                force=args.force,
                debug=args.debug,
            )
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                raise
            console.print(f"[red]embed failed for {m}: {e}[/red]")
            sys.exit(1)


# ── trace ─────────────────────────────────────────────────────────────────────

def cmd_trace(args):
    from src.chain import run_chain_traversal

    _ensure_dirs()
    models = ["dinov2", "clip"] if args.model == "all" else [args.model]
    ef = getattr(args, "ef", 50)
    for m in models:
        try:
            run_chain_traversal(
                model_name=m,
                embeddings_dir=EMBEDDINGS_DIR,
                results_dir=RESULTS_DIR,
                max_steps=args.max_steps,
                ef=ef,
                force=args.force,
                debug=args.debug,
            )
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                raise
            console.print(f"[red]trace failed for {m}: {e}[/red]")
            sys.exit(1)


# ── analyze ───────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    from src.analyze import analyze_model, compare_models

    _ensure_dirs()

    if args.compare:
        try:
            compare_models(RESULTS_DIR, EMBEDDINGS_DIR, debug=args.debug)
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                raise
            console.print(f"[red]compare failed: {e}[/red]")
            sys.exit(1)
        return

    models = ["dinov2", "clip"] if args.model == "all" else [args.model]
    for m in models:
        try:
            analyze_model(m, RESULTS_DIR, EMBEDDINGS_DIR, debug=args.debug)
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                raise
            console.print(f"[red]analyze failed for {m}: {e}[/red]")
            sys.exit(1)

    if args.model == "all":
        try:
            compare_models(RESULTS_DIR, EMBEDDINGS_DIR, debug=args.debug)
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                raise
            console.print(f"[red]compare failed: {e}[/red]")


# ── export ────────────────────────────────────────────────────────────────────

def cmd_export(args):
    from src.export import run_export

    _ensure_dirs()
    try:
        run_export(
            embeddings_dir=EMBEDDINGS_DIR,
            results_dir=RESULTS_DIR,
            viz_dir=VIZ_DIR,
            thumb_size=args.thumb_size,
            force=args.force,
            debug=args.debug,
        )
    except SystemExit:
        raise
    except Exception as e:
        if args.debug:
            raise
        console.print(f"[red]export failed: {e}[/red]")
        sys.exit(1)


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status(args):
    import os

    def _file_info(path: Path) -> str:
        if path.exists():
            mb = os.path.getsize(path) / 1024 / 1024
            return f"[green]✓[/green]  {mb:.1f} MB"
        return "[red]✗[/red]  missing"

    def _dir_count(path: Path, ext: str = "*") -> str:
        if not path.exists():
            return "[red]✗[/red]  missing"
        count = len(list(path.rglob(ext)))
        return f"[green]✓[/green]  {count} files"

    table = Table(show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")

    table.add_row("data/ images", _dir_count(DATA_DIR, "*.jpg"))
    table.add_row("embeddings/dinov2.npy", _file_info(EMBEDDINGS_DIR / "dinov2.npy"))
    table.add_row("embeddings/clip.npy", _file_info(EMBEDDINGS_DIR / "clip.npy"))
    table.add_row("embeddings/index.json", _file_info(EMBEDDINGS_DIR / "index.json"))
    table.add_row("results/chains_dinov2.json", _file_info(RESULTS_DIR / "chains_dinov2.json"))
    table.add_row("results/chains_clip.json", _file_info(RESULTS_DIR / "chains_clip.json"))
    table.add_row("results/nn_map_dinov2.npy", _file_info(RESULTS_DIR / "nn_map_dinov2.npy"))
    table.add_row("results/nn_map_clip.npy", _file_info(RESULTS_DIR / "nn_map_clip.npy"))
    table.add_row("results/stats_dinov2.json", _file_info(RESULTS_DIR / "stats_dinov2.json"))
    table.add_row("results/stats_clip.json", _file_info(RESULTS_DIR / "stats_clip.json"))
    table.add_row("results/comparison.json", _file_info(RESULTS_DIR / "comparison.json"))
    table.add_row("viz/data.json", _file_info(VIZ_DIR / "data.json"))
    table.add_row("viz/index.html", _file_info(VIZ_DIR / "index.html"))
    table.add_row("notebooks/analysis.ipynb", _file_info(NOTEBOOKS_DIR / "analysis.ipynb"))

    console.print(table)

    # Shape info for embeddings
    for m, dim in [("dinov2", 768), ("clip", 512)]:
        p = EMBEDDINGS_DIR / f"{m}.npy"
        if p.exists():
            import numpy as np
            arr = np.load(p)
            console.print(f"  {m}: shape={arr.shape}, expected dim={dim}")


# ── run-all ───────────────────────────────────────────────────────────────────

def cmd_run_all(args):
    import os

    steps = [
        ("setup-data", "setup-data"),
        ("embed", "embed (DINOv2 + CLIP)"),
        ("trace", "trace (DINOv2 + CLIP)"),
        ("analyze", "analyze (stats + compare)"),
        ("export", "export"),
    ]

    class _FakeArgs:
        debug = args.debug
        force = args.force
        model = "all"
        dataset = "cifar10"
        n_per_class = None
        batch_size = 16  # conservative default; safe for large N on MPS/CPU
        max_steps = 100
        ef = 50
        compare = False
        thumb_size = 64

    fa = _FakeArgs()

    results_summary = []

    def run_step(n, label, fn):
        console.rule(f"[bold][{n}/5] {label}[/bold]")
        try:
            fn(fa)
            results_summary.append((label, True, ""))
        except SystemExit as e:
            results_summary.append((label, False, str(e)))
            console.print(f"[red]Step failed. Aborting run-all.[/red]")
            sys.exit(1)
        except Exception as e:
            if args.debug:
                raise
            results_summary.append((label, False, str(e)))
            console.print(f"[red]{label} failed: {e}[/red]")
            sys.exit(1)

    run_step(1, "setup-data", cmd_setup_data)
    run_step(2, "embed", cmd_embed)
    run_step(3, "trace", cmd_trace)
    run_step(4, "analyze", cmd_analyze)
    run_step(5, "export", cmd_export)

    console.rule("[bold]Run-All Complete[/bold]")
    n_images = len(list(DATA_DIR.rglob("*.jpg")))
    console.print(f"[1/5] setup-data     [green]✓[/green]  {n_images} images in data/")

    for m, dim in [("DINOv2", 768), ("CLIP", 512)]:
        p = EMBEDDINGS_DIR / f"{m.lower()}.npy"
        if p.exists():
            import numpy as np
            arr = np.load(p)
            console.print(f"[2/5] embed          [green]✓[/green]  {m} {arr.shape}")

    for m in ["dinov2", "clip"]:
        p = RESULTS_DIR / f"chains_{m}.json"
        if p.exists():
            size_kb = os.path.getsize(p) // 1024
            console.print(f"[3/5] trace          [green]✓[/green]  {m} chains ({size_kb} KB)")

    console.print(f"[4/5] analyze        [green]✓[/green]  Stats computed")

    vp = VIZ_DIR / "data.json"
    if vp.exists():
        mb = os.path.getsize(vp) / 1024 / 1024
        console.print(f"[5/5] export         [green]✓[/green]  viz/data.json ({mb:.1f} MB)")

    console.print("\n[bold green]All done.[/bold green] Open [bold]viz/index.html[/bold] in your browser.")


# ── argument parsing ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="nn-chain-explorer: study NN-chain topology in image embedding spaces",
    )
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks")

    sub = parser.add_subparsers(dest="command", required=True)

    # setup-data
    p_setup = sub.add_parser("setup-data", help="Download and sample images (CIFAR-10 or ImageNet)")
    p_setup.add_argument("--force", action="store_true", help="Overwrite existing data")
    p_setup.add_argument(
        "--dataset",
        choices=["cifar10", "imagenet"],
        default="cifar10",
        help="Dataset to download: cifar10 (default, 1K images) or imagenet (1.2M images)",
    )
    p_setup.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Limit images per class (e.g. 100 for a 100K ImageNet run). Default: all available.",
    )
    p_setup.add_argument(
        "--hf-dataset",
        default="zh-plus/tiny-imagenet",
        help=(
            "HuggingFace dataset ID for --dataset imagenet. "
            "Default: zh-plus/tiny-imagenet (ungated, 100K images, 200 classes). "
            "For full ImageNet: ILSVRC/imagenet-1k (gated, requires huggingface-cli login)."
        ),
    )

    # embed
    p_embed = sub.add_parser("embed", help="Extract embeddings")
    p_embed.add_argument(
        "--model", choices=["dinov2", "clip", "all"], default="all", help="Which model to use"
    )
    p_embed.add_argument("--data", default="./data", help="Image directory")
    p_embed.add_argument("--batch-size", type=int, default=32)
    p_embed.add_argument("--force", action="store_true")

    # trace
    p_trace = sub.add_parser("trace", help="Trace NN chains")
    p_trace.add_argument("--model", choices=["dinov2", "clip", "all"], default="all")
    p_trace.add_argument("--max-steps", type=int, default=100)
    p_trace.add_argument(
        "--ef",
        type=int,
        default=50,
        help="hnswlib query-time ef parameter (higher = more accurate, slower). Default: 50",
    )
    p_trace.add_argument("--force", action="store_true")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Compute statistics")
    p_analyze.add_argument("--model", choices=["dinov2", "clip", "all"], default="all")
    p_analyze.add_argument("--compare", action="store_true", help="Cross-encoder comparison only")
    p_analyze.add_argument("--force", action="store_true")

    # export
    p_export = sub.add_parser("export", help="Generate viz/data.json")
    p_export.add_argument("--thumb-size", type=int, default=64)
    p_export.add_argument("--force", action="store_true")

    # status
    sub.add_parser("status", help="Show pipeline status")

    # run-all
    p_run_all = sub.add_parser("run-all", help="Run full pipeline")
    p_run_all.add_argument("--force", action="store_true")

    args = parser.parse_args()

    # Propagate --debug to all subcommands
    if not hasattr(args, "debug"):
        args.debug = False

    try:
        if args.command == "setup-data":
            cmd_setup_data(args)
        elif args.command == "embed":
            cmd_embed(args)
        elif args.command == "trace":
            cmd_trace(args)
        elif args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "export":
            cmd_export(args)
        elif args.command == "status":
            cmd_status(args)
        elif args.command == "run-all":
            cmd_run_all(args)
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            console.print(f"[red]Error:[/red] {e}")
            console.print("Run with [bold]--debug[/bold] for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
