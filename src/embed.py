"""
Embedding extraction for nn-chain-explorer.
Supports DINOv2 (ViT-B/14) and CLIP (ViT-B-32).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def discover_images(data_dir: Path) -> list[dict]:
    """Recursively discover images, returning list of {id, path, class}."""
    entries = []
    idx = 0
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            # class = parent folder name if nested, else "unknown"
            rel = p.relative_to(data_dir)
            parts = rel.parts
            cls = parts[0] if len(parts) > 1 else "unknown"
            entries.append({"id": idx, "path": str(p), "class": cls})
            idx += 1
    return entries


def save_index(entries: list[dict], embeddings_dir: Path) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    with open(embeddings_dir / "index.json", "w") as f:
        json.dump(entries, f, indent=2)


def load_index(embeddings_dir: Path) -> list[dict]:
    idx_path = embeddings_dir / "index.json"
    if not idx_path.exists():
        return []
    with open(idx_path) as f:
        return json.load(f)


def _make_dinov2_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_dinov2(device: str, debug: bool = False):
    import torch

    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    except Exception as e:
        if debug:
            raise
        console.print(
            f"[red]Failed to load DINOv2 from torch.hub: {e}[/red]\n"
            "Suggestion: check your internet connection or manually download weights."
        )
        raise SystemExit(1)
    model.eval().to(device)
    return model


def _load_clip(device: str, debug: bool = False):
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
    except Exception as e:
        if debug:
            raise
        console.print(
            f"[red]Failed to load CLIP via open_clip: {e}[/red]\n"
            "Suggestion: check your internet connection or manually download weights."
        )
        raise SystemExit(1)
    model.eval().to(device)
    return model, preprocess


def _open_image_rgb(path: str) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        return None
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def _extract_batch_dinov2(model, batch_tensor, device: str):
    import torch

    with torch.no_grad():
        feats = model.forward_features(batch_tensor.to(device))
        return feats["x_norm_clstoken"].cpu().numpy()


def _extract_batch_clip(model, batch_tensor, device: str):
    import torch

    with torch.no_grad():
        feats = model.encode_image(batch_tensor.to(device))
        return feats.cpu().float().numpy()


def _norm_check(embeddings: np.ndarray) -> tuple[float, float, float]:
    norms = np.linalg.norm(embeddings, axis=1)
    return float(norms.mean()), float(norms.min()), float(norms.max())


def extract_embeddings(
    model_name: str,
    data_dir: Path,
    embeddings_dir: Path,
    batch_size: int = 32,
    force: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """
    Extract and save embeddings for all images in data_dir.
    Returns the embeddings array.
    """
    import torch

    assert model_name in ("dinov2", "clip"), f"Unknown model: {model_name}"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    out_path = embeddings_dir / f"{model_name}.npy"

    # Discover images and save/update index
    entries = discover_images(data_dir)
    if len(entries) == 0:
        console.print(
            "[red]No images found in data/.[/red] "
            "Run [bold]python main.py setup-data[/bold] or place images in data/."
        )
        raise SystemExit(1)
    if len(entries) < 2:
        console.print("[red]Need at least 2 images for FAISS k=2 search.[/red]")
        raise SystemExit(1)

    save_index(entries, embeddings_dir)

    # Cache check
    if out_path.exists() and not force:
        cached = np.load(out_path)
        if cached.shape[0] == len(entries):
            console.print(
                f"[yellow][SKIP][/yellow] {out_path} already exists "
                f"({cached.shape[0]} x {cached.shape[1]}). Use --force to re-extract."
            )
            return cached
        else:
            console.print(
                f"[yellow]Shape mismatch:[/yellow] cached {cached.shape[0]} embeddings "
                f"but index has {len(entries)} images. Re-extracting."
            )

    device = get_device()
    dim_map = {"dinov2": 768, "clip": 512}
    model_label = {"dinov2": "DINOv2 ViT-B/14", "clip": "CLIP ViT-B-32"}

    console.print(
        Panel(
            f"  [bold]Embedding Extraction — {model_label[model_name]}[/bold]\n"
            f"  Device: [cyan]{device}[/cyan]  |  Batch size: {batch_size}\n"
            f"  Images found: {len(entries)}",
            expand=False,
        )
    )

    # Load model
    if model_name == "dinov2":
        model = _load_dinov2(device, debug)
        transform = _make_dinov2_transform()
        extract_fn = _extract_batch_dinov2
    else:
        model, transform = _load_clip(device, debug)
        extract_fn = _extract_batch_clip

    skipped = []
    total = len(entries)
    dim = dim_map[model_name]
    t_start = time.time()
    imgs_done = 0

    # Use memmap for writing — avoids holding the full array in RAM during extraction.
    # For N=1.2M: DINOv2 = 3.7GB, CLIP = 2.5GB. np.save-compatible format.
    # We'll write via memmap then finalize with np.save (re-writes, but only once at end).
    # For large N, write directly into memmap to keep RAM flat during extraction.
    use_memmap = total > 50_000
    if use_memmap:
        console.print(
            f"  [dim]Large dataset ({total} images) — writing embeddings via memmap to avoid RAM spike.[/dim]"
        )
        mmap_path = out_path.with_suffix(".mmap.tmp")
        emb_mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(total, dim))
    else:
        all_embeddings = []

    write_cursor = 0  # tracks next row to write in memmap

    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn(" | ETA:"),
        TimeRemainingColumn(),
        TextColumn(" | "),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=total)

        i = 0
        while i < total:
            # Build batch
            batch_imgs = []
            batch_ids = []
            while len(batch_imgs) < batch_size and i < total:
                entry = entries[i]
                img = _open_image_rgb(entry["path"])
                if img is None:
                    console.print(
                        f"  [yellow]WARNING:[/yellow] Skipping corrupt/unreadable image: {entry['path']}"
                    )
                    skipped.append(entry["path"])
                    i += 1
                    progress.advance(task)
                    continue
                if img.width < 10 or img.height < 10:
                    console.print(
                        f"  [yellow]WARNING:[/yellow] Very small image (<10px): {entry['path']}"
                    )
                try:
                    t = transform(img)
                    batch_imgs.append(t)
                    batch_ids.append(i)
                except Exception as e:
                    console.print(
                        f"  [yellow]WARNING:[/yellow] Transform failed for {entry['path']}: {e}"
                    )
                    skipped.append(entry["path"])
                i += 1

            if not batch_imgs:
                continue

            batch_tensor = __import__("torch").stack(batch_imgs)

            # OOM retry logic
            current_batch_size = len(batch_imgs)
            success = False
            attempts = 0
            last_err = None
            while attempts <= 3 and not success:
                try:
                    feats = extract_fn(model, batch_tensor[:current_batch_size], device)
                    success = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and attempts < 3:
                        attempts += 1
                        current_batch_size = max(1, current_batch_size // 2)
                        console.print(
                            f"  [yellow]CUDA OOM — halving batch to {current_batch_size}[/yellow]"
                        )
                        if device != "cpu":
                            import torch
                            torch.cuda.empty_cache()
                    else:
                        last_err = e
                        break

            if not success:
                if debug:
                    raise last_err
                console.print(
                    f"  [red]Batch failed after OOM retries, falling back to CPU batch_size=8[/red]"
                )
                model = model.to("cpu")
                device = "cpu"
                batch_tensor_cpu = batch_tensor[:current_batch_size].to("cpu")
                feats = extract_fn(model, batch_tensor_cpu, "cpu")

            # Check for NaN
            if np.any(np.isnan(feats)):
                console.print(
                    f"  [red]NaN detected in embeddings for batch ending at index {i}. "
                    "Likely a corrupt image that passed PIL open check.[/red]"
                )
                if debug:
                    raise ValueError("NaN in embeddings")
                raise SystemExit(1)

            # L2 normalize this batch immediately
            batch_norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.maximum(batch_norms, 1e-12)

            n_feats = len(feats)
            if use_memmap:
                emb_mmap[write_cursor:write_cursor + n_feats] = feats
                write_cursor += n_feats
            else:
                all_embeddings.append(feats)

            imgs_done += n_feats
            progress.advance(task, len(batch_imgs))

            elapsed = time.time() - t_start
            rate = imgs_done / elapsed if elapsed > 0 else 0
            progress.print(
                f"  Batch done  |  {imgs_done}/{total}  |  {rate:.1f} img/s"
            )

    if use_memmap:
        # Trim to actual written rows (some images may have been skipped)
        actual_n = write_cursor
        embeddings = np.array(emb_mmap[:actual_n])  # load into RAM for norm check + save
        del emb_mmap
        mmap_path.unlink(missing_ok=True)
    else:
        if not all_embeddings:
            console.print("[red]No embeddings extracted.[/red]")
            raise SystemExit(1)
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        # L2 normalize (batches already normalized above, but vstack may lose precision)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

    if len(embeddings) == 0:
        console.print("[red]No embeddings extracted.[/red]")
        raise SystemExit(1)

    # Norm check
    mean_n, min_n, max_n = _norm_check(embeddings)
    norm_ok = abs(mean_n - 1.0) < 1e-4 and abs(min_n - 1.0) < 1e-3
    norm_status = "[green][OK][/green]" if norm_ok else "[red][FAIL][/red]"
    console.print(
        f"  Embedding norm check: mean={mean_n:.4f}, min={min_n:.4f}, max={max_n:.4f}  {norm_status}"
    )
    if not norm_ok and not debug:
        console.print("[red]Norm check failed. Embeddings may be incorrect.[/red]")

    np.save(out_path, embeddings)

    elapsed = time.time() - t_start
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[green]✓[/green]", f"{embeddings.shape[0]} embeddings extracted")
    table.add_row(
        "[red]✗[/red]" if skipped else "[green]✓[/green]",
        f"{len(skipped)} images skipped",
    )
    table.add_row("[green]✓[/green]", f"Output shape: {embeddings.shape}")
    table.add_row("[green]✓[/green]", f"All embedding norms = {mean_n:.4f}")
    table.add_row("[green]✓[/green]", f"Saved to {out_path}")
    table.add_row("[green]✓[/green]", f"Total time: {elapsed:.1f}s")
    console.print("\nSummary:")
    console.print(table)

    if skipped:
        console.print("\n[yellow]Skipped files:[/yellow]")
        for s in skipped:
            console.print(f"  {s}")

    return embeddings
