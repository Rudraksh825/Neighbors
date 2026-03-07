"""
Export pipeline results to viz/data.json for standalone visualization.
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()

THUMB_SIZE = 64
THUMB_QUALITY = 70
WARN_SIZE_MB = 100


def _grey_placeholder_b64(size: int = 64) -> str:
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _encode_thumbnail(path: str, size: int = THUMB_SIZE) -> str:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return _grey_placeholder_b64(size)


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        console.print(f"[red]Missing:[/red] {path} — run the {label} step first.")
        raise SystemExit(1)
    with open(path) as f:
        return json.load(f)


def run_export(
    embeddings_dir: Path,
    results_dir: Path,
    viz_dir: Path,
    thumb_size: int = THUMB_SIZE,
    force: bool = False,
    debug: bool = False,
) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / "data.json"

    if out_path.exists() and not force:
        console.print(
            f"[yellow][SKIP][/yellow] {out_path} already exists. Use --force to re-export."
        )
        return

    console.print(Panel("  [bold]Export for Visualization[/bold]", expand=False))

    # 1. Load chain results
    console.print("[1/4] Loading chain results...", end="  ")
    chains_d = _load_json(results_dir / "chains_dinov2.json", "trace --model dinov2")
    chains_c = _load_json(results_dir / "chains_clip.json", "trace --model clip")
    console.print("[green]✓[/green]")

    # Load nn_maps
    nn_d = np.load(results_dir / "nn_map_dinov2.npy").tolist()
    nn_c = np.load(results_dir / "nn_map_clip.npy").tolist()

    # 2. Load stats
    console.print("[2/4] Loading analysis stats...", end="  ")
    stats_d_path = results_dir / "stats_dinov2.json"
    stats_c_path = results_dir / "stats_clip.json"
    comp_path = results_dir / "comparison.json"

    stats_d = _load_json(stats_d_path, "analyze --model dinov2") if stats_d_path.exists() else {}
    stats_c = _load_json(stats_c_path, "analyze --model clip") if stats_c_path.exists() else {}
    comparison = _load_json(comp_path, "analyze --compare") if comp_path.exists() else {}
    console.print("[green]✓[/green]")

    # Load image index
    idx_path = embeddings_dir / "index.json"
    image_index = _load_json(idx_path, "embed")
    N = len(image_index)

    # 3. Generate thumbnails
    console.print("[3/4] Generating thumbnails (64x64)...")
    thumbnails = []
    failed_thumbs = 0
    t0 = time.time()

    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(" | "),
        TimeElapsedColumn(),
        TextColumn(" | {task.fields[rate]:.0f} img/s"),
        console=console,
    ) as progress:
        task = progress.add_task("Thumbnails", total=N, rate=0)
        for i, entry in enumerate(image_index):
            b64 = _encode_thumbnail(entry["path"], thumb_size)
            if b64 == _grey_placeholder_b64(thumb_size):
                failed_thumbs += 1
            thumbnails.append(b64)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            progress.update(task, advance=1, rate=rate)

    total_thumb_bytes = sum(len(t) for t in thumbnails)
    total_thumb_mb = total_thumb_bytes / 1024 / 1024
    console.print(f"  [green]✓[/green] Thumbnail size: {total_thumb_mb:.1f} MB")
    if failed_thumbs:
        console.print(f"  [yellow]✗[/yellow] Failed thumbnails: {failed_thumbs}")
    else:
        console.print(f"  [green]✓[/green] Failed thumbnails: 0")

    # 4. Assemble and write
    console.print("[4/4] Writing viz/data.json...", end="  ")

    images_out = []
    for i, entry in enumerate(image_index):
        images_out.append({
            "id": entry["id"],
            "path": entry["path"],
            "class": entry["class"],
            "thumbnail": thumbnails[i],
        })

    output = {
        "metadata": {
            "n_images": N,
            "models": ["dinov2", "clip"],
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        },
        "images": images_out,
        "dinov2": {
            "nn_map": nn_d,
            "chains": chains_d["chains"],
            "stats": stats_d,
        },
        "clip": {
            "nn_map": nn_c,
            "chains": chains_c["chains"],
            "stats": stats_c,
        },
        "comparison": comparison,
    }

    with open(out_path, "w") as f:
        json.dump(output, f)

    file_mb = os.path.getsize(out_path) / 1024 / 1024
    console.print(f"[green]✓[/green]  ({file_mb:.1f} MB)")

    if file_mb > WARN_SIZE_MB:
        console.print(
            f"[yellow]WARNING:[/yellow] viz/data.json is {file_mb:.1f} MB (> {WARN_SIZE_MB} MB). "
            "Consider reducing thumb size with --thumb-size."
        )

    console.print(
        "\n[green]Done.[/green] Open [bold]viz/index.html[/bold] in your browser.\n"
        "Note: Open from viz/ directory or via: [bold]python -m http.server 8080 --directory viz[/bold]"
    )
