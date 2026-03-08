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


VIZ_MAX_NODES = 1000  # max nodes to render in D3 visualization
VIZ_TOP_HUBS = 500   # always include top-N hubs


def _select_viz_nodes(image_index: list[dict], nn_map_d: list | None, nn_map_c: list | None) -> list[int]:
    """
    For large datasets (N > VIZ_MAX_NODES), select a curated subset:
    - top-VIZ_TOP_HUBS hub nodes (by in-degree in dinov2, or clip if no dinov2)
    - random sample to fill up to VIZ_MAX_NODES

    Returns sorted list of original indices to include.
    """
    N = len(image_index)
    if N <= VIZ_MAX_NODES:
        return list(range(N))

    # Compute in-degree for hub selection
    nn_map = nn_map_d if nn_map_d is not None else nn_map_c
    in_degree = np.zeros(N, dtype=np.int32)
    for nn in nn_map:
        in_degree[nn] += 1

    # Top hubs
    top_hubs = set(np.argsort(in_degree)[-VIZ_TOP_HUBS:].tolist())

    # Random sample for remaining slots
    rng = np.random.default_rng(42)
    remaining_pool = [i for i in range(N) if i not in top_hubs]
    n_random = VIZ_MAX_NODES - len(top_hubs)
    random_sample = set(rng.choice(remaining_pool, size=min(n_random, len(remaining_pool)), replace=False).tolist())

    selected = sorted(top_hubs | random_sample)
    console.print(
        f"  [dim]Large dataset: sampling {len(selected)} of {N} nodes for visualization "
        f"(top {len(top_hubs)} hubs + {len(random_sample)} random).[/dim]"
    )
    return selected


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

    # 1. Load chain results (at least one model required)
    console.print("[1/4] Loading chain results...", end="  ")
    chains_d_path = results_dir / "chains_dinov2.json"
    chains_c_path = results_dir / "chains_clip.json"
    nn_d_path = results_dir / "nn_map_dinov2.npy"
    nn_c_path = results_dir / "nn_map_clip.npy"

    if not chains_d_path.exists() and not chains_c_path.exists():
        console.print(
            f"\n[red]No chain results found.[/red] "
            "Run [bold]python main.py trace --model all[/bold] first."
        )
        raise SystemExit(1)

    chains_d = _load_json(chains_d_path, "trace --model dinov2") if chains_d_path.exists() else None
    chains_c = _load_json(chains_c_path, "trace --model clip") if chains_c_path.exists() else None
    nn_d = np.load(nn_d_path).tolist() if nn_d_path.exists() else None
    nn_c = np.load(nn_c_path).tolist() if nn_c_path.exists() else None
    console.print("[green]✓[/green]")

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

    # Determine if we need to subsample for visualization
    viz_indices = _select_viz_nodes(image_index, nn_d, nn_c)
    viz_n = len(viz_indices)
    sampled = viz_n < N

    # Build index remapping: original_id -> viz_id (for edges within the viz subset)
    idx_remap = {orig: viz for viz, orig in enumerate(viz_indices)}

    # 3. Generate thumbnails for selected nodes only
    console.print(f"[3/4] Generating thumbnails ({thumb_size}x{thumb_size}) for {viz_n} nodes...")
    thumbnails = {}  # orig_id -> b64
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
        task = progress.add_task("Thumbnails", total=viz_n, rate=0)
        for pos, orig_i in enumerate(viz_indices):
            entry = image_index[orig_i]
            b64 = _encode_thumbnail(entry["path"], thumb_size)
            placeholder = _grey_placeholder_b64(thumb_size)
            if b64 == placeholder:
                failed_thumbs += 1
            thumbnails[orig_i] = b64
            elapsed = time.time() - t0
            rate = (pos + 1) / elapsed if elapsed > 0 else 0
            progress.update(task, advance=1, rate=rate)

    total_thumb_bytes = sum(len(t) for t in thumbnails.values())
    total_thumb_mb = total_thumb_bytes / 1024 / 1024
    console.print(f"  [green]✓[/green] Thumbnail size: {total_thumb_mb:.1f} MB")
    if failed_thumbs:
        console.print(f"  [yellow]✗[/yellow] Failed thumbnails: {failed_thumbs}")
    else:
        console.print(f"  [green]✓[/green] Failed thumbnails: 0")

    # 4. Assemble and write
    console.print("[4/4] Writing viz/data.json...", end="  ")

    images_out = []
    for orig_i in viz_indices:
        entry = image_index[orig_i]
        images_out.append({
            "id": orig_i,
            "viz_id": idx_remap[orig_i],
            "path": entry["path"],
            "class": entry["class"],
            "thumbnail": thumbnails[orig_i],
        })

    def _filter_chains(chains_dict: dict | None) -> dict | None:
        """Keep only chains for viz_indices (as strings)."""
        if chains_dict is None:
            return None
        return {str(orig_i): chains_dict["chains"][str(orig_i)]
                for orig_i in viz_indices
                if str(orig_i) in chains_dict["chains"]}

    def _filter_nn_map(nn_map: list | None) -> list | None:
        """Return nn_map values only for viz_indices."""
        if nn_map is None:
            return None
        return [nn_map[orig_i] for orig_i in viz_indices]

    output = {
        "metadata": {
            "n_images": N,
            "viz_n": viz_n,
            "sampled": sampled,
            "models": (
                ["dinov2", "clip"] if (chains_d and chains_c)
                else ["dinov2"] if chains_d
                else ["clip"]
            ),
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        },
        "images": images_out,
        "viz_indices": viz_indices,  # original indices included in viz
    }

    if chains_d is not None:
        output["dinov2"] = {
            "nn_map": _filter_nn_map(nn_d),
            "chains": _filter_chains(chains_d),
            "stats": stats_d,
        }
    if chains_c is not None:
        output["clip"] = {
            "nn_map": _filter_nn_map(nn_c),
            "chains": _filter_chains(chains_c),
            "stats": stats_c,
        }
    output["comparison"] = comparison

    with open(out_path, "w") as f:
        json.dump(output, f)

    file_mb = os.path.getsize(out_path) / 1024 / 1024
    console.print(f"[green]✓[/green]  ({file_mb:.1f} MB)")

    if file_mb > WARN_SIZE_MB:
        console.print(
            f"[yellow]WARNING:[/yellow] viz/data.json is {file_mb:.1f} MB (> {WARN_SIZE_MB} MB). "
            "Consider reducing thumb size with --thumb-size."
        )

    if sampled:
        console.print(
            f"\n[dim]Note: Visualization shows {viz_n} of {N} images "
            f"(top {VIZ_TOP_HUBS} hubs + random sample).[/dim]"
        )

    console.print(
        "\n[green]Done.[/green] Open [bold]viz/index.html[/bold] in your browser.\n"
        "Note: Open from viz/ directory or via: [bold]python -m http.server 8080 --directory viz[/bold]"
    )
