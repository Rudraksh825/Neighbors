"""
NN-chain traversal and cycle detection for nn-chain-explorer.
Uses brute-force NumPy cosine similarity (inner product on L2-normalized embeddings).
No FAISS dependency — avoids OpenMP/Python 3.13 compatibility issues on macOS.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def build_nn_map(embeddings: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Build nearest-neighbor map via brute-force inner product (cosine sim on L2-normalized vectors).
    Processes in chunks to keep memory usage bounded.
    Returns int32 array of shape [N] where nn_map[i] = index of NN of image i.
    """
    embeddings = embeddings.astype(np.float32)
    N, dim = embeddings.shape
    CHUNK = 256  # rows processed at once; keeps peak memory at ~256*N*4 bytes (~1GB for N=1000)

    console.print(f"[1/3] Building NN index (brute-force cosine, no FAISS)...")
    t0 = time.time()
    console.print(f"  [green]✓[/green] Ready  |  Dim: {dim}  |  N vectors: {N}")

    console.print(f"[2/3] Computing NN map (chunked inner-product, k=2 with self-exclusion)...")
    t0 = time.time()

    nn_map = np.empty(N, dtype=np.int32)
    arange = np.arange(N)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        # sim[i, j] = dot(embeddings[start+i], embeddings[j])
        sim = embeddings[start:end] @ embeddings.T  # shape [chunk, N]
        # Exclude self by setting diagonal to -inf
        for local_i in range(end - start):
            sim[local_i, start + local_i] = -np.inf
        nn_map[start:end] = np.argmax(sim, axis=1).astype(np.int32)

    t1 = time.time()

    # Self-loop check
    self_loops = np.sum(nn_map == arange)
    if self_loops > 0:
        console.print(
            f"  [yellow]WARNING:[/yellow] {self_loops} self-loop(s) found after exclusion."
        )
    else:
        console.print(
            f"  [green]✓[/green] NN map computed in {t1-t0:.2f}s\n"
            f"  [green]✓[/green] Self-exclusion check: 0 self-loops found  [green][OK][/green]"
        )

    return nn_map


def trace_chain(start: int, nn_map: np.ndarray, max_steps: int = 100) -> dict:
    """Trace NN chain from start until cycle detected or max_steps reached."""
    chain = [start]
    visited = {start: 0}

    current = start
    for step in range(max_steps):
        nxt = int(nn_map[current])
        if nxt in visited:
            cycle_start_pos = visited[nxt]
            return {
                "chain": chain + [nxt],
                "transient_length": cycle_start_pos,
                "cycle_length": len(chain) - cycle_start_pos,
                "cycle_entry_node": nxt,
                "cycle_nodes": chain[cycle_start_pos:],
                "terminated_by": "cycle",
            }
        visited[nxt] = len(chain)
        chain.append(nxt)
        current = nxt

    return {
        "chain": chain,
        "transient_length": len(chain),
        "cycle_length": -1,
        "cycle_entry_node": -1,
        "cycle_nodes": [],
        "terminated_by": "max_steps",
    }


def run_chain_traversal(
    model_name: str,
    embeddings_dir: Path,
    results_dir: Path,
    max_steps: int = 100,
    force: bool = False,
    debug: bool = False,
) -> dict:
    """
    Load embeddings, build FAISS index, trace all chains, save results.
    Returns the full chains dict.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"chains_{model_name}.json"
    out_nn = results_dir / f"nn_map_{model_name}.npy"

    if out_json.exists() and not force:
        console.print(
            f"[yellow][SKIP][/yellow] {out_json} already exists. Use --force to re-run."
        )
        with open(out_json) as f:
            return json.load(f)

    emb_path = embeddings_dir / f"{model_name}.npy"
    if not emb_path.exists():
        console.print(
            f"[red]Embeddings not found: {emb_path}[/red]\n"
            f"Run: [bold]python main.py embed --model {model_name}[/bold] first."
        )
        raise SystemExit(1)

    embeddings = np.load(emb_path).astype(np.float32)
    N = len(embeddings)

    console.print(
        Panel(
            f"  [bold]NN-Chain Traversal — {model_name.upper()}[/bold]\n"
            f"  N images: {N}  |  Max steps: {max_steps}",
            expand=False,
        )
    )

    nn_map = build_nn_map(embeddings, debug)

    # Save nn_map
    np.save(out_nn, nn_map)

    console.print(f"[3/3] Tracing chains...")
    t0 = time.time()

    chains = {}
    n_cycle = 0
    n_max = 0
    tau_sum = 0
    tau_vals = []
    cycle_len_vals = []
    cycle_node_set = set()

    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(" | "),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Tracing", total=N)

        for i in range(N):
            result = trace_chain(i, nn_map, max_steps)
            chains[str(i)] = result

            if result["terminated_by"] == "cycle":
                n_cycle += 1
                tau_vals.append(result["transient_length"])
                cycle_len_vals.append(result["cycle_length"])
                for cn in result["cycle_nodes"]:
                    cycle_node_set.add(cn)
            else:
                n_max += 1
                tau_vals.append(result["transient_length"])

            progress.advance(task)

            if (i + 1) % 100 == 0 or (i + 1) == N:
                avg_tau = np.mean(tau_vals) if tau_vals else 0
                avg_cyc = np.mean(cycle_len_vals) if cycle_len_vals else 0
                progress.print(
                    f"  Live stats ({i+1}/{N}): "
                    f"Avg τ={avg_tau:.2f}  Avg ℓ={avg_cyc:.2f}  "
                    f"Cycles={n_cycle}  Unique cycle nodes={len(cycle_node_set)}"
                )

    t1 = time.time()

    tau_arr = np.array(tau_vals)
    cyc_arr = np.array(cycle_len_vals) if cycle_len_vals else np.array([0])

    # Hub in-degree
    in_degree = np.bincount(nn_map, minlength=N)
    max_hub_idx = int(np.argmax(in_degree))

    # Load index for hub image path
    idx_path = embeddings_dir / "index.json"
    hub_path = "unknown"
    if idx_path.exists():
        with open(idx_path) as f:
            image_index = json.load(f)
        if max_hub_idx < len(image_index):
            hub_path = image_index[max_hub_idx]["path"]

    fixed_points = int(np.sum(np.array(cycle_len_vals) == 1)) if cycle_len_vals else 0

    metadata = {
        "model": model_name,
        "n_images": N,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_steps": max_steps,
        "n_terminated_by_cycle": n_cycle,
        "n_terminated_by_max_steps": n_max,
    }

    output = {"metadata": metadata, "chains": chains}
    with open(out_json, "w") as f:
        json.dump(output, f)

    # Summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[green]✓[/green]", f"All {N} chains traced in {t1-t0:.2f}s")
    table.add_row(
        "[green]✓[/green]",
        f"Avg τ: {tau_arr.mean():.2f} ± {tau_arr.std():.2f}  |  Max τ: {tau_arr.max()}",
    )
    if cycle_len_vals:
        table.add_row(
            "[green]✓[/green]",
            f"Avg ℓ: {cyc_arr.mean():.2f} ± {cyc_arr.std():.2f}",
        )
        fp_pct = 100 * fixed_points / N
        table.add_row("[green]✓[/green]", f"Fixed points (ℓ=1): {fixed_points}/{N}  ({fp_pct:.1f}%)")
    table.add_row("[green]✓[/green]", f"Unique cycle nodes: {len(cycle_node_set)}  ({100*len(cycle_node_set)/N:.1f}% of dataset)")
    table.add_row("[green]✓[/green]", f"Max hub in-degree: {in_degree[max_hub_idx]}  (image {max_hub_idx} — {hub_path})")
    if n_max > 0:
        table.add_row(
            "[yellow]⚠[/yellow]",
            f"Terminated by max_steps: {n_max}  ({100*n_max/N:.1f}% — unexpected for finite dataset)",
        )
    else:
        table.add_row("[green]✓[/green]", "Terminated by max_steps: 0  [OK]")
    table.add_row("[green]✓[/green]", f"Saved to {out_json}")

    console.print("\nFinal Summary:")
    console.print(table)

    return output
