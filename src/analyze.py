"""
Statistical analysis of NN-chain results for nn-chain-explorer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _load_chains(results_dir: Path, model_name: str) -> dict:
    path = results_dir / f"chains_{model_name}.json"
    if not path.exists():
        console.print(
            f"[red]Missing:[/red] {path}\n"
            f"Run: [bold]python main.py trace --model {model_name}[/bold] first."
        )
        raise SystemExit(1)
    with open(path) as f:
        return json.load(f)


def _load_nn_map(results_dir: Path, model_name: str) -> np.ndarray:
    path = results_dir / f"nn_map_{model_name}.npy"
    if not path.exists():
        console.print(f"[red]Missing nn_map:[/red] {path}")
        raise SystemExit(1)
    return np.load(path).astype(np.int32)


def analyze_model(
    model_name: str,
    results_dir: Path,
    embeddings_dir: Path,
    debug: bool = False,
) -> dict:
    """Compute per-model statistics. Returns stats dict and saves to results/."""
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"  [bold]Statistical Analysis — {model_name.upper()}[/bold]",
            expand=False,
        )
    )

    data = _load_chains(results_dir, model_name)
    chains = data["chains"]
    N = data["metadata"]["n_images"]
    nn_map = _load_nn_map(results_dir, model_name)

    # Load image index for hub identification
    idx_path = embeddings_dir / "index.json"
    image_index = []
    if idx_path.exists():
        with open(idx_path) as f:
            image_index = json.load(f)

    stats = {}

    def step(n, label, fn):
        t0 = time.time()
        console.print(f"[{n}/7] {label}...", end="  ")
        result = fn()
        console.print(f"[green]✓[/green] ({time.time()-t0:.2f}s)")
        return result

    # 1. Transient length distribution
    def calc_tau():
        tau_vals = [chains[str(i)]["transient_length"] for i in range(N)]
        tau_arr = np.array(tau_vals)
        hist, edges = np.histogram(tau_arr, bins=range(int(tau_arr.max()) + 2))
        return {
            "values": tau_vals,
            "mean": float(tau_arr.mean()),
            "std": float(tau_arr.std()),
            "min": int(tau_arr.min()),
            "max": int(tau_arr.max()),
            "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
        }

    stats["transient"] = step(1, "Transient length distribution", calc_tau)

    # 2. Cycle length distribution
    def calc_cycle_len():
        cyc_vals = [
            chains[str(i)]["cycle_length"]
            for i in range(N)
            if chains[str(i)]["terminated_by"] == "cycle"
        ]
        if not cyc_vals:
            return {"values": [], "mean": 0, "std": 0, "min": 0, "max": 0, "histogram": {}}
        cyc_arr = np.array(cyc_vals)
        hist, edges = np.histogram(cyc_arr, bins=range(int(cyc_arr.max()) + 2))
        return {
            "values": cyc_vals,
            "mean": float(cyc_arr.mean()),
            "std": float(cyc_arr.std()),
            "min": int(cyc_arr.min()),
            "max": int(cyc_arr.max()),
            "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
        }

    stats["cycle_length"] = step(2, "Cycle length distribution", calc_cycle_len)

    # 3. Hub in-degree distribution
    def calc_hub():
        in_degree = np.bincount(nn_map, minlength=N)
        hist, edges = np.histogram(in_degree, bins=range(int(in_degree.max()) + 2))
        max_idx = int(np.argmax(in_degree))
        hub_path = image_index[max_idx]["path"] if max_idx < len(image_index) else "unknown"
        hub_class = image_index[max_idx]["class"] if max_idx < len(image_index) else "unknown"
        return {
            "in_degree": in_degree.tolist(),
            "mean": float(in_degree.mean()),
            "std": float(in_degree.std()),
            "max": int(in_degree.max()),
            "max_hub_idx": max_idx,
            "max_hub_path": hub_path,
            "max_hub_class": hub_class,
            "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
        }

    stats["hub"] = step(3, "Hub in-degree", calc_hub)

    # 4. Basin of attraction sizes
    def calc_basins():
        # For each image, find which cycle it flows into (identified by frozenset of cycle nodes)
        cycle_id_map = {}  # image_idx -> cycle canonical id
        cycle_members: dict[str, list] = {}  # cycle_id -> list of cycle nodes

        for i in range(N):
            c = chains[str(i)]
            if c["terminated_by"] == "cycle" and c["cycle_nodes"]:
                key = tuple(sorted(c["cycle_nodes"]))
                cid = str(key)
                cycle_id_map[i] = cid
                cycle_members[cid] = c["cycle_nodes"]

        # Basin size = number of images flowing into each cycle
        basin_sizes: dict[str, int] = {}
        for cid in cycle_id_map.values():
            basin_sizes[cid] = basin_sizes.get(cid, 0) + 1

        sizes = sorted(basin_sizes.values(), reverse=True)
        return {
            "n_distinct_cycles": len(cycle_members),
            "basin_sizes": sizes,
            "max_basin_size": int(sizes[0]) if sizes else 0,
            "mean_basin_size": float(np.mean(sizes)) if sizes else 0,
        }

    stats["basins"] = step(4, "Basin of attraction sizes", calc_basins)

    # 5. Fixed point rate
    def calc_fixed():
        n_fixed = sum(
            1 for i in range(N)
            if chains[str(i)]["cycle_length"] == 1
        )
        return {"n_fixed": n_fixed, "rate": float(n_fixed / N)}

    stats["fixed_points"] = step(5, "Fixed point rate", calc_fixed)

    # 6. Hub concentration (what % of images account for 50% of NN pointers)
    def calc_concentration():
        in_degree = np.array(stats["hub"]["in_degree"])
        sorted_deg = np.sort(in_degree)[::-1]
        cumsum = np.cumsum(sorted_deg)
        total = cumsum[-1]
        half = total / 2
        idx_50 = int(np.searchsorted(cumsum, half)) + 1
        pct_50 = float(idx_50 / N * 100)
        return {"top_pct_for_50_of_pointers": pct_50, "n_nodes_for_50_pct": idx_50}

    stats["hub_concentration"] = step(6, "Hub concentration", calc_concentration)

    # 7. Distinct cycle identification
    def calc_distinct_cycles():
        seen = {}
        for i in range(N):
            c = chains[str(i)]
            if c["terminated_by"] == "cycle" and c["cycle_nodes"]:
                key = tuple(sorted(c["cycle_nodes"]))
                seen[key] = c["cycle_nodes"]
        return {
            "n_distinct_cycles": len(seen),
            "cycle_node_sets": [list(v) for v in seen.values()],
        }

    stats["distinct_cycles"] = step(7, "Distinct cycle identification", calc_distinct_cycles)

    # Summary table
    console.print()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    hub_info = stats["hub"]
    tau_info = stats["transient"]
    cyc_info = stats["cycle_length"]
    fp_info = stats["fixed_points"]
    basin_info = stats["basins"]
    conc_info = stats["hub_concentration"]

    table.add_row("N images", str(N))
    table.add_row("Avg transient length τ", f"{tau_info['mean']:.2f} ± {tau_info['std']:.2f}")
    table.add_row("Max transient length", str(tau_info["max"]))
    table.add_row("Avg cycle length ℓ", f"{cyc_info['mean']:.2f} ± {cyc_info['std']:.2f}")
    table.add_row("Fixed point rate (ℓ=1)", f"{fp_info['rate']*100:.1f}%")
    table.add_row("Distinct cycles", str(basin_info["n_distinct_cycles"]))
    table.add_row("Largest basin size", str(basin_info["max_basin_size"]))
    table.add_row("Max hub in-degree", str(hub_info["max"]))
    table.add_row("Top hub image", hub_info["max_hub_path"])
    table.add_row("50% pointer concentration", f"top {conc_info['top_pct_for_50_of_pointers']:.1f}% of nodes")

    console.print(table)

    # Save
    out_path = results_dir / f"stats_{model_name}.json"
    with open(out_path, "w") as f:
        # Remove large raw arrays before saving to keep file manageable
        save_stats = {k: v for k, v in stats.items()}
        # Keep in_degree as list (needed for export), but don't save cycle_node_sets in full
        json.dump(save_stats, f, indent=2)

    console.print(f"\n[green]✓[/green] Saved to {out_path}")
    return stats


def compare_models(
    results_dir: Path,
    embeddings_dir: Path,
    debug: bool = False,
) -> dict:
    """Cross-encoder comparison. Requires both dinov2 and clip results."""
    for m in ("dinov2", "clip"):
        p = results_dir / f"chains_{m}.json"
        if not p.exists():
            console.print(
                f"[red]Missing:[/red] {p}\n"
                f"Run: [bold]python main.py trace --model {m}[/bold] first."
            )
            raise SystemExit(1)

    console.print(Panel("  [bold]Cross-Encoder Comparison — DINOv2 vs CLIP[/bold]", expand=False))

    data_d = _load_chains(results_dir, "dinov2")
    data_c = _load_chains(results_dir, "clip")
    nn_d = _load_nn_map(results_dir, "dinov2")
    nn_c = _load_nn_map(results_dir, "clip")

    N = min(data_d["metadata"]["n_images"], data_c["metadata"]["n_images"])

    # Load image index for class info
    idx_path = embeddings_dir / "index.json"
    image_index = []
    if idx_path.exists():
        with open(idx_path) as f:
            image_index = json.load(f)

    chains_d = data_d["chains"]
    chains_c = data_c["chains"]

    # 8. NN agreement rate
    agree = np.sum(nn_d[:N] == nn_c[:N])
    agree_rate = float(agree / N)

    # 9. NN agreement per class
    class_agree: dict[str, list] = {}
    for i in range(N):
        cls = image_index[i]["class"] if i < len(image_index) else "unknown"
        if cls not in class_agree:
            class_agree[cls] = []
        class_agree[cls].append(int(nn_d[i] == nn_c[i]))
    agreement_by_class = {cls: float(np.mean(vals)) for cls, vals in class_agree.items()}

    # 10. Transient length correlation
    tau_d = np.array([chains_d[str(i)]["transient_length"] for i in range(N)])
    tau_c = np.array([chains_c[str(i)]["transient_length"] for i in range(N)])
    pearson_r = float(np.corrcoef(tau_d, tau_c)[0, 1])

    # 11. Cycle node Jaccard
    cycle_nodes_d = set()
    cycle_nodes_c = set()
    for i in range(N):
        for cn in chains_d[str(i)].get("cycle_nodes", []):
            cycle_nodes_d.add(cn)
        for cn in chains_c[str(i)].get("cycle_nodes", []):
            cycle_nodes_c.add(cn)
    intersection = cycle_nodes_d & cycle_nodes_c
    union = cycle_nodes_d | cycle_nodes_c
    jaccard = float(len(intersection) / len(union)) if union else 0.0

    # 12. Chain co-convergence
    co_converge = 0
    for i in range(N):
        cn_d = set(chains_d[str(i)].get("cycle_nodes", []))
        cn_c = set(chains_c[str(i)].get("cycle_nodes", []))
        if cn_d & cn_c:
            co_converge += 1
    co_converge_rate = float(co_converge / N)

    comparison = {
        "n_images": N,
        "nn_agreement_rate": agree_rate,
        "nn_agreement_by_class": agreement_by_class,
        "tau_pearson_r": pearson_r,
        "cycle_node_jaccard": jaccard,
        "chain_co_convergence_rate": co_converge_rate,
        "cycle_nodes_dinov2_count": len(cycle_nodes_d),
        "cycle_nodes_clip_count": len(cycle_nodes_c),
        "cycle_nodes_intersection_count": len(intersection),
    }

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("NN agreement rate", f"{agree_rate*100:.1f}%")
    table.add_row("τ Pearson r (DINOv2 vs CLIP)", f"{pearson_r:.4f}")
    table.add_row("Cycle node Jaccard", f"{jaccard:.4f}")
    table.add_row("Chain co-convergence rate", f"{co_converge_rate*100:.1f}%")
    table.add_row("Cycle nodes (DINOv2)", str(len(cycle_nodes_d)))
    table.add_row("Cycle nodes (CLIP)", str(len(cycle_nodes_c)))
    table.add_row("Cycle nodes (intersection)", str(len(intersection)))

    console.print(table)

    # Per-class agreement
    console.print("\n[bold]NN Agreement by Class:[/bold]")
    cls_table = Table(show_header=True, header_style="bold")
    cls_table.add_column("Class", style="cyan")
    cls_table.add_column("Agreement", justify="right")
    for cls, rate in sorted(agreement_by_class.items(), key=lambda x: -x[1]):
        cls_table.add_row(cls, f"{rate*100:.1f}%")
    console.print(cls_table)

    out_path = results_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    console.print(f"\n[green]✓[/green] Saved to {out_path}")

    return comparison
