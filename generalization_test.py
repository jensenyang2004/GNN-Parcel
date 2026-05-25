"""
Generalization Test Across Checkpoints
=======================================
Sweeps a list of trained checkpoints over a list of agent counts on a SINGLE
full-size map. Produces:
  - A console table of CR (with 95% CI) and SR per (checkpoint, N)
  - A CSV matrix with the same data
  - A line plot: x = number of agents, y = completion rate,
    one line per checkpoint, shaded 95% CI band
  - A raw JSON dump of per-episode CR/SR for later re-plotting

No CLI arguments — edit the CONFIG block below and run:
    python generalization_test.py
"""

import os
import sys
import json
import csv
import random
import time

import torch
import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_parcel import load_map
from evaluate import load_actor, make_env, run_episode


# ============================================================================
# CONFIG — edit these
# ============================================================================

# Single full-size map per run
MAP_PATH = "instances/city.map"

# Agent counts to sweep on the x-axis
AGENT_COUNTS = [64, 100, 200, 600]

# Checkpoints to compare (one line per entry).
# Each entry: label, model_dir (folder containing actor_net.pth),
# critic_type ("gnn" | "attention"), gnn_layers (only used for gnn).
CHECKPOINTS = [

    # {
    #     "label":       "EDGE GNN-2L (200)",
    #     "model_dir":   "exp_checkpoint/edge_gnn_l2_warehouse_200",
    #     "critic_type": "gnn",
    #     "gnn_layers":  2,
    # },
    {
        "label":       "Edge GNN-1L (64)",
        "model_dir":   "output/edge_gnn_l1_mix_64/best",
        "critic_type": "gnn",
        "gnn_layers":  1,
    },
    # {
    #     "label":       "Edge GNN-2L (64)",
    #     "model_dir":   "exp_checkpoint/edge_gnn_l2_64",
    #     "critic_type": "gnn",
    #     "gnn_layers":  2,
    # },
    # {
    #     "label":       "GNN-2L (64)",
    #     "model_dir":   "exp_checkpoint/gnn_l2_warehouse_64_2",
    #     "critic_type": "gnn",
    #     "gnn_layers":  2,
    # },
    # {
    #     "label":       "Attention (64)",
    #     "model_dir":   "exp_checkpoint/attention_warehouse_64_2",
    #     "critic_type": "attention",
    #     "gnn_layers":  2,   # ignored for attention
    # },
]

# Episodes per (checkpoint, N) — higher = tighter CI
N_EPISODES = 25

# Episode horizon. Bump above 256 for big maps so agents don't just time out.
TIME_LIMIT = 256

SEED = 42
DEVICE = "cpu"   # "cuda" / "mps" if you have it

# Action selection mode at eval time.
#   True  -> greedy argmax (deterministic; conventional benchmark setting)
#   False -> sample from softmax (breaks symmetric-tie corridor deadlocks)
GREEDY = True

# Output paths
OUTPUT_DIR  = "output/generalization_city_map"
PLOT_FILE   = "generalization.png"
CSV_FILE    = "generalization.csv"
JSON_FILE   = "generalization_raw.json"


# ============================================================================
# Runner
# ============================================================================

def evaluate_checkpoint(label, model_dir, critic_type, gnn_layers,
                        base_map, agent_counts, n_episodes,
                        time_limit, device, greedy=True):
    """
    Returns dict[N] -> {crs: [...], srs: [...], cr_mean, cr_std, sr_mean, n}
    or dict[N] -> None for configs that were skipped / failed.
    """
    results = {}

    actor_path = os.path.join(model_dir, "actor_net.pth")
    if not os.path.exists(actor_path):
        print(f"  [{label}] SKIPPED: {actor_path} not found")
        return {n: None for n in agent_counts}

    free_cells = sum(1 for r in base_map for c in r if c == 0)

    for n in agent_counts:
        if n * 2 > free_cells:
            print(f"  [{label}] N={n:4d}: skip (only {free_cells} free cells)")
            results[n] = None
            continue

        try:
            controller = load_actor(model_dir, n, critic_type, gnn_layers, device)
        except Exception as e:
            print(f"  [{label}] N={n:4d}: load failed - {e}")
            results[n] = None
            continue

        env = make_env(base_map, n, time_limit, device)

        crs, srs = [], []
        t0 = time.time()
        for _ in range(n_episodes):
            try:
                cr, sr = run_episode(env, controller, greedy=greedy)
                crs.append(float(cr))
                srs.append(float(sr))
            except Exception:
                continue
        elapsed = time.time() - t0

        if not crs:
            results[n] = None
            print(f"  [{label}] N={n:4d}: no valid episodes")
            continue

        crs_arr = numpy.array(crs)
        srs_arr = numpy.array(srs)
        n_valid = len(crs)
        cr_mean = float(crs_arr.mean())
        cr_std  = float(crs_arr.std())
        sr_mean = float(srs_arr.mean())
        ci95    = 1.96 * cr_std / max(numpy.sqrt(n_valid), 1)

        results[n] = {
            "crs":     crs,
            "srs":     srs,
            "cr_mean": cr_mean,
            "cr_std":  cr_std,
            "ci95":    ci95,
            "sr_mean": sr_mean,
            "n":       n_valid,
        }
        print(f"  [{label}] N={n:4d}  CR={cr_mean:.3f}±{ci95:.3f}  "
              f"SR={sr_mean:.3f}  ({n_valid}/{n_episodes} eps)  [{elapsed:.1f}s]")

    return results


# ============================================================================
# Output helpers
# ============================================================================

def write_csv(all_results, agent_counts, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "metric"] + [f"N={n}" for n in agent_counts])
        for label, res in all_results.items():
            cr_row = [label, "CR_mean"]
            ci_row = [label, "CR_95CI"]
            sr_row = [label, "SR_mean"]
            for n in agent_counts:
                r = res.get(n)
                if r is None:
                    cr_row.append("")
                    ci_row.append("")
                    sr_row.append("")
                else:
                    cr_row.append(f"{r['cr_mean']:.4f}")
                    ci_row.append(f"{r['ci95']:.4f}")
                    sr_row.append(f"{r['sr_mean']:.4f}")
            w.writerow(cr_row)
            w.writerow(ci_row)
            w.writerow(sr_row)


def print_table(all_results, agent_counts, map_path):
    map_short = os.path.basename(map_path).replace(".map", "")
    col_w = 14
    header = f"{'Checkpoint':<30}" + "".join(f"{'N='+str(n):^{col_w}}" for n in agent_counts)
    print(f"\n{'='*len(header)}")
    print(f"Completion Rate ± 95% CI    map={map_short}    eps={N_EPISODES}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for label, res in all_results.items():
        row = f"{label[:29]:<30}"
        for n in agent_counts:
            r = res.get(n)
            if r is None:
                row += f"{'—':^{col_w}}"
            else:
                row += f"{r['cr_mean']:.2f}±{r['ci95']:.2f}".center(col_w)
        print(row)
    print()


def plot_lines(all_results, agent_counts, map_path, plot_path):
    map_short = os.path.basename(map_path).replace(".map", "")
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (label, res) in enumerate(all_results.items()):
        xs, ys, lo, hi = [], [], [], []
        for n in agent_counts:
            r = res.get(n)
            if r is None:
                continue
            xs.append(n)
            ys.append(r["cr_mean"])
            lo.append(r["cr_mean"] - r["ci95"])
            hi.append(r["cr_mean"] + r["ci95"])
        if not xs:
            continue
        line, = ax.plot(xs, ys, marker=markers[i % len(markers)],
                        label=label, linewidth=2, markersize=7)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())

    mode_tag = "greedy" if GREEDY else "stochastic"
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Completion rate")
    ax.set_title(f"Generalization on {map_short} — {mode_tag} actions "
                 f"(full size, 95% CI, {N_EPISODES} eps)")
    ax.set_xticks(agent_counts)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Map not found: {MAP_PATH}")

    base_map = load_map(MAP_PATH)
    free_cells = sum(1 for r in base_map for c in r if c == 0)
    action_mode = "greedy (argmax)" if GREEDY else "stochastic (sampled)"
    print(f"Map: {MAP_PATH}")
    print(f"  size={len(base_map)}x{len(base_map[0])}  free cells={free_cells}")
    print(f"Agents: {AGENT_COUNTS}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Episodes per cell: {N_EPISODES}  |  Time limit: {TIME_LIMIT}")
    print(f"Action mode: {action_mode}\n")

    all_results = {}
    t_start = time.time()
    for ckpt in CHECKPOINTS:
        label = ckpt["label"]
        print(f"--- {label}  ({ckpt['model_dir']}) ---")
        all_results[label] = evaluate_checkpoint(
            label        = label,
            model_dir    = ckpt["model_dir"],
            critic_type  = ckpt["critic_type"],
            gnn_layers   = ckpt.get("gnn_layers", 2),
            base_map     = base_map,
            agent_counts = AGENT_COUNTS,
            n_episodes   = N_EPISODES,
            time_limit   = TIME_LIMIT,
            device       = device,
            greedy       = GREEDY,
        )
    total_time = time.time() - t_start

    # Console matrix
    print_table(all_results, AGENT_COUNTS, MAP_PATH)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILE)
    write_csv(all_results, AGENT_COUNTS, csv_path)
    print(f"CSV  : {csv_path}")

    # Raw JSON (keeps per-episode arrays so you can re-plot without rerunning)
    json_path = os.path.join(OUTPUT_DIR, JSON_FILE)
    serializable = {
        "map": MAP_PATH,
        "agent_counts": AGENT_COUNTS,
        "n_episodes": N_EPISODES,
        "time_limit": TIME_LIMIT,
        "seed": SEED,
        "action_mode": "greedy" if GREEDY else "stochastic",
        "checkpoints": CHECKPOINTS,
        "results": {
            label: {str(n): r for n, r in res.items()}
            for label, res in all_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"JSON : {json_path}")

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, PLOT_FILE)
    plot_lines(all_results, AGENT_COUNTS, MAP_PATH, plot_path)
    print(f"Plot : {plot_path}")

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
