"""
PARCEL Headless Evaluation
==========================
Tests a trained actor's generalization across different map sizes and agent counts.
The actor is a per-agent FFN — its weights are independent of N and map size,
so the same checkpoint can be evaluated on any configuration.

Usage:
    # Basic: test the model it was trained with (16 agents, 64x64)
    python evaluate.py --model_dir output/gnn2_a64 --critic_type gnn --gnn_layers 2

    # Sweep map sizes and agent counts
    python evaluate.py --model_dir output/gnn2_fixed --critic_type gnn --gnn_layers 2 \\
        --map_sizes 32 48 64 --agent_counts 4 8 16 32

    # Both maps, more episodes for tighter estimates
    python evaluate.py --model_dir output/gnn2_fixed --critic_type gnn --gnn_layers 2 \\
        --maps instances/random-64-64-10.map instances/warehouse.map \\
        --agent_counts 4 8 16 32 64 --n_episodes 100

    # Save results to JSON
    python evaluate.py --model_dir output/gnn2_fixed --critic_type gnn --gnn_layers 2 \\
        --output results.json
"""

import os
import sys
import json
import argparse
import random
import time
import torch
import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cactus.constants import (
    ENV_NR_AGENTS, ENV_NR_ACTIONS, ENV_OBSERVATION_DIM, ENV_OBSTACLES,
    ENV_TIME_LIMIT, ENV_GAMMA, ENV_OBSERVATION_SIZE, TORCH_DEVICE,
    ENV_INIT_GOAL_RADIUS, ENV_COMPLETION_RATE, NR_GRID_ACTIONS,
    HIDDEN_LAYER_DIM, NR_ATTENTION_HEADS, GRAD_NORM_CLIP, LEARNING_RATE,
    CLIP_RATIO, UPDATE_ITERATIONS, WAIT, ACTOR_NET_FILENAME, EPISODES_PER_EPOCH
)
from cactus.env.mapf_gridworld import MAPFGridWorld
from parcel.parcel_controller import (
    PARCELController, PARCEL_RALLOC, PARCEL_ROWS, PARCEL_COLS,
    PARCEL_OBSTACLE_MAP, PARCEL_CRITIC_TYPE
)
from parcel.gnn_critic import GNN_NR_LAYERS, GNN_EMBED_DIM
from train_parcel import load_map, crop_subgrid


# -----------------------------------------------------------------------
# Actor loader — creates a controller sized for eval_nr_agents but loads
# actor weights trained with any nr_agents (FFN is agent-count agnostic)
# -----------------------------------------------------------------------

OBS_DIM = [5, 7, 7]  # fixed: 5 channels, 7x7 local view


def load_actor(model_dir, eval_nr_agents, critic_type, gnn_layers, device):
    """
    Build a PARCELController sized for eval_nr_agents and load the trained actor.
    The critic is a dummy — only the actor is used at eval time.
    """
    params = {
        ENV_NR_AGENTS: eval_nr_agents,
        ENV_NR_ACTIONS: NR_GRID_ACTIONS,
        ENV_OBSERVATION_DIM: OBS_DIM,
        TORCH_DEVICE: device,
        EPISODES_PER_EPOCH: 1,
        ENV_TIME_LIMIT: 256,
        ENV_GAMMA: 1,
        HIDDEN_LAYER_DIM: 64,
        GNN_EMBED_DIM: 64,
        GNN_NR_LAYERS: gnn_layers,
        "attention_embed_dim": 64,
        NR_ATTENTION_HEADS: 2,
        LEARNING_RATE: 0.001,
        CLIP_RATIO: 0.1,
        UPDATE_ITERATIONS: 1,
        GRAD_NORM_CLIP: 1.0,
        # Dummy map info — not used at eval time (no grouping needed)
        PARCEL_ROWS: 64,
        PARCEL_COLS: 64,
        PARCEL_OBSTACLE_MAP: torch.zeros(64, 64, dtype=torch.bool),
        PARCEL_RALLOC: 9999,
        PARCEL_CRITIC_TYPE: critic_type,
        "output_dim": NR_GRID_ACTIONS,
    }
    controller = PARCELController(params)
    actor_path = os.path.join(model_dir, ACTOR_NET_FILENAME)
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"No actor weights found at {actor_path}")
    controller.policy_network.load_state_dict(
        torch.load(actor_path, map_location=device, weights_only=True)
    )
    controller.policy_network.eval()
    return controller


def make_env(obstacle_list, nr_agents, time_limit, device):
    return MAPFGridWorld({
        ENV_OBSTACLES: torch.tensor(obstacle_list, dtype=torch.bool),
        ENV_NR_AGENTS: nr_agents,
        ENV_TIME_LIMIT: time_limit,
        ENV_GAMMA: 1,
        ENV_OBSERVATION_SIZE: 7,
        TORCH_DEVICE: device,
        ENV_INIT_GOAL_RADIUS: None,  # goals anywhere — full generalization test
    })


# -----------------------------------------------------------------------
# Single episode runner
# -----------------------------------------------------------------------

def run_episode(env, controller):
    obs = env.reset()
    done = False
    while not done:
        action = controller.joint_policy(obs, greedy=True)
        action[env.is_terminated()] = WAIT  # freeze agents already at goal
        obs, _, _, _, info = env.step(action)
        done = env.is_done_all()
    cr = info[ENV_COMPLETION_RATE]
    sr = float(env.is_terminated().all().item())
    return cr, sr


# -----------------------------------------------------------------------
# Per-configuration evaluation
# -----------------------------------------------------------------------

def evaluate_config(base_map, map_size, nr_agents, controller, n_episodes, time_limit, device):
    """
    Run n_episodes on random crops of size map_size x map_size with nr_agents agents.
    Returns dict with mean/std CR and SR.
    """
    crs, srs = [], []
    for _ in range(n_episodes):
        crop = crop_subgrid(base_map, map_size)
        # Check the crop has enough free cells for the agents
        free_cells = sum(1 for r in crop for c in r if c == 0)
        if free_cells < nr_agents * 2:
            # Map too dense for this many agents — skip episode
            continue
        env = make_env(crop, nr_agents, time_limit, device)
        try:
            cr, sr = run_episode(env, controller)
            crs.append(cr)
            srs.append(sr)
        except Exception:
            # e.g. no valid start positions for this density
            continue

    if not crs:
        return None  # no valid episodes for this config

    crs = numpy.array(crs)
    srs = numpy.array(srs)
    return {
        "n_valid": len(crs),
        "cr_mean": float(crs.mean()),
        "cr_std":  float(crs.std()),
        "sr_mean": float(srs.mean()),
        "sr_std":  float(srs.std()),
    }


# -----------------------------------------------------------------------
# Table printer
# -----------------------------------------------------------------------

def print_table(results, map_name, map_sizes, agent_counts):
    short_name = os.path.basename(map_name.replace(".map", ""))
    print(f"\n{'='*72}")
    print(f"Map: {short_name}")
    print(f"{'='*72}")

    # Header
    col_w = 14
    header = f"{'Agents':>7} |" + "".join(f"{'Crop '+str(s):^{col_w}}" for s in map_sizes)
    print(header)
    print(f"{'-'*7}-+" + "-"*col_w*len(map_sizes))
    subheader = f"{'':>7} |" + "".join(f"{'CR    SR':^{col_w}}" for _ in map_sizes)
    print(subheader)
    print(f"{'-'*7}-+" + "-"*col_w*len(map_sizes))

    for n in agent_counts:
        row = f"{n:>7} |"
        for s in map_sizes:
            key = (map_name, s, n)
            r = results.get(key)
            if r is None:
                row += f"{'  —  ':^{col_w}}"
            else:
                cell = f"{r['cr_mean']:.2f}  {r['sr_mean']:.2f}"
                row += f"{cell:^{col_w}}"
        print(row)
    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Headless generalization evaluation for PARCEL actors"
    )

    # Model
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing actor_net.pth")
    parser.add_argument("--critic_type", type=str, default="gnn",
                        choices=["attention", "gnn"])
    parser.add_argument("--gnn_layers", type=int, default=2)

    # Maps
    parser.add_argument("--maps", type=str, nargs="+",
                        default=["instances/random-64-64-10.map"],
                        help="Map files to evaluate on")

    # Sweep parameters
    parser.add_argument("--map_sizes", type=int, nargs="+",
                        default=[32, 48, 64],
                        help="Crop sizes to test (grid width=height)")
    parser.add_argument("--agent_counts", type=int, nargs="+",
                        default=[4, 8, 16, 32],
                        help="Agent counts to test")

    # Episode settings
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="Episodes per (map, crop_size, n_agents) configuration")
    parser.add_argument("--time_limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    # --- Load maps ---
    print("Loading maps...")
    base_maps = {}
    for map_path in args.maps:
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Map not found: {map_path}\nRun: python generate_maps.py"
            )
        base_maps[map_path] = load_map(map_path)
        m = base_maps[map_path]
        print(f"  {map_path}: {len(m)}x{len(m[0])}")

    # --- Summary of sweep ---
    total_configs = len(args.maps) * len(args.map_sizes) * len(args.agent_counts)
    total_episodes = total_configs * args.n_episodes
    print(f"\nSweep: {len(args.maps)} maps × {len(args.map_sizes)} sizes"
          f" × {len(args.agent_counts)} agent counts"
          f" = {total_configs} configs × {args.n_episodes} episodes"
          f" = {total_episodes} total episodes\n")

    # --- Evaluate each config ---
    results = {}
    t0 = time.time()

    for map_path, base_map in base_maps.items():
        map_short = os.path.basename(map_path)
        for nr_agents in args.agent_counts:
            # Load actor fresh for each agent count (controller sizing)
            controller = load_actor(
                args.model_dir, nr_agents, args.critic_type, args.gnn_layers, device
            )
            for map_size in args.map_sizes:
                # Skip trivially impossible configs (too many agents for the crop)
                max_free = map_size * map_size  # rough upper bound
                if nr_agents * 4 > max_free:
                    print(f"  Skip {map_short} | size={map_size} | agents={nr_agents}"
                          f"  (too dense)")
                    continue

                t1 = time.time()
                r = evaluate_config(
                    base_map, map_size, nr_agents,
                    controller, args.n_episodes, args.time_limit, device
                )
                elapsed = time.time() - t1

                key = (map_path, map_size, nr_agents)
                results[key] = r

                if r is None:
                    status = "no valid episodes"
                else:
                    status = (f"CR={r['cr_mean']:.3f}±{r['cr_std']:.3f}"
                              f"  SR={r['sr_mean']:.3f}±{r['sr_std']:.3f}"
                              f"  ({r['n_valid']}/{args.n_episodes} episodes)"
                              f"  [{elapsed:.1f}s]")
                print(f"  {map_short:30s} | size={map_size:3d} | agents={nr_agents:3d} | {status}")

    total_time = time.time() - t0

    # --- Print tables ---
    for map_path in base_maps:
        print_table(results, map_path, args.map_sizes, args.agent_counts)

    print(f"Total evaluation time: {total_time:.1f}s")

    # --- Save JSON ---
    if args.output:
        serializable = {}
        for (map_path, map_size, nr_agents), r in results.items():
            k = f"{os.path.basename(map_path)}|size={map_size}|agents={nr_agents}"
            serializable[k] = r
        with open(args.output, "w") as f:
            json.dump({
                "model_dir": args.model_dir,
                "critic_type": args.critic_type,
                "n_episodes": args.n_episodes,
                "time_limit": args.time_limit,
                "seed": args.seed,
                "results": serializable,
            }, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
