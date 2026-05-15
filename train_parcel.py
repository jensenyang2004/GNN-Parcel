"""
PARCEL Training Script — Matching Paper Setup (Section 5)
==========================================================
Training setup per Section 5.2:
  - Maps: Random-64-64-10 and Warehouse (movingai benchmark format)
  - Training: crop random 64x64 sub-grids, rotated/mirrored randomly
  - W = 2000 epochs, Y = 10 episodes/epoch, T = 256 horizon
  - N = 16 or 64 agents
  - U = 0.5 (50% threshold), eta = 2, sliding_window = 50
  - lr = 0.001, 2 attention heads, z = 64

Usage:
    python train_parcel.py --nr_agents 16 --epochs 2000
    python train_parcel.py --nr_agents 64 --epochs 2000
"""

import os
import sys
import argparse
import random
import time
import torch
import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cactus.constants import *
from cactus.curriculum import CACTUSCurriculum
from cactus.data import save_json
from cactus.env.mapf_gridworld import MAPFGridWorld

from parcel.parcel_controller import (
    PARCELController, PARCEL_RALLOC, PARCEL_ROWS, PARCEL_COLS, PARCEL_OBSTACLE_MAP
)


# -----------------------------------------------------------------------
# Map loading and sub-grid cropping (Section 5.2)
# -----------------------------------------------------------------------

def load_map(filename):
    """Load a movingai .map file. Returns 2D list (1=obstacle, 0=free)."""
    with open(filename, "r") as f:
        lines = f.readlines()
    obstacles = []
    for line in lines[4:]:  # skip 4-line header (type/height/width/map)
        row = [1 if c == "@" else 0 for c in line.strip()]
        if row:
            obstacles.append(row)
    return obstacles


def crop_subgrid(obstacle_map, crop_size=64):
    """
    Crop a random crop_size x crop_size sub-grid from the full map,
    then randomly rotate (0/90/180/270 degrees) and mirror (yes/no).

    Matches paper Section 5.2:
    "crop random 64x64 sub-grids of Random and Warehouse,
     which are rotated and mirrored randomly"
    """
    rows = len(obstacle_map)
    cols = len(obstacle_map[0])

    # Pad with walls if map is smaller than crop_size
    if rows < crop_size or cols < crop_size:
        pad_r = max(0, crop_size - rows)
        pad_c = max(0, crop_size - cols)
        obstacle_map = [row + [1] * pad_c for row in obstacle_map]
        obstacle_map = obstacle_map + [[1] * len(obstacle_map[0])] * pad_r
        rows = len(obstacle_map)
        cols = len(obstacle_map[0])

    # Random crop origin
    r0 = random.randint(0, rows - crop_size)
    c0 = random.randint(0, cols - crop_size)
    grid = [obstacle_map[r0 + r][c0:c0 + crop_size] for r in range(crop_size)]

    # Rotate and mirror randomly
    arr = numpy.array(grid)
    arr = numpy.rot90(arr, random.randint(0, 3))
    if random.random() < 0.5:
        arr = numpy.fliplr(arr)

    return arr.tolist()


def make_env_from_obstacles(obstacle_list, nr_agents, time_limit, device):
    """Create a MAPFGridWorld from a 2D obstacle list."""
    return MAPFGridWorld({
        ENV_OBSTACLES: torch.tensor(obstacle_list, dtype=torch.bool),
        ENV_NR_AGENTS: nr_agents,
        ENV_TIME_LIMIT: time_limit,
        ENV_GAMMA: 1,
        ENV_OBSERVATION_SIZE: 7,
        TORCH_DEVICE: device,
        ENV_INIT_GOAL_RADIUS: 2,
    })


class CroppingMapEnv:
    """
    Wrapper that generates a fresh random sub-grid crop at each reset().
    Gives training diversity across different map regions, orientations,
    and both Random + Warehouse maps.
    """

    def __init__(self, base_maps, nr_agents, time_limit, device, crop_size=64):
        self.base_maps = base_maps
        self.nr_agents = nr_agents
        self.time_limit = time_limit
        self.device = device
        self.crop_size = crop_size
        self._refresh_env()

    def _refresh_env(self):
        base_map = random.choice(self.base_maps)
        crop = crop_subgrid(base_map, self.crop_size)
        self._env = make_env_from_obstacles(crop, self.nr_agents, self.time_limit, self.device)

    def reset(self):
        self._refresh_env()
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


# -----------------------------------------------------------------------
# PARCEL episode runner
# -----------------------------------------------------------------------

def run_episode_parcel(env, controller, training_mode=True):
    obs = env.reset()

    # Notify controller of start positions for spatial grouping (Section 4.1)
    if training_mode:
        start_positions = [
            (env.current_positions[a, 0].item(), env.current_positions[a, 1].item())
            for a in range(env.nr_agents)
        ]
        controller.notify_episode_start(start_positions)

    done = False
    info = {ENV_COMPLETION_RATE: 0.0}
    while not done:
        joint_action = controller.joint_policy(obs)
        next_obs, rewards, terminated, truncated, info = env.step(joint_action)
        done = env.is_done_all()
        if training_mode:
            controller.update(obs, joint_action, rewards, terminated, truncated, done, info)
        obs = next_obs

    return {
        COMPLETION_RATE: info[ENV_COMPLETION_RATE],
        TERMINATED: env.is_terminated().all(),
    }


def run_episodes_parcel(nr_episodes, envs, controller, training_mode=True):
    completion_sum = 0.0
    successes = 0.0
    for _ in range(nr_episodes):
        env = random.choice(envs)
        result = run_episode_parcel(env, controller, training_mode)
        completion_sum += result[COMPLETION_RATE]
        if result[TERMINATED]:
            successes += 1
    sr = successes / nr_episodes
    return {
        SUCCESS_RATE: sr,
        SUCCESS_RATE_VARIANCE: sr * (1.0 - sr),
        COMPLETION_RATE: completion_sum / nr_episodes,
    }


def test_run_parcel(test_envs, controller):
    """Evaluate on full-map goal distance (no radius restriction)."""
    completion_sum = 0.0
    successes = 0.0
    for env in test_envs:
        backup = env._env.init_goal_radius
        env._env.set_init_goal_radius(None)
        result = run_episode_parcel(env, controller, training_mode=False)
        completion_sum += result[COMPLETION_RATE]
        if result[TERMINATED]:
            successes += 1
        env._env.set_init_goal_radius(backup)
    n = len(test_envs)
    sr = successes / n
    return {
        SUCCESS_RATE: sr,
        SUCCESS_RATE_VARIANCE: sr * (1.0 - sr),
        COMPLETION_RATE: completion_sum / n,
    }


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run_training_parcel(train_envs, test_envs, controller, params):
    inner_envs = [e._env for e in train_envs]
    curriculum = CACTUSCurriculum(inner_envs, params)

    episodes_per_epoch = params[EPISODES_PER_EPOCH]
    nr_epochs = params[NUMBER_OF_EPOCHS]
    log_interval = params[EPOCH_LOG_INTERVAL]
    directory = params.get(DIRECTORY, None)

    success_rates, completion_rates, training_times = [], [], []
    total_time = 0.0
    prev_total_time = 0.0
    training_result = {COMPLETION_RATE: 0.0, SUCCESS_RATE_VARIANCE: 0.0}

    for epoch in range(nr_epochs + 1):
        start = time.time()

        curriculum.update_curriculum(
            training_result[COMPLETION_RATE],
            training_result[SUCCESS_RATE_VARIANCE]
        )
        # Sync ralloc to controller for spatial grouping
        controller.set_ralloc(curriculum.radius)

        training_result = run_episodes_parcel(
            episodes_per_epoch, train_envs, controller, training_mode=True
        )
        total_time += time.time() - start

        if epoch % log_interval == 0:
            training_time = total_time - prev_total_time
            prev_total_time = total_time
            test_result = test_run_parcel(test_envs, controller)
            print(f"Epoch {epoch:4d} | Ralloc={curriculum.radius:3d} | "
                  f"Train CR={training_result[COMPLETION_RATE]:.3f} | "
                  f"Test CR={test_result[COMPLETION_RATE]:.3f} | "
                  f"Test SR={test_result[SUCCESS_RATE]:.3f} | "
                  f"Time={training_time:.1f}s")
            success_rates.append(float(test_result[SUCCESS_RATE]))
            completion_rates.append(float(test_result[COMPLETION_RATE]))
            training_times.append(training_time)

        if epoch > 0 and epoch % 500 == 0 and directory:
            os.makedirs(directory, exist_ok=True)
            controller.save_model_weights(directory)
            print(f"  [Checkpoint saved at epoch {epoch}]")

    result = {
        TOTAL_TIME: total_time,
        TIME_PER_EPOCH: total_time / max(nr_epochs, 1),
        SUCCESS_RATE: success_rates,
        COMPLETION_RATE: completion_rates,
        TRAINING_TIME: training_times,
    }
    if directory:
        os.makedirs(directory, exist_ok=True)
        controller.save_model_weights(directory)
        save_json(os.path.join(directory, "results.json"), result)
        print(f"\nModel saved to: {directory}")
    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PARCEL — paper-accurate setup")

    # Paper Section 5.2 exact settings
    parser.add_argument("--nr_agents", type=int, default=16,
                        help="N: number of agents (paper: 16 or 64)")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="W: training epochs (paper: 2000)")
    parser.add_argument("--episodes_per_epoch", type=int, default=10,
                        help="Y: episodes per epoch (paper: 10)")
    parser.add_argument("--time_limit", type=int, default=256,
                        help="T: horizon per episode (paper: 256)")

    # Maps
    parser.add_argument("--random_map", type=str,
                        default="instances/random-64-64-10.map",
                        help="Path to Random-64-64-10 map file")
    parser.add_argument("--warehouse_map", type=str,
                        default="instances/warehouse.map",
                        help="Path to Warehouse map file")
    parser.add_argument("--crop_size", type=int, default=64,
                        help="Sub-grid crop size (paper: 64x64)")
    parser.add_argument("--nr_train_envs", type=int, default=4)
    parser.add_argument("--nr_test_envs", type=int, default=4)

    # Curriculum — paper: U=0.5, eta=2
    parser.add_argument("--improvement_threshold", type=float, default=0.5,
                        help="U: curriculum threshold (paper: 0.5 = 50%%)")
    parser.add_argument("--deviation_factor", type=float, default=2.0,
                        help="eta: deviation factor (paper: 2)")
    parser.add_argument("--sliding_window", type=int, default=50,
                        help="Sliding window size for curriculum (paper: 50)")

    # Network — paper: hidden=64, embed z=64, heads=2
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--nr_heads", type=int, default=2)

    # Optimizer — paper: lr=0.001
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip_ratio", type=float, default=0.1)
    parser.add_argument("--update_iterations", type=int, default=4)

    # Misc
    parser.add_argument("--output_dir", type=str, default="output/parcel")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load benchmark maps ---
    print("\nLoading maps...")
    for path in [args.random_map, args.warehouse_map]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Map not found: {path}\n"
                f"Run: python generate_maps.py  to generate the benchmark maps first."
            )
    random_obstacles = load_map(args.random_map)
    warehouse_obstacles = load_map(args.warehouse_map)
    base_maps = [random_obstacles, warehouse_obstacles]
    print(f"  Random map:    {len(random_obstacles)}x{len(random_obstacles[0])}")
    print(f"  Warehouse map: {len(warehouse_obstacles)}x{len(warehouse_obstacles[0])}")

    # --- Create environments ---
    train_envs = [
        CroppingMapEnv(base_maps, args.nr_agents, args.time_limit, device, args.crop_size)
        for _ in range(args.nr_train_envs)
    ]
    test_envs = [
        CroppingMapEnv(base_maps, args.nr_agents, args.time_limit, device, args.crop_size)
        for _ in range(args.nr_test_envs)
    ]
    obs_dim = train_envs[0]._env.observation_dim  # [5, 7, 7]
    print(f"  Observation dim: {obs_dim}")

    # --- Controller ---
    controller_params = {
        ENV_NR_AGENTS: args.nr_agents,
        ENV_NR_ACTIONS: NR_GRID_ACTIONS,
        ENV_OBSERVATION_DIM: obs_dim,
        TORCH_DEVICE: device,
        EPISODES_PER_EPOCH: args.episodes_per_epoch,
        ENV_TIME_LIMIT: args.time_limit,
        ENV_GAMMA: 1,
        HIDDEN_LAYER_DIM: args.hidden_dim,
        "attention_embed_dim": args.embed_dim,
        NR_ATTENTION_HEADS: args.nr_heads,
        LEARNING_RATE: args.lr,
        CLIP_RATIO: args.clip_ratio,
        UPDATE_ITERATIONS: args.update_iterations,
        GRAD_NORM_CLIP: 1.0,
        PARCEL_ROWS: train_envs[0]._env.rows,
        PARCEL_COLS: train_envs[0]._env.columns,
        PARCEL_OBSTACLE_MAP: train_envs[0]._env.obstacle_map,
        PARCEL_RALLOC: 2,
        "verbose": args.verbose,
    }

    controller = PARCELController(controller_params)
    controller.set_map_info(
        train_envs[0]._env.rows,
        train_envs[0]._env.columns,
        train_envs[0]._env.obstacle_map,
    )

    print(f"\n[PARCEL] Controller:")
    print(f"  Agents:           {args.nr_agents}")
    print(f"  Parameters:       {controller.get_parameter_count():,}  (paper: <600,000)")
    print(f"  Attention heads:  {args.nr_heads}, embed dim: {args.embed_dim}")

    # --- Training params ---
    train_params = {
        NUMBER_OF_EPOCHS: args.epochs,
        EPISODES_PER_EPOCH: args.episodes_per_epoch,
        EPOCH_LOG_INTERVAL: args.log_interval,
        CURRICULUM_NAME: CACTUS_CURRICULUM,
        RADIUS_UPDATE_INTERVAL: 1,
        IMPROVEMENT_THRESHOLD: args.improvement_threshold,
        DEVIATION_FACTOR: args.deviation_factor,
        SLIDING_WINDOW_SIZE: args.sliding_window,
        TEST_INIT_GOAL_RADIUS: None,
        ALGORITHM_NAME: "PARCEL",
        DIRECTORY: args.output_dir,
        RENDER_MODE: False,
    }

    print(f"\nTraining (paper Section 5.2):")
    print(f"  W={args.epochs} epochs × Y={args.episodes_per_epoch} episodes"
          f" = {args.epochs * args.episodes_per_epoch:,} total episodes")
    print(f"  T={args.time_limit} steps/episode")
    print(f"  Curriculum: U={args.improvement_threshold}, eta={args.deviation_factor}")
    print(f"  Maps: Random-64-64-10 + Warehouse → cropped {args.crop_size}×{args.crop_size}")
    print("=" * 60)

    result = run_training_parcel(train_envs, test_envs, controller, train_params)

    print("\n" + "=" * 60)
    print("Training complete.")
    if result[SUCCESS_RATE]:
        print(f"Final test success rate:    {result[SUCCESS_RATE][-1]:.3f}")
        print(f"Final test completion rate: {result[COMPLETION_RATE][-1]:.3f}")
    print(f"Total training time:        {result[TOTAL_TIME]:.1f}s")


if __name__ == "__main__":
    main()