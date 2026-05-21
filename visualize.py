"""
PARCEL Visualization
====================
Loads a trained actor and runs it live in a pygame window.

Usage:
    python visualize.py --model_dir output/gnn2 --critic_type gnn --gnn_layers 2
    python visualize.py --model_dir output/attn --critic_type attention
    python visualize.py --model_dir output/gnn2 --critic_type gnn --map instances/random-64-64-10.map
    python visualize.py --model_dir output/gnn2 --critic_type gnn --nr_agents 8 --fps 5

Controls:
    ESC / close window  — quit
    R                   — reset episode immediately
"""

import os
import sys
import argparse
import random
import torch
import numpy
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cactus.constants import *
from cactus.env.mapf_gridworld import MAPFGridWorld
from cactus.rendering.gridworld_viewer import GridworldViewer

from parcel.parcel_controller import (
    PARCELController, PARCEL_RALLOC, PARCEL_ROWS, PARCEL_COLS,
    PARCEL_OBSTACLE_MAP, PARCEL_CRITIC_TYPE
)
from cactus.constants import WAIT
from parcel.gnn_critic import GNN_NR_LAYERS, GNN_EMBED_DIM
from train_parcel import load_map, crop_subgrid


# -----------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------

def make_env(obstacle_list, nr_agents, time_limit, device):
    return MAPFGridWorld({
        ENV_OBSTACLES: torch.tensor(obstacle_list, dtype=torch.bool),
        ENV_NR_AGENTS: nr_agents,
        ENV_TIME_LIMIT: time_limit,
        ENV_GAMMA: 1,
        ENV_OBSERVATION_SIZE: 7,
        TORCH_DEVICE: device,
        ENV_INIT_GOAL_RADIUS: None,  # goals anywhere on map
    })


# -----------------------------------------------------------------------
# Extended viewer with goal labels and step counter
# -----------------------------------------------------------------------

class PARCELViewer(GridworldViewer):

    def __init__(self, width, height, cell_size=12, fps=8):
        super().__init__(width, height, cell_size, fps)
        pygame.display.set_caption("PARCEL — agent visualization")
        self._font = None

    def _get_font(self):
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", max(10, self.cell_size - 2), bold=True)
        return self._font

    def draw_state(self, env, step=0, completion_rate=0.0):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        self.screen.fill(BLACK)
        cs = self.cell_size

        # Draw grid
        for x in range(env.rows):
            for y in range(env.columns):
                if not env.obstacle_map[x][y]:
                    pygame.draw.rect(
                        self.screen, WHITE,
                        pygame.Rect(x * cs + 1, y * cs + 1, cs - 2, cs - 2), 0
                    )

        # Draw goal squares (filled rectangle, agent color, slightly inset)
        for agent_id in range(env.nr_agents):
            if env.is_done()[agent_id]:
                continue
            gx = env.goal_positions[agent_id, 0].item()
            gy = env.goal_positions[agent_id, 1].item()
            color = self.agent_color(agent_id)
            margin = max(2, cs // 4)
            pygame.draw.rect(
                self.screen, color,
                pygame.Rect(gx * cs + margin, gy * cs + margin,
                            cs - 2 * margin, cs - 2 * margin), 0
            )

        # Draw agent circles (filled, with agent-id number if cell large enough)
        font = self._get_font()
        for agent_id in range(env.nr_agents):
            px = env.current_positions[agent_id, 0].item()
            py = env.current_positions[agent_id, 1].item()
            color = self.agent_color(agent_id)
            radius = max(2, cs // 2 - 1)
            cx = px * cs + cs // 2
            cy = py * cs + cs // 2
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            if cs >= 14:
                label = font.render(str(agent_id), True, BLACK)
                lr = label.get_rect(center=(cx, cy))
                self.screen.blit(label, lr)

        # HUD: step and completion rate
        hud = font.render(
            f"Step {step:3d}  |  CR {completion_rate:.0%}  |  R=reset  ESC=quit",
            True, (220, 220, 80)
        )
        self.screen.blit(hud, (4, 4))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return self._check_keys()

    def _check_keys(self):
        key_state = pygame.key.get_pressed()
        reset = key_state[pygame.K_r]
        quit_ = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or key_state[pygame.K_ESCAPE]:
                quit_ = True
        return quit_, reset


# -----------------------------------------------------------------------
# Run one episode with rendering
# -----------------------------------------------------------------------

def run_visual_episode(env, controller, viewer, time_limit, greedy=True):
    obs = env.reset()
    done = False
    step = 0
    info = {ENV_COMPLETION_RATE: 0.0}

    while not done and step < time_limit:
        quit_, reset = viewer.draw_state(env, step, info.get(ENV_COMPLETION_RATE, 0.0))
        if quit_:
            return "quit"
        if reset:
            return "reset"

        joint_action = controller.joint_policy(obs, greedy=greedy)
        joint_action[env.is_terminated()] = WAIT  # freeze agents already at their goal
        obs, _, _, _, info = env.step(joint_action)
        done = env.is_done_all()
        step += 1

    # Hold the final frame for a moment
    for _ in range(int(viewer.fps * 1.5)):
        quit_, reset = viewer.draw_state(env, step, info.get(ENV_COMPLETION_RATE, 0.0))
        if quit_:
            return "quit"
        if reset:
            return "reset"

    return "done"


def benchmark_headless(base_map, controller, nr_agents, crop_size, time_limit, device, n=10):
    """Run n episodes headlessly and report mean CR — sanity check before rendering."""
    from train_parcel import crop_subgrid
    cr_total = 0.0
    for _ in range(n):
        crop = crop_subgrid(base_map, crop_size)
        env = make_env(crop, nr_agents, time_limit, device)
        obs = env.reset()
        done = False
        info = {ENV_COMPLETION_RATE: 0.0}
        while not done:
            action = controller.joint_policy(obs, greedy=True)
            action[env.is_terminated()] = WAIT
            obs, _, _, _, info = env.step(action)
            done = env.is_done_all()
        cr_total += info[ENV_COMPLETION_RATE]
    return cr_total / n


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize a trained PARCEL actor")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing actor_net.pth (e.g. output/gnn2)")
    parser.add_argument("--critic_type", type=str, default="gnn",
                        choices=["attention", "gnn"])
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--nr_agents", type=int, default=16)
    parser.add_argument("--map", type=str, default="instances/random-64-64-10.map",
                        help="Map file to load (.map format)")
    parser.add_argument("--crop_size", type=int, default=64,
                        help="Sub-grid size — should match training crop size (default 64)")
    parser.add_argument("--cell_size", type=int, default=18,
                        help="Pixels per grid cell")
    parser.add_argument("--fps", type=int, default=6,
                        help="Frames per second (lower = slower)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: greedy argmax)")
    parser.add_argument("--time_limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")  # rendering is CPU-bound anyway

    # --- Load map ---
    if not os.path.exists(args.map):
        raise FileNotFoundError(
            f"Map not found: {args.map}\nRun: python generate_maps.py"
        )
    base_map = load_map(args.map)
    print(f"Map loaded: {len(base_map)}x{len(base_map[0])}")

    # --- Build env from a random crop ---
    crop = crop_subgrid(base_map, args.crop_size)
    env = make_env(crop, args.nr_agents, args.time_limit, device)
    obs_dim = env.observation_dim
    print(f"Grid: {env.rows}x{env.columns}  |  Agents: {args.nr_agents}  |  Obs: {obs_dim}")

    # --- Build controller (actor only needed at test time) ---
    params = {
        ENV_NR_AGENTS: args.nr_agents,
        ENV_NR_ACTIONS: NR_GRID_ACTIONS,
        ENV_OBSERVATION_DIM: obs_dim,
        TORCH_DEVICE: device,
        EPISODES_PER_EPOCH: 1,
        ENV_TIME_LIMIT: args.time_limit,
        ENV_GAMMA: 1,
        HIDDEN_LAYER_DIM: 64,
        GNN_EMBED_DIM: 64,
        GNN_NR_LAYERS: args.gnn_layers,
        "attention_embed_dim": 64,
        NR_ATTENTION_HEADS: 2,
        LEARNING_RATE: 0.001,
        CLIP_RATIO: 0.1,
        UPDATE_ITERATIONS: 1,
        GRAD_NORM_CLIP: 1.0,
        PARCEL_ROWS: env.rows,
        PARCEL_COLS: env.columns,
        PARCEL_OBSTACLE_MAP: env.obstacle_map,
        PARCEL_RALLOC: args.crop_size,  # no radius restriction at eval
        PARCEL_CRITIC_TYPE: args.critic_type,
        "output_dim": NR_GRID_ACTIONS,
    }

    controller = PARCELController(params)
    controller.load_model_weights(args.model_dir)
    controller.policy_network.eval()
    print(f"Model loaded from: {args.model_dir}  ({args.critic_type} critic)")
    print(f"Actor params: {sum(p.numel() for p in controller.policy_network.parameters()):,}")

    greedy = not args.stochastic
    print(f"Policy mode: {'greedy (argmax)' if greedy else 'stochastic (sample)'}")

    # --- Headless sanity check ---
    print(f"\nHeadless benchmark (10 episodes, greedy) ...", end=" ", flush=True)
    mean_cr = benchmark_headless(base_map, controller, args.nr_agents,
                                 args.crop_size, args.time_limit, device, n=10)
    print(f"Mean CR = {mean_cr:.3f}")
    if mean_cr < 0.3:
        print("  [!] Low CR — model may not have finished training, or "
              "crop_size/nr_agents mismatch vs training setup.")

    # --- Pygame ---
    viewer = PARCELViewer(
        width=env.rows,
        height=env.columns,
        cell_size=args.cell_size,
        fps=args.fps,
    )
    print("\nRunning. Press R to reset, ESC to quit.\n")

    episode = 0
    while True:
        print(f"Episode {episode + 1}  (new random crop)  ", end="", flush=True)
        # Fresh crop each episode for variety
        crop = crop_subgrid(base_map, args.crop_size)
        env = make_env(crop, args.nr_agents, args.time_limit, device)

        result = run_visual_episode(env, controller, viewer, args.time_limit, greedy=greedy)
        if result == "quit":
            break
        episode += 1
        print("done")

    viewer.close()
    print("Bye.")


if __name__ == "__main__":
    main()
