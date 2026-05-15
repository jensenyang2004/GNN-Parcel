"""
PARCEL Controller
=================
Implements Algorithm 1 of the paper:
  - Inherits episode collection and PPO actor update from A2CController
  - Replaces the critic with MaskedAttentionCritic
  - Adds spatial grouping: computes grouping mask M after each epoch
  - Stores per-episode grouping information from environment start positions

Key differences from base A2CController:
  1. Uses MaskedAttentionCritic instead of standard critics
  2. Computes spatial groups at episode start (via env.current_positions)
  3. Builds union grouping mask M across Y episodes each epoch
  4. Updates critic mask before training each epoch

Usage:
    params = {
        ...standard params...,
        "algorithm_name": "PARCEL",
        PARCEL_GROUPING_RALLOC: 2,  # will be updated by CACTUS curriculum
    }
    controller = PARCELController(params)
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from os.path import join

from cactus.controller.a2c_controller import A2CController
from cactus.controller.controller import Controller
from cactus.modules.ffn_module import FFNModule
from cactus.constants import (
    ENV_NR_AGENTS, ENV_NR_ACTIONS, ENV_OBSERVATION_DIM,
    HIDDEN_LAYER_DIM, NR_ATTENTION_HEADS, GRAD_NORM_CLIP,
    LEARNING_RATE, CLIP_RATIO, UPDATE_ITERATIONS, OUTPUT_DIM,
    ACTOR_NET_FILENAME, EPSILON, NR_GRID_ACTIONS, GRID_ACTIONS,
    ENV_2D, WAIT
)
from cactus.utils import get_param_or_default, assertEquals

from parcel.grouping import (
    compute_groups, build_grouping_mask, union_grouping_masks, group_summary
)
from parcel.attention_critic import MaskedAttentionCritic

# PARCEL-specific param keys
PARCEL_RALLOC = "parcel_ralloc"           # Current allocation radius (set by curriculum)
PARCEL_ROWS = "parcel_rows"               # Map rows (set at init from env)
PARCEL_COLS = "parcel_cols"               # Map cols (set at init from env)
PARCEL_OBSTACLE_MAP = "parcel_obstacles"  # Obstacle map tensor


class PARCELController(Controller):
    """
    PARCEL: Partitioned Attention-based Reverse Curricula for Enhanced Learning.

    Extends Controller (not A2CController directly) to have full control
    over the training loop while reusing episode memory infrastructure.

    Architecture:
      Actor:  FFNModule (same as CACTUS/A2C) — decentralized policy
      Critic: MaskedAttentionCritic — grouped self-attention

    Training flow (per epoch):
      1. Y episodes collected (see update())
      2. Per episode: record start positions → compute groups → store episode mask
      3. After Y episodes: union all episode masks → set on critic → train
    """

    def __init__(self, params) -> None:
        super().__init__(params)

        # --- Actor (same as A2CController) ---
        params[OUTPUT_DIM] = self.nr_actions
        self.policy_network = FFNModule(params).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=get_param_or_default(params, LEARNING_RATE, 0.001)
        )

        # --- PPO hyperparams ---
        self.clip_ratio = get_param_or_default(params, CLIP_RATIO, 0.1)
        self.update_iterations = get_param_or_default(params, UPDATE_ITERATIONS, 4)

        # --- PARCEL Attention Critic ---
        self.critic_network = MaskedAttentionCritic(params)

        # --- Spatial grouping state ---
        self.episode_masks = []        # Per-episode grouping masks [N, N]
        self.current_start_positions = None  # Set at episode start via notify_episode_start()

        # Map properties (needed for BFS in grouping)
        self.ralloc = get_param_or_default(params, PARCEL_RALLOC, 2)
        self.rows = get_param_or_default(params, PARCEL_ROWS, None)
        self.cols = get_param_or_default(params, PARCEL_COLS, None)
        self.obstacle_map = get_param_or_default(params, PARCEL_OBSTACLE_MAP, None)

        # Logging
        self.verbose = get_param_or_default(params, "verbose", False)

    # ------------------------------------------------------------------
    # Spatial grouping interface
    # ------------------------------------------------------------------

    def set_map_info(self, rows, cols, obstacle_map):
        """Called once after env is created to set map dimensions."""
        self.rows = rows
        self.cols = cols
        self.obstacle_map = obstacle_map

    def set_ralloc(self, ralloc):
        """Called by CACTUS curriculum when allocation radius is updated."""
        self.ralloc = ralloc

    def notify_episode_start(self, start_positions):
        """
        Call this at the START of each episode with agent start positions.
        This computes the spatial grouping for the episode.

        Args:
            start_positions: list of (row, col) tuples, length N
        """
        self.current_start_positions = start_positions

        if self.rows is None or self.obstacle_map is None:
            # Map info not set yet; default to fully connected (all same group)
            mask = torch.zeros(self.nr_agents, self.nr_agents, device=self.device)
        else:
            groups = compute_groups(
                start_positions, self.ralloc,
                self.rows, self.cols, self.obstacle_map
            )
            mask = build_grouping_mask(groups, self.nr_agents, self.device)

            if self.verbose:
                print(f"  [PARCEL] Ralloc={self.ralloc} | {group_summary(groups, self.nr_agents)}")

        self.episode_masks.append(mask)

    def _build_epoch_mask(self):
        """
        After Y episodes: aggregate all per-episode masks via union.
        If no episode masks collected, fall back to fully connected.
        """
        if not self.episode_masks:
            return torch.zeros(self.nr_agents, self.nr_agents, device=self.device)
        return union_grouping_masks(self.episode_masks)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def calculate_action_masks(self, joint_observation):
        """Compute invalid action masks from observation (obstacles in local view)."""
        channels = self.observation_dim[0]
        o_size = self.observation_dim[1]
        batch_size = joint_observation.size(0)
        joint_observation = joint_observation.view(batch_size, self.nr_agents, channels, o_size, o_size)
        half_size = o_size // 2
        agent_ids = list(range(self.nr_agents))

        invalid_actions = torch.zeros(
            batch_size, self.nr_agents, self.nr_actions,
            dtype=torch.bool, device=self.device
        )
        grid_ops = self.grid_operations[GRID_ACTIONS]
        for action_idx, delta in enumerate(grid_ops):
            dx, dy = delta[0].item(), delta[1].item()
            # Channel 2 = obstacle channel (same as A2CController)
            invalid_actions[:, agent_ids, action_idx] = \
                joint_observation[:, agent_ids, 2, half_size + dx, half_size + dy].bool()

        action_mask = torch.zeros_like(invalid_actions, dtype=torch.float32)
        action_mask[invalid_actions] = float('-inf')
        return action_mask.view(-1, self.nr_actions)

    def joint_policy(self, joint_observation):
        """Sample actions from the current policy (used during episode rollout)."""
        joint_observation = joint_observation.view(1, self.nr_agents, -1)
        action_mask = self.calculate_action_masks(joint_observation)

        obs_flat = joint_observation.view(-1, self.policy_network.input_shape)
        action_logits = self.policy_network(obs_flat)  # [N, nr_actions]

        assertEquals(action_mask.size(), action_logits.size())
        probs = F.softmax(action_logits + action_mask, dim=-1).detach()
        m = Categorical(probs)
        return m.sample().view(self.nr_agents)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """
        Called after every Y episodes (by Controller.update).

        Steps:
          1. Build epoch-level grouping mask M (union of episode masks)
          2. Update critic's mask
          3. Train critic to minimize MSE loss (Eq. 5)
          4. Train actor with PPO using critic baseline (Eq. 2)
          5. Clear episode masks for next epoch
        """
        # --- Step 1: Build and set grouping mask ---
        epoch_mask = self._build_epoch_mask()
        self.critic_network.set_grouping_mask(epoch_mask)

        if self.verbose:
            connected = (epoch_mask == 0.0).sum().item()
            total = self.nr_agents ** 2
            print(f"[PARCEL] Epoch mask: {connected}/{total} agent pairs coordinated")

        # --- Step 2: Get training data from memory ---
        obs, actions, returns, _, _ = self.memory.get_training_data(truncated=True)
        # obs:     [batch*N, obs_size]
        # actions: [batch*N]
        # returns: [batch*N] (discounted Monte Carlo returns)

        action_mask = self.calculate_action_masks(obs)
        obs_flat = obs.view(-1, self.policy_network.input_shape)
        actions_flat = actions.view(-1)
        returns_flat = returns.view(-1)

        # Normalize returns for stable training
        returns_normalized = (returns_flat - returns_flat.mean()) / (returns_flat.std() + EPSILON)

        # --- Step 3: Initial policy probs (for PPO ratio) ---
        with torch.no_grad():
            logits = self.policy_network(obs_flat)
            old_probs = F.softmax(logits + action_mask, dim=-1).detach()

        # --- Step 4: Multiple PPO update iterations ---
        for iteration in range(self.update_iterations):
            # Train critic
            self.critic_network.train(obs_flat, actions_flat, returns_normalized)

            # Compute advantage = return - critic baseline
            logits = self.policy_network(obs_flat)
            probs = F.softmax(logits + action_mask, dim=-1)

            baseline = self.critic_network.counterfactual_baseline(
                obs_flat, actions_flat, probs
            )
            advantages = (returns_normalized - baseline).detach()

            # PPO clipped policy loss
            policy_loss = self._ppo_loss(advantages, probs, actions_flat, old_probs)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                get_param_or_default({"grad_norm_clip": 1.0}, GRAD_NORM_CLIP, 1.0)
            )
            self.policy_optimizer.step()

        # --- Step 5: Clear episode masks for next epoch ---
        self.episode_masks.clear()

    def _ppo_loss(self, advantages, probs, actions, old_probs):
        """
        PPO clipped surrogate loss (Schulman et al. 2017).

        L^CLIP = -E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
        where r_t = pi(a|s) / pi_old(a|s)
        """
        m = Categorical(probs)
        m_old = Categorical(old_probs)

        log_prob = m.log_prob(actions)
        log_prob_old = m_old.log_prob(actions).detach()

        ratio = torch.exp(log_prob - log_prob_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        return loss.mean()

    # ------------------------------------------------------------------
    # Model saving / loading
    # ------------------------------------------------------------------

    def get_parameter_count(self):
        actor_params = sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)
        critic_params = self.critic_network.get_parameter_count()
        return actor_params + critic_params

    def save_model_weights(self, path):
        torch.save(self.policy_network.state_dict(), join(path, ACTOR_NET_FILENAME))
        self.critic_network.save_model_weights(path)

    def load_model_weights(self, path):
        self.policy_network.load_state_dict(
            torch.load(join(path, ACTOR_NET_FILENAME), map_location=self.device)
        )
        self.policy_network.eval()
        self.critic_network.load_model_weights(path)