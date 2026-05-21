"""
PARCEL GCN Critic
=================
Drop-in replacement for MaskedAttentionCritic using a Graph Convolutional
Network (GCN) instead of masked multi-head self-attention.

The grouping mask maps directly to a graph:
  mask[i, j] = 0.0   → edge (i, j) exists  (same spatial group)
  mask[i, j] = -inf  → no edge              (different groups)

GCN layer: H' = ELU(A_norm @ H @ W)
  A_norm = D^{-1/2} (A + I) D^{-1/2}  (symmetric normalized, with self-loops)

Architecture:
  obs_i → node encoder (Linear→ELU→Linear→ELU) → h_i [embed_dim]
  H = stack(h_i)  [N, embed_dim]
  for each layer: H = ELU(A_norm @ H @ W)
  Q_i = Linear(h_i, nr_actions)

Parameter count vs attention:
  Attention (2 heads, embed=64): ~120K
  GCN 1-layer (embed=64):         ~25K   (~5x fewer)
  GCN 2-layer (embed=64):         ~29K   (~4x fewer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from cactus.constants import (
    ENV_OBSERVATION_DIM, ENV_NR_AGENTS, ENV_NR_ACTIONS,
    HIDDEN_LAYER_DIM, FLOAT_TYPE
)
from cactus.utils import assertContains, get_param_or_default

GNN_NR_LAYERS = "gnn_nr_layers"
GNN_EMBED_DIM = "gnn_embed_dim"


def mask_to_adj_norm(mask, device):
    """
    Convert grouping mask to symmetric-normalized adjacency with self-loops.

    Args:
        mask: FloatTensor [N, N] — 0.0 = same group (edge), -inf = different group
        device: torch device

    Returns:
        A_norm: FloatTensor [N, N] — D^{-1/2} (A + I) D^{-1/2}
    """
    N = mask.size(0)
    A = (mask == 0.0).float()           # binary adjacency [N, N]
    A = A + torch.eye(N, device=device) # self-loops; may make diagonal 2 if already 1
    A = (A > 0).float()                 # clamp back to binary
    D = A.sum(dim=1)                    # degree vector [N]
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt.isinf()] = 0.0  # isolated node with no self-loop (shouldn't happen)
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    A_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    return A_norm  # [N, N]


class GCNLayer(nn.Module):
    """Single GCN layer: H' = ELU(A_norm @ H @ W)"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, H, A_norm):
        """
        Args:
            H:      FloatTensor [batch, N, in_dim]
            A_norm: FloatTensor [N, N]

        Returns:
            FloatTensor [batch, N, out_dim]
        """
        # Aggregate neighbor features: [batch, N, in_dim]
        AH = torch.bmm(A_norm.unsqueeze(0).expand(H.size(0), -1, -1), H)
        return F.elu(self.linear(AH))


def _make_node_encoder(input_dim, hidden_dim, embed_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, embed_dim),
        nn.ELU(),
    )


class GNNCriticModule(nn.Module):
    """
    GCN critic network.

    Pipeline:
      [N agents] × obs → node encoder → H [N, embed]
      → nr_layers × GCNLayer(A_norm) → H_out [N, embed]
      → output linear → Q values [N, nr_actions]
    """

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, HIDDEN_LAYER_DIM)
        super().__init__()

        self.input_shape = int(numpy.prod(params[ENV_OBSERVATION_DIM]))
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        hidden_dim = params[HIDDEN_LAYER_DIM]
        embed_dim = get_param_or_default(params, GNN_EMBED_DIM, 64)
        nr_layers = get_param_or_default(params, GNN_NR_LAYERS, 1)

        self.node_encoder = _make_node_encoder(self.input_shape, hidden_dim, embed_dim)

        self.gcn_layers = nn.ModuleList([
            GCNLayer(embed_dim, embed_dim) for _ in range(nr_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, self.nr_actions)

    def forward(self, joint_obs, A_norm):
        """
        Args:
            joint_obs: FloatTensor [batch*N, input_shape] or [batch, N, input_shape]
            A_norm:    FloatTensor [N, N] — normalized adjacency

        Returns:
            Q values: FloatTensor [batch*N, nr_actions]
        """
        N = self.nr_agents
        batch = joint_obs.numel() // (N * self.input_shape)
        obs_flat = joint_obs.view(batch * N, self.input_shape)

        # Node encoding: [batch*N, embed_dim] → [batch, N, embed_dim]
        H = self.node_encoder(obs_flat).view(batch, N, -1)

        # GCN message passing
        for layer in self.gcn_layers:
            H = layer(H, A_norm)

        # Output Q values: [batch, N, nr_actions] → [batch*N, nr_actions]
        q_values = self.output_layer(H)
        return q_values.view(batch * N, self.nr_actions)


class GNNCritic:
    """
    GCN-based critic with the same interface as MaskedAttentionCritic.

    Replaces the NxN attention with GCN message passing over the spatial
    grouping graph. The grouping mask is converted to a normalized adjacency
    matrix before each forward pass.
    """

    def __init__(self, params):
        self.device = params["device"]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.grad_norm_clip = get_param_or_default(params, "grad_norm_clip", 1.0)
        self.learning_rate = get_param_or_default(params, "learning_rate", 0.001)

        self.q_net = GNNCriticModule(params).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )
        self.grouping_mask = None
        self._adj_norm_cache = None  # cached adjacency to avoid recomputing each call

    def set_grouping_mask(self, mask):
        """Called by PARCELController after each epoch with the new mask M."""
        self.grouping_mask = mask.to(self.device)
        # Recompute normalized adjacency when mask changes
        self._adj_norm_cache = mask_to_adj_norm(self.grouping_mask, self.device)

    def _get_adj_norm(self):
        """Return cached adjacency, or fully connected if no mask set yet."""
        if self._adj_norm_cache is None:
            N = self.nr_agents
            # Default: all agents connected (fully connected graph, normalized)
            mask = torch.zeros(N, N, device=self.device)
            self._adj_norm_cache = mask_to_adj_norm(mask, self.device)
        return self._adj_norm_cache

    def get_parameter_count(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def save_model_weights(self, path):
        from os.path import join
        torch.save(self.q_net.state_dict(), join(path, "critic_net.pth"))

    def load_model_weights(self, path):
        from os.path import join
        self.q_net.load_state_dict(
            torch.load(join(path, "critic_net.pth"), map_location=self.device, weights_only=True)
        )
        self.q_net.eval()

    def counterfactual_baseline(self, observations, actions, probs):
        """
        Expected Q value under current policy: sum_a pi(a|tau) * Q(tau, a).

        Args:
            observations: [batch*N, obs_size]
            actions:      [batch*N]
            probs:        [batch*N, nr_actions]

        Returns:
            baseline: [batch*N]
        """
        A_norm = self._get_adj_norm()
        q_values = self.q_net(observations, A_norm)  # [batch*N, nr_actions]
        baseline = (probs * q_values).sum(dim=-1)
        return baseline.detach()

    def train(self, observations, actions, targets):
        """
        Train critic to minimize MSE loss.

        Args:
            observations: [batch*N, obs_size]
            actions:      [batch*N]
            targets:      [batch*N]
        """
        A_norm = self._get_adj_norm()
        q_values = self.q_net(observations, A_norm)  # [batch*N, nr_actions]

        actions_idx = actions.view(-1, 1)
        q_taken = q_values.gather(1, actions_idx).squeeze(1)  # [batch*N]

        targets_flat = targets.view(-1)
        loss = F.mse_loss(q_taken, targets_flat)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item()
