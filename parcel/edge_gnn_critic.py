"""
PARCEL Edge-Encoded GCN Critic
================================
Extends the plain GCN critic by embedding each edge's scalar weight into a
vector and concatenating it to the neighbor's node feature before the linear
transform (Option B).

This lets the network learn qualitatively different messages depending on how
close two agents' start positions are — not just scale the same message up/down.

Architecture per GCN layer:
    e_ij (scalar ∈ [0,1])
        → Linear(1, edge_dim) → ELU
        = edge_embed_ij  [edge_dim]

    msg_ij = concat(h_j [embed_dim],  edge_embed_ij [edge_dim])
                     ↑ what j sees     ↑ how close j is to i

    h'_i = ELU( sum_{j: w[i,j]>0} w_norm[i,j] * (W_msg @ msg_ij) )

    where w_norm = degree-normalized soft edge weights (from build_edge_weights)

Parameter cost vs plain GCN (embed=64, edge_dim=8):
    Plain GCN 1-layer:  ~25K params
    Edge GCN 1-layer:   ~26K params  (+~1K for edge embedding + wider W_msg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from cactus.constants import (
    ENV_OBSERVATION_DIM, ENV_NR_AGENTS, ENV_NR_ACTIONS, HIDDEN_LAYER_DIM
)
from cactus.utils import assertContains, get_param_or_default
from parcel.gnn_critic import _make_node_encoder  # reuse identical encoder
from parcel.grouping import build_edge_weights_batched

EDGE_GNN_NR_LAYERS = "edge_gnn_nr_layers"
EDGE_GNN_EMBED_DIM = "edge_gnn_embed_dim"
EDGE_GNN_EDGE_DIM  = "edge_gnn_edge_dim"
EDGE_GNN_CHUNK_STEPS = "edge_gnn_chunk_steps"


def _normalize_edge_weights(W):
    """
    Degree-normalize a soft edge weight matrix.
    D^{-1/2} W D^{-1/2}, with D_i = sum_j W[i,j].
    W can be [N, N] or [T, N, N]; normalization is applied along the last two dims.
    """
    D = W.sum(dim=-1)
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt = D_inv_sqrt.masked_fill(D_inv_sqrt.isinf(), 0.0)
    # unsqueeze(-1) → row scale, unsqueeze(-2) → col scale; works for both 2D and 3D
    return D_inv_sqrt.unsqueeze(-1) * W * D_inv_sqrt.unsqueeze(-2)


class EdgeGCNLayer(nn.Module):
    """
    One edge-encoded GCN layer.

    For each agent i:
        h'_i = ELU( sum_j w_norm[i,j] * W_msg @ concat(h_j, embed(e_ij)) )

    W_msg: Linear(embed_dim + edge_dim, embed_dim)  — no bias on the main path,
    bias handled by the output ELU's affine.
    """

    def __init__(self, embed_dim, edge_dim):
        super().__init__()
        self.edge_embed = nn.Sequential(
            nn.Linear(3, edge_dim),   # input: (dr, dc, w) — signed direction + proximity
            nn.ELU(),
        )
        # Combines neighbor node feature + edge embedding → embed_dim output
        self.msg_linear = nn.Linear(embed_dim + edge_dim, embed_dim)

    def forward(self, H, W_norm, edge_weights):
        """
        Args:
            H:            FloatTensor [T, N, embed_dim]
            W_norm:       FloatTensor [T, N, N]  — degree-normalized soft weights
            edge_weights: FloatTensor [T, N, N, 3]  — (dr, dc, w) per timestep

        Returns:
            FloatTensor [T, N, embed_dim]
        """
        T, N, embed_dim = H.shape

        W_h = self.msg_linear.weight[:, :embed_dim]   # [embed_dim, embed_dim]
        W_e = self.msg_linear.weight[:, embed_dim:]   # [embed_dim, edge_dim]

        # Node part: O(T × N × embed)
        AH = torch.bmm(W_norm, H)       # [T, N, embed]
        node_out = AH @ W_h.t()         # [T, N, embed]

        # Edge part: embed (dr, dc, w), aggregate over j, then project.
        edge_embeds = self.edge_embed(edge_weights)                               # [T, N, N, edge_dim]
        edge_agg = (W_norm.unsqueeze(-1) * edge_embeds).sum(dim=-2)               # [T, N, edge_dim]
        edge_out = edge_agg @ W_e.t()                                             # [T, N, embed]

        if self.msg_linear.bias is not None:
            bias_out = W_norm.sum(dim=-1, keepdim=True) * self.msg_linear.bias   # [T, N, embed]
            return F.elu(node_out + edge_out + bias_out)
        return F.elu(node_out + edge_out)


class EdgeGNNCriticModule(nn.Module):
    """
    Full edge-encoded GCN critic network.

    Pipeline:
        obs_i → node encoder → h_i [embed_dim]
        H = stack(h_i)  [N, embed_dim]
        for each layer: H = EdgeGCNLayer(H, W_norm, edge_weights)
        Q_i = Linear(h_i, nr_actions)
    """

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, HIDDEN_LAYER_DIM)
        super().__init__()

        self.input_shape = int(numpy.prod(params[ENV_OBSERVATION_DIM]))
        self.nr_agents   = params[ENV_NR_AGENTS]
        self.nr_actions  = params[ENV_NR_ACTIONS]
        hidden_dim = params[HIDDEN_LAYER_DIM]
        embed_dim  = get_param_or_default(params, EDGE_GNN_EMBED_DIM, 64)
        edge_dim   = get_param_or_default(params, EDGE_GNN_EDGE_DIM, 8)
        nr_layers  = get_param_or_default(params, EDGE_GNN_NR_LAYERS, 1)

        self.node_encoder = _make_node_encoder(self.input_shape, hidden_dim, embed_dim)

        self.gcn_layers = nn.ModuleList([
            EdgeGCNLayer(embed_dim, edge_dim) for _ in range(nr_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, self.nr_actions)

    def forward(self, joint_obs, W_norm, edge_weights):
        """
        Args:
            joint_obs:    FloatTensor [T*N, input_shape]
            W_norm:       FloatTensor [T, N, N]  — normalized adjacency per timestep
            edge_weights: FloatTensor [T, N, N, 3]  — (dr, dc, w) per timestep

        Returns:
            Q values: FloatTensor [T*N, nr_actions]
        """
        N = self.nr_agents
        T = joint_obs.shape[0] // N

        H = self.node_encoder(joint_obs.view(T * N, self.input_shape))
        H = H.view(T, N, -1)

        for layer in self.gcn_layers:
            H = layer(H, W_norm, edge_weights)

        return self.output_layer(H).view(T * N, self.nr_actions)


class EdgeGNNCritic:
    """
    Edge-encoded GCN critic. Same interface as GNNCritic and MaskedAttentionCritic.

    The realtime training path stores per-step positions for the epoch and builds
    edge weights in timestep chunks. set_edge_weights() remains available for
    callers that still pass precomputed edge weights.
    """

    def __init__(self, params):
        self.device      = params["device"]
        self.nr_agents   = params[ENV_NR_AGENTS]
        self.nr_actions  = params[ENV_NR_ACTIONS]
        self.grad_norm_clip = get_param_or_default(params, "grad_norm_clip", 1.0)
        self.learning_rate  = get_param_or_default(params, "learning_rate", 0.001)
        self.chunk_steps = max(1, int(get_param_or_default(params, EDGE_GNN_CHUNK_STEPS, 16)))

        self.q_net = EdgeGNNCriticModule(params).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )

        self._edge_weights = None      # [T, N, N, 3]
        self._w_norm       = None      # [T, N, N] degree-normalized
        self._positions_batch = None   # [T, N, 2]
        self._ralloc = None

    def set_grouping_mask(self, mask):
        """Called by controller — grouping is encoded in edge_weights, mask unused here."""
        pass

    def set_edge_weights(self, edge_weights):
        """
        Called by PARCELController before each train() call.
        edge_weights: FloatTensor [T, N, N, 3] from build_edge_weights_batched().
        W_norm is computed from the proximity channel (index 2).
        """
        self._edge_weights = edge_weights.to(self.device)
        self._w_norm = _normalize_edge_weights(self._edge_weights[..., 2])
        self._positions_batch = None
        self._ralloc = None

    def set_positions(self, positions_batch, ralloc):
        """
        Store per-step agent positions so realtime edge features can be built
        in memory-safe timestep chunks during critic forward/training.
        """
        self._positions_batch = positions_batch.detach()
        self._ralloc = ralloc
        self._edge_weights = None
        self._w_norm = None

    def _edge_tensors_for_slice(self, start_step, end_step):
        if self._positions_batch is not None:
            pos = self._positions_batch[start_step:end_step].to(self.device)
            edge_weights = build_edge_weights_batched(pos, self._ralloc, self.device)
            w_norm = _normalize_edge_weights(edge_weights[..., 2])
            return w_norm, edge_weights

        if self._edge_weights is None or self._w_norm is None:
            raise RuntimeError("EdgeGNNCritic requires set_positions() or set_edge_weights() before forward.")

        return (
            self._w_norm[start_step:end_step],
            self._edge_weights[start_step:end_step],
        )

    def _forward_chunked(self, observations):
        total_samples = observations.shape[0]
        if total_samples % self.nr_agents != 0:
            raise ValueError(
                f"Expected observations length to be divisible by nr_agents={self.nr_agents}, "
                f"got {total_samples}."
            )

        total_steps = total_samples // self.nr_agents
        outputs = []
        for start_step in range(0, total_steps, self.chunk_steps):
            end_step = min(start_step + self.chunk_steps, total_steps)
            start_sample = start_step * self.nr_agents
            end_sample = end_step * self.nr_agents
            obs_chunk = observations[start_sample:end_sample]
            w_norm, edge_weights = self._edge_tensors_for_slice(start_step, end_step)
            outputs.append(self.q_net(obs_chunk, w_norm, edge_weights))
        return torch.cat(outputs, dim=0)

    def get_parameter_count(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def save_model_weights(self, path):
        from os.path import join
        torch.save(self.q_net.state_dict(), join(path, "critic_net.pth"))

    def load_model_weights(self, path):
        from os.path import join
        self.q_net.load_state_dict(
            torch.load(join(path, "critic_net.pth"), map_location=self.device,
                       weights_only=True)
        )
        self.q_net.eval()

    def counterfactual_baseline(self, observations, actions, probs):
        with torch.no_grad():
            q_values = self._forward_chunked(observations)
            baseline = (probs * q_values).sum(dim=-1).detach()
        return baseline

    def train(self, observations, actions, targets):
        self.optimizer.zero_grad()

        total_samples = observations.shape[0]
        if total_samples % self.nr_agents != 0:
            raise ValueError(
                f"Expected observations length to be divisible by nr_agents={self.nr_agents}, "
                f"got {total_samples}."
            )

        total_steps = total_samples // self.nr_agents
        total_loss = 0.0
        total_count = 0

        for start_step in range(0, total_steps, self.chunk_steps):
            end_step = min(start_step + self.chunk_steps, total_steps)
            start_sample = start_step * self.nr_agents
            end_sample = end_step * self.nr_agents
            chunk_sample_count = end_sample - start_sample

            obs_chunk = observations[start_sample:end_sample]
            actions_chunk = actions[start_sample:end_sample]
            targets_chunk = targets[start_sample:end_sample]
            w_norm, edge_weights = self._edge_tensors_for_slice(start_step, end_step)
            q_values = self.q_net(obs_chunk, w_norm, edge_weights)

            q_taken = q_values.gather(1, actions_chunk.view(-1, 1)).squeeze(1)
            loss = F.mse_loss(q_taken, targets_chunk.view(-1), reduction="mean")
            weight = chunk_sample_count / total_samples
            (loss * weight).backward()

            total_loss += loss.item() * chunk_sample_count
            total_count += chunk_sample_count

        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        return total_loss / total_count
