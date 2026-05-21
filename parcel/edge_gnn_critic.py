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

EDGE_GNN_NR_LAYERS = "edge_gnn_nr_layers"
EDGE_GNN_EMBED_DIM = "edge_gnn_embed_dim"
EDGE_GNN_EDGE_DIM  = "edge_gnn_edge_dim"


def _normalize_edge_weights(W, device):
    """
    Degree-normalize a soft edge weight matrix (already includes self-loops).
    D^{-1/2} W D^{-1/2}, with D_ii = sum_j W[i,j].
    Agents with zero degree (isolated, no self-loop) get 0 rows/cols.
    """
    D = W.sum(dim=1)                      # [N]
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt.isinf()] = 0.0
    W_norm = D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
    return W_norm


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
            H:            FloatTensor [batch, N, embed_dim]
            W_norm:       FloatTensor [N, N]  — degree-normalized soft weights
            edge_weights: FloatTensor [N, N]  — raw soft weights (for embedding)

        Returns:
            FloatTensor [batch, N, embed_dim]

        Efficient factored implementation avoids [batch, N, N, embed_dim] tensors.

        Original: aggregated[i] = sum_j w[i,j] * W_msg @ concat(h_j, e_ij)
        Factored into three batch-independent-or-small terms:
          node part:  (W_norm @ H) @ W_h.T          [batch, N, embed_dim]
          edge part:  (W_norm ⊙ E_transformed).sum(j)  [N, embed_dim], broadcast
          bias part:  row_sums[i] * bias              [N, embed_dim], broadcast
        """
        batch, N, embed_dim = H.shape

        # Split msg_linear weights: [out=embed_dim, in=embed_dim+edge_dim]
        W_h = self.msg_linear.weight[:, :embed_dim]   # [embed_dim, embed_dim]
        W_e = self.msg_linear.weight[:, embed_dim:]   # [embed_dim, edge_dim]

        # Node part: aggregate neighbors then transform — O(batch × N × embed)
        AH = torch.bmm(W_norm.unsqueeze(0).expand(batch, -1, -1), H)  # [batch, N, embed]
        node_out = AH @ W_h.t()                                         # [batch, N, embed]

        # Edge part: embed (dr, dc, w) features then aggregate — O(N² × edge_dim), no batch axis
        edge_embeds = self.edge_embed(edge_weights)                     # [N, N, 3] → [N, N, edge_dim]
        edge_out = (W_norm.unsqueeze(-1) * (edge_embeds @ W_e.t())).sum(dim=1)  # [N, embed]

        # Bias part: each (i,j) pair contributes w[i,j]*bias, so sum = row_sum * bias
        if self.msg_linear.bias is not None:
            bias_out = W_norm.sum(dim=1, keepdim=True) * self.msg_linear.bias  # [N, embed]
            return F.elu(node_out + edge_out.unsqueeze(0) + bias_out.unsqueeze(0))
        return F.elu(node_out + edge_out.unsqueeze(0))


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
            joint_obs:    FloatTensor [batch*N, input_shape]
            W_norm:       FloatTensor [N, N]  — normalized adjacency
            edge_weights: FloatTensor [N, N]  — raw soft weights

        Returns:
            Q values: FloatTensor [batch*N, nr_actions]
        """
        N = self.nr_agents
        batch = joint_obs.numel() // (N * self.input_shape)

        H = self.node_encoder(joint_obs.view(batch * N, self.input_shape))
        H = H.view(batch, N, -1)

        for layer in self.gcn_layers:
            H = layer(H, W_norm, edge_weights)

        return self.output_layer(H).view(batch * N, self.nr_actions)


class EdgeGNNCritic:
    """
    Edge-encoded GCN critic. Same interface as GNNCritic and MaskedAttentionCritic.

    Requires two tensors per epoch:
        grouping_mask   [N, N]     — 0 / -inf  (which pairs are in same group)
        edge_weights    [N, N, 3]  — (dr, dc, w) features from build_edge_weights()
    """

    def __init__(self, params):
        self.device      = params["device"]
        self.nr_agents   = params[ENV_NR_AGENTS]
        self.nr_actions  = params[ENV_NR_ACTIONS]
        self.grad_norm_clip = get_param_or_default(params, "grad_norm_clip", 1.0)
        self.learning_rate  = get_param_or_default(params, "learning_rate", 0.001)

        self.q_net = EdgeGNNCriticModule(params).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )

        self._edge_weights_raw  = None   # [N, N] soft weights
        self._w_norm_cache      = None   # [N, N] degree-normalized

    def set_grouping_mask(self, mask):
        """Called by controller — mask not directly used here (edge_weights carry the info)."""
        pass

    def set_edge_weights(self, edge_weights):
        """
        Called by PARCELController after each epoch alongside set_grouping_mask.
        edge_weights: FloatTensor [N, N, 3] from build_edge_weights() — (dr, dc, w).
        W_norm is degree-normalized from the proximity channel (w = index 2).
        """
        self._edge_weights_raw = edge_weights.to(self.device)
        self._w_norm_cache = _normalize_edge_weights(
            self._edge_weights_raw[:, :, 2], self.device
        )

    def _get_weights(self):
        if self._edge_weights_raw is None:
            # Default: fully connected, no direction, uniform proximity
            N = self.nr_agents
            W_feats = torch.zeros(N, N, 3, device=self.device)
            W_feats[:, :, 2] = 1.0  # proximity channel = 1
            self._edge_weights_raw = W_feats
            self._w_norm_cache = _normalize_edge_weights(W_feats[:, :, 2], self.device)
        return self._w_norm_cache, self._edge_weights_raw

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
        W_norm, E = self._get_weights()
        q_values = self.q_net(observations, W_norm, E)
        return (probs * q_values).sum(dim=-1).detach()

    def train(self, observations, actions, targets):
        W_norm, E = self._get_weights()
        q_values = self.q_net(observations, W_norm, E)

        q_taken = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        loss = F.mse_loss(q_taken, targets.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        return loss.item()
