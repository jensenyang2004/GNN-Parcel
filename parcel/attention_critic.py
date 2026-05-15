"""
PARCEL Masked Self-Attention Critic
=====================================
Implements Section 4.2 of the paper (Eq. 3 and 4):

    att(Wq, Wk, Wv, M) = softmax( (Wq @ Wk^T + M) / sqrt(z) ) @ Wv

The grouping mask M ensures that agents of different spatial groups
cannot attend to each other (M[i,j] = -inf → softmax weight → 0).

Architecture:
  - Three MLP encoders: q, k, v (each: Linear → ELU → Linear → ELU → Linear)
  - Multi-head attention with grouping mask
  - Output layer: attention embeddings → Q values per action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from cactus.constants import (
    ENV_OBSERVATION_DIM, ENV_NR_AGENTS, ENV_NR_ACTIONS,
    HIDDEN_LAYER_DIM, NR_ATTENTION_HEADS, FLOAT_TYPE
)
from cactus.utils import assertContains, get_param_or_default

# Constant key for embedding dimension
ATTENTION_EMBED_DIM = "attention_embed_dim"


def _make_encoder(input_dim, hidden_dim, embed_dim):
    """Small MLP encoder for Q, K, V projections."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, embed_dim),
    )


class MaskedAttentionHead(nn.Module):
    """
    A single masked self-attention head.

    For each agent i in group Cx:
        agent_att_i = sum_{j in Cx} P_{i,j} * Wv_j

    where P_{i,j} = softmax( (Wq_i @ Wk_j^T + M_{i,j}) / sqrt(z) )
    """

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_encoder = _make_encoder(input_dim, hidden_dim, embed_dim)
        self.k_encoder = _make_encoder(input_dim, hidden_dim, embed_dim)
        self.v_encoder = _make_encoder(input_dim, hidden_dim, embed_dim)

    def forward(self, obs_flat, mask):
        """
        Args:
            obs_flat: FloatTensor [batch, N, input_dim]
            mask:     FloatTensor [N, N] — grouping mask (0 or -inf)

        Returns:
            FloatTensor [batch, N, embed_dim] — attended value vectors
        """
        batch, N, _ = obs_flat.shape

        # Project to Q, K, V embeddings: [batch, N, embed_dim]
        Wq = self.q_encoder(obs_flat)
        Wk = self.k_encoder(obs_flat)
        Wv = self.v_encoder(obs_flat)

        # Attention logits: [batch, N, N]
        scale = numpy.sqrt(self.embed_dim)
        logits = torch.bmm(Wq, Wk.transpose(1, 2)) / scale  # [batch, N, N]

        # Add grouping mask (broadcast over batch dimension)
        # mask: [N, N] → [1, N, N]
        logits = logits + mask.unsqueeze(0)

        # Softmax over key dimension
        attention_weights = F.softmax(logits, dim=-1)  # [batch, N, N]

        # Weighted sum of values: [batch, N, embed_dim]
        attended = torch.bmm(attention_weights, Wv)
        return attended


class MaskedMultiHeadAttention(nn.Module):
    """
    Multi-head version: multiple heads summed, then projected to output.
    Matches paper: "multiple attention heads are used for Eq. 3,
                    which are summed and processed to critic values Q̂_i"
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, nr_heads, nr_actions):
        super().__init__()
        self.nr_heads = nr_heads
        self.embed_dim = embed_dim

        self.heads = nn.ModuleList([
            MaskedAttentionHead(input_dim, hidden_dim, embed_dim)
            for _ in range(nr_heads)
        ])
        # Final linear: summed attention → Q values per action
        self.output_layer = nn.Linear(embed_dim, nr_actions)

    def forward(self, obs_flat, mask):
        """
        Args:
            obs_flat: [batch, N, input_dim]
            mask:     [N, N]

        Returns:
            Q values: [batch, N, nr_actions]
        """
        # Sum attention outputs from all heads: [batch, N, embed_dim]
        attended = sum(head(obs_flat, mask) for head in self.heads)

        # Project to action space: [batch, N, nr_actions]
        q_values = self.output_layer(attended)
        return q_values


class MaskedAttentionCriticModule(nn.Module):
    """
    Full PARCEL critic module.

    Pipeline:
      observation → [multi-head masked attention] → Q values per agent
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
        embed_dim = get_param_or_default(params, ATTENTION_EMBED_DIM, 64)
        nr_heads = get_param_or_default(params, NR_ATTENTION_HEADS, 2)

        self.attention = MaskedMultiHeadAttention(
            input_dim=self.input_shape,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            nr_heads=nr_heads,
            nr_actions=self.nr_actions,
        )

    def forward(self, joint_obs, mask):
        """
        Args:
            joint_obs: FloatTensor of shape [batch, N, input_shape]
                       or [batch * N, input_shape] (will be reshaped)
            mask: FloatTensor [N, N]

        Returns:
            Q values: FloatTensor [batch * N, nr_actions]
        """
        N = self.nr_agents
        # Reshape to [batch, N, input_shape]
        batch = joint_obs.numel() // (N * self.input_shape)
        obs_flat = joint_obs.view(batch, N, self.input_shape)

        # Masked multi-head attention → [batch, N, nr_actions]
        q_values = self.attention(obs_flat, mask)

        # Flatten to [batch * N, nr_actions] to match expected critic interface
        return q_values.view(batch * N, self.nr_actions)


class MaskedAttentionCritic:
    """
    Wrapper for MaskedAttentionCriticModule, matching the interface of
    existing critics (QCritic, QMIXCritic, etc.) in critic.py.

    Used by PARCELController to train and query Q values.
    """

    def __init__(self, params):
        self.device = params["device"]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.grad_norm_clip = get_param_or_default(params, "grad_norm_clip", 1.0)
        self.learning_rate = get_param_or_default(params, "learning_rate", 0.001)

        self.q_net = MaskedAttentionCriticModule(params).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )
        # Current grouping mask — updated each epoch by PARCELController
        self.grouping_mask = None

    def set_grouping_mask(self, mask):
        """Called by PARCELController after each epoch with the new mask M."""
        self.grouping_mask = mask.to(self.device)

    def _get_mask(self, batch_size):
        """Returns current mask, or fully connected mask if none set yet."""
        N = self.nr_agents
        if self.grouping_mask is None:
            # Default: all agents in one group (fully connected)
            return torch.zeros(N, N, device=self.device)
        return self.grouping_mask

    def get_parameter_count(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def save_model_weights(self, path):
        from os.path import join
        torch.save(self.q_net.state_dict(), join(path, "critic_net.pth"))

    def load_model_weights(self, path):
        from os.path import join
        self.q_net.load_state_dict(
            torch.load(join(path, "critic_net.pth"), map_location=self.device)
        )
        self.q_net.eval()

    def counterfactual_baseline(self, observations, actions, probs):
        """
        Compute per-agent baseline = sum_a pi(a|tau) * Q(tau, a).
        This is the expected Q value under current policy.

        Args:
            observations: [batch*N, obs_size]
            actions:      [batch*N]
            probs:        [batch*N, nr_actions]

        Returns:
            baseline: [batch*N]
        """
        mask = self._get_mask(observations.size(0))
        q_values = self.q_net(observations, mask)  # [batch*N, nr_actions]
        # Expected value: sum over actions weighted by policy probs
        baseline = (probs * q_values).sum(dim=-1)  # [batch*N]
        return baseline.detach()

    def train(self, observations, actions, targets):
        """
        Train critic to minimize MSE loss (Eq. 5 in paper).

        Args:
            observations: [batch*N, obs_size]  (flattened joint obs)
            actions:      [batch*N]
            targets:      [batch*N]  (Monte Carlo returns)
        """
        mask = self._get_mask(observations.size(0))
        q_values = self.q_net(observations, mask)  # [batch*N, nr_actions]

        # Gather Q values for taken actions
        actions_idx = actions.view(-1, 1)
        q_taken = q_values.gather(1, actions_idx).squeeze(1)  # [batch*N]

        targets_flat = targets.view(-1)
        assert q_taken.shape == targets_flat.shape, \
            f"Shape mismatch: {q_taken.shape} vs {targets_flat.shape}"

        loss = F.mse_loss(q_taken, targets_flat)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item()