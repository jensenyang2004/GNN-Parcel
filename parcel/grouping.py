"""
PARCEL Spatial Grouping
=======================
Implements Section 4.1 of the paper:
  - Bounding region Gi = {p in V | d(p, pstart_i) <= Ralloc}
  - Two agents grouped if their bounding regions overlap (Gi ∩ Gj != ∅)
  - Grouping is transitive: connected components via union-find
  - Grouping mask M: M[i,j] = 0 if same group, -inf otherwise
"""

import torch
import collections


def compute_bounding_region(start_pos, ralloc, rows, cols, obstacle_map):
    """
    BFS from start_pos to find all cells reachable within ralloc steps
    (shortest path distance <= ralloc), excluding obstacles.

    Args:
        start_pos: (row, col) tuple
        ralloc: int, allocation radius
        rows, cols: grid dimensions
        obstacle_map: bool tensor [rows, cols], True = obstacle

    Returns:
        set of (row, col) tuples within the bounding region
    """
    region = set()
    queue = collections.deque()
    queue.append((start_pos[0], start_pos[1], 0))
    visited = set()
    visited.add((start_pos[0], start_pos[1]))

    while queue:
        r, c, dist = queue.popleft()
        region.add((r, c))
        if dist >= ralloc:
            continue
        # 4-connected neighbors (no diagonals, matching grid movement)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not obstacle_map[nr][nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
    return region


def compute_groups(start_positions, ralloc, rows, cols, obstacle_map):
    """
    Compute spatial groups via connected components.
    Two agents i, j are in the same group if Gi ∩ Gj != ∅ (transitive).

    Args:
        start_positions: list of (row, col) tuples, length N
        ralloc: int
        rows, cols: grid dimensions
        obstacle_map: bool tensor or 2D list

    Returns:
        List of sets, each set containing agent indices in the same group.
        Example: [{0, 1}, {2}, {3, 4}]
    """
    N = len(start_positions)

    # --- Compute bounding regions for all agents ---
    regions = [
        compute_bounding_region(pos, ralloc, rows, cols, obstacle_map)
        for pos in start_positions
    ]

    # --- Union-Find for connected components ---
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # --- Check pairwise overlaps ---
    for i in range(N):
        for j in range(i + 1, N):
            if regions[i] & regions[j]:  # non-empty intersection = potential conflict
                union(i, j)

    # --- Build group sets ---
    group_dict = collections.defaultdict(set)
    for i in range(N):
        root = find(i)
        group_dict[root].add(i)

    return list(group_dict.values())


def build_grouping_mask(groups, N, device):
    """
    Build the NxN grouping mask M.

    M[i, j] = 0.0   if agents i and j belong to the same spatial group
    M[i, j] = -inf  if agents i and j belong to different groups

    When added to attention logits before softmax:
      - Same group: attention is computed normally
      - Different group: softmax weight → 0 (agents are invisible to each other)

    Args:
        groups: list of sets of agent indices
        N: total number of agents
        device: torch device

    Returns:
        FloatTensor of shape [N, N]
    """
    mask = torch.full((N, N), float('-inf'), dtype=torch.float32, device=device)
    for group in groups:
        group_list = list(group)
        for i in group_list:
            for j in group_list:
                mask[i, j] = 0.0
    return mask


def union_grouping_masks(masks):
    """
    Aggregate multiple per-episode grouping masks into one epoch-level mask.

    Strategy: Union — if agents i and j were ever in the same group during any
    episode this epoch, they remain connected. This ensures training consistency
    across all Y episodes.

    Args:
        masks: list of FloatTensor [N, N], one per episode

    Returns:
        FloatTensor [N, N] — the union mask
    """
    if len(masks) == 0:
        raise ValueError("Cannot union empty list of masks")
    N = masks[0].size(0)
    device = masks[0].device
    result = torch.full((N, N), float('-inf'), dtype=torch.float32, device=device)
    for mask in masks:
        # Wherever any episode had i,j connected (mask[i,j] == 0), keep connected
        connected = (mask == 0.0)
        result[connected] = 0.0
    return result


def build_edge_weights(groups, start_positions, ralloc, N, device):
    """
    Build NxN edge feature matrix for the edge-encoded GCN critic.

    Each edge (i→j) carries a 3-dim feature vector [dr, dc, w]:
        dr = (row_j - row_i) / denom   signed normalized row displacement
        dc = (col_j - col_i) / denom   signed normalized col displacement
        w  = max(0, 1 - |dr| - |dc|)  proximity weight (same as before)

    Agents in different groups get [0, 0, 0].
    Self-loops get [0, 0, 1] (no direction, max proximity).

    The directional components let the critic learn to attend differently
    based on where a neighbor is relative to the agent — e.g. attend more
    to agents that are ahead in the direction of intended movement.

    Args:
        groups:          list of sets of agent indices
        start_positions: list of (row, col) tuples, length N
        ralloc:          int, current allocation radius
        N:               total number of agents
        device:          torch device

    Returns:
        FloatTensor [N, N, 3]  — axis 2 is (dr, dc, w)
    """
    features = torch.zeros(N, N, 3, dtype=torch.float32, device=device)

    # Self-loops: no direction, max proximity
    for i in range(N):
        features[i, i, 2] = 1.0

    denom = max(2 * ralloc, 1)  # avoid division by zero at ralloc=0
    for group in groups:
        group_list = list(group)
        for i in group_list:
            for j in group_list:
                if i == j:
                    continue
                ri, ci = start_positions[i]
                rj, cj = start_positions[j]
                dr = (rj - ri) / denom
                dc = (cj - ci) / denom
                w = max(0.0, 1.0 - abs(dr) - abs(dc))
                features[i, j, 0] = dr
                features[i, j, 1] = dc
                features[i, j, 2] = w

    return features


def group_summary(groups, N):
    """
    Human-readable summary of current grouping.
    Useful for debugging and logging.
    """
    independent = sum(1 for g in groups if len(g) == 1)
    coordinated = sum(len(g) for g in groups if len(g) > 1)
    n_groups = len(groups)
    return (f"Groups: {n_groups} total | "
            f"Independent agents: {independent}/{N} | "
            f"Coordinated agents: {coordinated}/{N} | "
            f"Group sizes: {sorted([len(g) for g in groups], reverse=True)}")