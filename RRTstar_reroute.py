"""
Informed RRT* (2D) with:
- Polygon obstacles (convex or concave, non-self-intersecting)
- Fast nearest / near queries via scipy.spatial.cKDTree (rebuilt periodically)
- Numba-accelerated collision + geometry kernels
- Parallelized heavy scans (separator scoring + clearance + pruning-mark)

Dependencies:
  pip install numpy matplotlib scipy numba

Run:
  python RRTstar_reroute_numba_full.py
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numba
from numba import njit, prange
import time


# ---------------------------
# Polygon packing (Numba-friendly)
# ---------------------------

def pack_polygons(polys: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack list of polygons (each (m,2)) into:
      V: (N,2) all vertices concatenated
      offsets: (P+1,) start indices, so poly i is V[offsets[i]:offsets[i+1]]
    """
    offsets = [0]
    verts = []
    for poly in polys:
        poly = np.asarray(poly, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ValueError("Each polygon must have shape (m,2) with m>=3.")
        verts.append(poly)
        offsets.append(offsets[-1] + poly.shape[0])
    V = np.vstack(verts) if verts else np.zeros((0, 2), dtype=np.float64)
    return V, np.asarray(offsets, dtype=np.int64)


def compute_poly_aabbs(V: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      aabb_min: (P,2)
      aabb_max: (P,2)
    """
    P = offsets.shape[0] - 1
    aabb_min = np.empty((P, 2), dtype=np.float64)
    aabb_max = np.empty((P, 2), dtype=np.float64)
    for k in range(P):
        s = offsets[k]
        e = offsets[k + 1]
        poly = V[s:e]
        aabb_min[k, 0] = np.min(poly[:, 0])
        aabb_min[k, 1] = np.min(poly[:, 1])
        aabb_max[k, 0] = np.max(poly[:, 0])
        aabb_max[k, 1] = np.max(poly[:, 1])
    return aabb_min, aabb_max


def append_polygon_to_packed(planner, poly: np.ndarray):
    """
    Append ONE polygon to planner.V/planner.offsets/planner.aabb_* incrementally.
    poly: (m,2) float64
    """
    poly = np.asarray(poly, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("poly must be (m,2) with m>=3")

    if planner.V.shape[0] == 0:
        planner.V = poly.copy()
        planner.offsets = np.array([0, poly.shape[0]], dtype=np.int64)
        planner.aabb_min = np.array([[poly[:, 0].min(), poly[:, 1].min()]], dtype=np.float64)
        planner.aabb_max = np.array([[poly[:, 0].max(), poly[:, 1].max()]], dtype=np.float64)
        return

    planner.V = np.vstack([planner.V, poly])

    last = int(planner.offsets[-1])
    planner.offsets = np.append(planner.offsets, last + poly.shape[0]).astype(np.int64)

    amin = np.array([[poly[:, 0].min(), poly[:, 1].min()]], dtype=np.float64)
    amax = np.array([[poly[:, 0].max(), poly[:, 1].max()]], dtype=np.float64)
    planner.aabb_min = np.vstack([planner.aabb_min, amin])
    planner.aabb_max = np.vstack([planner.aabb_max, amax])


# ---------------------------
# Numba kernels
# ---------------------------

@njit(cache=True, fastmath=True)
def sample_unit_disk(rng_u: float, rng_v: float) -> np.ndarray:
    # Uniform in unit disk: r=sqrt(u), theta=2pi*v
    theta = 2.0 * math.pi * rng_v
    r = math.sqrt(rng_u)
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)

@njit(cache=True, fastmath=True)
def orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True, fastmath=True)
def on_segment(ax, ay, bx, by, cx, cy, eps=1e-12) -> bool:
    return (min(ax, bx) - eps <= cx <= max(ax, bx) + eps and
            min(ay, by) - eps <= cy <= max(ay, by) + eps)

@njit(cache=True, fastmath=True)
def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy, eps=1e-12) -> bool:
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    # Proper intersection
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    # Colinear / endpoint touches
    if abs(o1) <= eps and on_segment(ax, ay, bx, by, cx, cy, eps): return True
    if abs(o2) <= eps and on_segment(ax, ay, bx, by, dx, dy, eps): return True
    if abs(o3) <= eps and on_segment(cx, cy, dx, dy, ax, ay, eps): return True
    if abs(o4) <= eps and on_segment(cx, cy, dx, dy, bx, by, eps): return True
    return False

@njit(cache=True, fastmath=True)
def point_in_poly(px, py, V, start, end) -> bool:
    # Ray casting even-odd rule; polygon is V[start:end]
    inside = False
    n = end - start
    for i in range(n):
        x1 = V[start + i, 0]
        y1 = V[start + i, 1]
        j = (i + 1) % n
        x2 = V[start + j, 0]
        y2 = V[start + j, 1]

        cond = (y1 > py) != (y2 > py)
        if cond:
            xinters = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-18) + x1
            if px < xinters:
                inside = not inside
    return inside

@njit(cache=True, fastmath=True)
def segment_intersects_poly(ax, ay, bx, by, V, start, end) -> bool:
    # Endpoints inside polygon => collision
    if point_in_poly(ax, ay, V, start, end) or point_in_poly(bx, by, V, start, end):
        return True

    n = end - start
    for i in range(n):
        cx = V[start + i, 0]
        cy = V[start + i, 1]
        j = (i + 1) % n
        dx = V[start + j, 0]
        dy = V[start + j, 1]
        if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


@njit(cache=True, fastmath=True)
def is_state_valid(px, py, lo, hi, V, offsets, aabb_min, aabb_max) -> bool:
    if px < lo[0] or py < lo[1] or px > hi[0] or py > hi[1]:
        return False
    P = offsets.shape[0] - 1
    for k in range(P):
        # AABB prune
        if px < aabb_min[k, 0] or px > aabb_max[k, 0] or py < aabb_min[k, 1] or py > aabb_max[k, 1]:
            continue
        s = offsets[k]
        e = offsets[k + 1]
        if point_in_poly(px, py, V, s, e):
            return False
    return True


@njit(cache=True, fastmath=True)
def is_segment_valid(ax, ay, bx, by, lo, hi, V, offsets, aabb_min, aabb_max) -> bool:
    if not is_state_valid(ax, ay, lo, hi, V, offsets, aabb_min, aabb_max):
        return False
    if not is_state_valid(bx, by, lo, hi, V, offsets, aabb_min, aabb_max):
        return False

    # segment AABB
    sminx = ax if ax < bx else bx
    smaxx = bx if ax < bx else ax
    sminy = ay if ay < by else by
    smaxy = by if ay < by else ay

    P = offsets.shape[0] - 1
    for k in range(P):
        # AABB overlap prune
        if smaxx < aabb_min[k, 0] or sminx > aabb_max[k, 0] or smaxy < aabb_min[k, 1] or sminy > aabb_max[k, 1]:
            continue
        s = offsets[k]
        e = offsets[k + 1]
        if segment_intersects_poly(ax, ay, bx, by, V, s, e):
            return False
    return True


@njit(cache=True, fastmath=True)
def steer(ax, ay, tx, ty, step_size) -> Tuple[float, float]:
    dx = tx - ax
    dy = ty - ay
    d = math.sqrt(dx * dx + dy * dy)
    if d <= step_size:
        return tx, ty
    inv = step_size / (d + 1e-18)
    return ax + inv * dx, ay + inv * dy

@njit(cache=True, fastmath=True)
def dist2(ax, ay, bx, by) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

@njit(cache=True, fastmath=True)
def dist(ax, ay, bx, by) -> float:
    return math.sqrt(dist2(ax, ay, bx, by))


# --------- Numba-friendly segment intersection point (flag + coords)

@njit(cache=True, fastmath=True)
def segment_intersection_point_numba(ax, ay, bx, by, cx, cy, dx, dy, eps=1e-12):
    """
    Segment AB with CD. Returns (hit, ix, iy).
    If parallel/collinear/no intersection -> (False, 0, 0).
    """
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy

    denom = rx * sy - ry * sx
    if abs(denom) < eps:
        return False, 0.0, 0.0

    qmpx = cx - ax
    qmpy = cy - ay

    t = (qmpx * sy - qmpy * sx) / denom
    u = (qmpx * ry - qmpy * rx) / denom

    if (-eps <= t <= 1.0 + eps) and (-eps <= u <= 1.0 + eps):
        ix = ax + t * rx
        iy = ay + t * ry
        return True, ix, iy

    return False, 0.0, 0.0


@njit(cache=True, fastmath=True)
def point_to_segment_distance_numba(px, py, ax, ay, bx, by, eps=1e-12):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom < eps:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)

    t = (apx * abx + apy * aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    projx = ax + t * abx
    projy = ay + t * aby
    dx = px - projx
    dy = py - projy
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True, fastmath=True, parallel=True)
def clearance_to_obstacles_packed_parallel(px, py, V, offsets, aabb_min, aabb_max):
    """
    Parallel min distance from point to any obstacle edge.
    Parallel over polygons (safe reduction).
    """
    P = offsets.shape[0] - 1
    per_poly_best = np.empty(P, dtype=np.float64)

    for k in prange(P):
        # AABB distance lower bound; skip only if cannot beat current best (not available here),
        # so we compute anyway but it still helps if you later extend this kernel.
        s = offsets[k]
        e = offsets[k + 1]
        m = e - s
        best = 1e300
        for i in range(m):
            i0 = s + i
            i1 = s + ((i + 1) % m)
            ax = V[i0, 0]; ay = V[i0, 1]
            bx = V[i1, 0]; by = V[i1, 1]
            d = point_to_segment_distance_numba(px, py, ax, ay, bx, by)
            if d < best:
                best = d
        per_poly_best[k] = best

    out = 1e300
    for k in range(P):
        if per_poly_best[k] < out:
            out = per_poly_best[k]
    return out


@njit(cache=True, fastmath=True)
def _clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True, fastmath=True, parallel=True)
def _best_for_sep_type_parallel(
    path,              # (T,2)
    seps,              # (S,2,2)
    blocked,           # (S,) uint8
    Vobs, offsets_obs, aabb_min_obs, aabb_max_obs,
    d_ref, w_latest, w_narrow, eps
):
    """
    Returns best (score, sid, edge_k, ix, iy). If none, score=-1e300.
    Parallel over separators.
    """
    T = path.shape[0]
    N_edges = T - 1
    S = seps.shape[0]

    scores = np.full(S, -1e300, dtype=np.float64)
    bestk  = np.full(S, -1, dtype=np.int64)
    ixarr  = np.zeros(S, dtype=np.float64)
    iyarr  = np.zeros(S, dtype=np.float64)

    for sid in prange(S):
        if blocked[sid] != 0:
            continue

        cx = seps[sid, 0, 0]; cy = seps[sid, 0, 1]
        dx = seps[sid, 1, 0]; dy = seps[sid, 1, 1]

        k_latest = -1
        ix = 0.0
        iy = 0.0

        for k in range(N_edges):
            ax = path[k, 0]; ay = path[k, 1]
            bx = path[k + 1, 0]; by = path[k + 1, 1]
            hit, tx, ty = segment_intersection_point_numba(ax, ay, bx, by, cx, cy, dx, dy, eps)
            if hit:
                k_latest = k
                ix = tx
                iy = ty

        if k_latest < 0:
            continue

        denom_edges = (N_edges - 1)
        latest_norm = k_latest / (denom_edges if denom_edges > 0 else 1)

        clear = clearance_to_obstacles_packed_parallel(ix, iy, Vobs, offsets_obs, aabb_min_obs, aabb_max_obs)
        narrow_norm = 1.0 - _clip01(clear / d_ref)

        scores[sid] = w_latest * latest_norm + w_narrow * narrow_norm
        bestk[sid] = k_latest
        ixarr[sid] = ix
        iyarr[sid] = iy

    # reduce
    best_score = -1e300
    best_sid = -1
    best_k = -1
    best_ix = 0.0
    best_iy = 0.0
    for sid in range(S):
        sc = scores[sid]
        if sc > best_score:
            best_score = sc
            best_sid = sid
            best_k = bestk[sid]
            best_ix = ixarr[sid]
            best_iy = iyarr[sid]

    return best_score, best_sid, best_k, best_ix, best_iy


@njit(cache=True, fastmath=True)
def pick_crossing_latest_and_narrowest_numba_parallel(
    path,
    seps_oo, seps_ow,
    blocked_oo, blocked_ow,
    Vobs, offsets_obs, aabb_min_obs, aabb_max_obs,
    d_ref, w_latest=1.0, w_narrow=2.0, eps=1e-12
):
    """
    Returns (found, sep_type, sid, edge_k, ix, iy, score)
    sep_type: 0='oo', 1='ow'
    Priority: oo first; if any found, do not consider ow.
    """
    if path is None or path.shape[0] < 2:
        return False, -1, -1, -1, 0.0, 0.0, -1e300

    sc0, sid0, k0, ix0, iy0 = _best_for_sep_type_parallel(
        path, seps_oo, blocked_oo, Vobs, offsets_obs, aabb_min_obs, aabb_max_obs,
        d_ref, w_latest, w_narrow, eps
    )
    if sid0 >= 0:
        return True, 0, sid0, k0, ix0, iy0, sc0

    sc1, sid1, k1, ix1, iy1 = _best_for_sep_type_parallel(
        path, seps_ow, blocked_ow, Vobs, offsets_obs, aabb_min_obs, aabb_max_obs,
        d_ref, w_latest, w_narrow, eps
    )
    if sid1 >= 0:
        return True, 1, sid1, k1, ix1, iy1, sc1

    return False, -1, -1, -1, 0.0, 0.0, -1e300


# --------- Fused parent choice + rewire (Numba) to reduce Python<->Numba crossings

@njit(cache=True, fastmath=True)
def choose_parent_and_rewire_numba(
    new_idx: int,
    near_idx: np.ndarray,          # int64 array of candidate indices
    X: np.ndarray,                 # (N,2)
    parent: np.ndarray,            # (N,) int64
    cost: np.ndarray,              # (N,) float64
    lo: np.ndarray, hi: np.ndarray,
    V: np.ndarray, offsets: np.ndarray,
    aabb_min: np.ndarray, aabb_max: np.ndarray
):
    """
    Picks best parent for node new_idx among near_idx, then rewires nodes in near_idx through new_idx if cheaper.

    Mutates:
      - parent[new_idx], cost[new_idx]
      - some parent[j], cost[j] for rewired neighbors

    Returns:
      best_parent (int), best_cost (float), rewired_count (int)
    """
    nx = X[new_idx, 0]
    ny = X[new_idx, 1]

    best_parent = parent[new_idx]
    best_cost = cost[new_idx]

    # Choose best parent
    for t in range(near_idx.shape[0]):
        j = near_idx[t]
        if j == new_idx:
            continue
        ax = X[j, 0]
        ay = X[j, 1]
        c = cost[j] + dist(ax, ay, nx, ny)
        if c >= best_cost:
            continue
        if is_segment_valid(ax, ay, nx, ny, lo, hi, V, offsets, aabb_min, aabb_max):
            best_cost = c
            best_parent = j

    parent[new_idx] = best_parent
    cost[new_idx] = best_cost

    # Rewire
    rewired = 0
    for t in range(near_idx.shape[0]):
        j = near_idx[t]
        if j == new_idx or j == best_parent:
            continue
        bx = X[j, 0]
        by = X[j, 1]
        new_cost = best_cost + dist(nx, ny, bx, by)
        if new_cost + 1e-12 >= cost[j]:
            continue
        if is_segment_valid(nx, ny, bx, by, lo, hi, V, offsets, aabb_min, aabb_max):
            parent[j] = new_idx
            cost[j] = new_cost
            rewired += 1

    return best_parent, best_cost, rewired


# --------- Parallel marking of edges crossing last polygon (for pruning)

@njit(cache=True, fastmath=True, parallel=True)
def find_edges_crossing_last_poly_parallel(X, parent, n, V, wall_start, wall_end):
    bad = np.zeros(n, dtype=np.uint8)
    for i in prange(1, n):
        p = parent[i]
        if p < 0:
            continue
        ax = X[p, 0]; ay = X[p, 1]
        bx = X[i, 0]; by = X[i, 1]
        if segment_intersects_poly(ax, ay, bx, by, V, wall_start, wall_end):
            bad[i] = 1
    return bad


# ---------------------------
# Planner
# ---------------------------

@dataclass
class PolyObstacle:
    vertices: np.ndarray  # (m,2)


class InformedRRTStar2D_Fast:
    """
    Array-based implementation (fast):
      - node positions in X (N,2)
      - parent indices in parent (N,)
      - cost-to-come in cost (N,)
    Uses cKDTree for nearest + near queries (rebuilt periodically).
    """

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        obstacles: List[PolyObstacle],
        step_size: float = 0.7,
        goal_sample_rate: float = 0.03,
        max_iter: int = 3000,
        rewire_factor: float = 1.7,
        kd_rebuild_every: int = 30,
        rng_seed: Optional[int] = 0,
    ):
        self.start = np.asarray(start, dtype=np.float64).reshape(2)
        self.goal = np.asarray(goal, dtype=np.float64).reshape(2)
        self.lo = np.asarray(bounds[0], dtype=np.float64).reshape(2)
        self.hi = np.asarray(bounds[1], dtype=np.float64).reshape(2)

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Pack polygons for numba kernels
        polys = [np.asarray(o.vertices, dtype=np.float64) for o in obstacles]
        self.V, self.offsets = pack_polygons(polys)
        self.aabb_min, self.aabb_max = compute_poly_aabbs(self.V, self.offsets)

        self.step_size = float(step_size)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)
        self.rewire_factor = float(rewire_factor)
        self.kd_rebuild_every = int(kd_rebuild_every)

        self.capacity = self.max_iter + 10  # start + possible goal nodes
        self.X = np.empty((self.capacity, 2), dtype=np.float64)
        self.parent = np.full((self.capacity,), -1, dtype=np.int64)
        self.cost = np.full((self.capacity,), np.inf, dtype=np.float64)

        self.n = 1
        self.X[0] = self.start
        self.cost[0] = 0.0

        self.best_goal_idx = -1
        self.c_best = np.inf
        self.c_min = float(np.linalg.norm(self.goal - self.start))

        self.x_center = 0.5 * (self.start + self.goal)
        theta = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
        self.R = np.array([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta),  math.cos(theta)]], dtype=np.float64)

        self.kdtree = cKDTree(self.X[:self.n])
        self._since_kd_rebuild = 0
        self._last_kd_rebuild_n = self.n

        # Safety checks
        if not is_state_valid(self.start[0], self.start[1], self.lo, self.hi, self.V, self.offsets, self.aabb_min, self.aabb_max):
            raise ValueError("Start is invalid (in collision or out of bounds).")
        if not is_state_valid(self.goal[0], self.goal[1], self.lo, self.hi, self.V, self.offsets, self.aabb_min, self.aabb_max):
            raise ValueError("Goal is invalid (in collision or out of bounds).")

    def _maybe_rebuild_kdtree(self):
        self._since_kd_rebuild += 1
        # Adaptive rebuild: baseline every kd_rebuild_every, but scale with n
        threshold = self.kd_rebuild_every 
        if self._since_kd_rebuild >= self.kd_rebuild_every and (self.n - self._last_kd_rebuild_n) >= threshold:
            self.kdtree = cKDTree(self.X[:self.n])
            self._since_kd_rebuild = 0
            self._last_kd_rebuild_n = self.n

    def _neighbor_radius(self) -> float:
        nV = max(self.n, 2)
        gamma = self.rewire_factor * self.step_size * 2.0
        r = gamma * math.sqrt(math.log(nV) / nV)
        return max(r, 1.5 * self.step_size)

    def _sample(self) -> np.ndarray:
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()

        if math.isfinite(self.c_best):
            # Informed sampling inside ellipse
            r1 = self.c_best / 2.0
            r2 = math.sqrt(max(self.c_best * self.c_best - self.c_min * self.c_min, 0.0)) / 2.0

            x = self.x_center.copy()
            for _ in range(60):
                u = random.random()
                v = random.random()
                disk = sample_unit_disk(u, v)
                x_local = np.array([r1 * disk[0], r2 * disk[1]], dtype=np.float64)
                x = self.R @ x_local + self.x_center
                if (self.lo[0] <= x[0] <= self.hi[0]) and (self.lo[1] <= x[1] <= self.hi[1]):
                    return x
            # fallback clamp
            return np.clip(x, self.lo, self.hi)

        # Before first solution: uniform in bounds
        return self.lo + np.random.rand(2) * (self.hi - self.lo)

    def plan(self) -> Optional[np.ndarray]:
        # Local bindings (reduce Python attribute lookup overhead)
        X = self.X
        parent = self.parent
        cost = self.cost
        lo = self.lo
        hi = self.hi
        V = self.V
        offsets = self.offsets
        aabb_min = self.aabb_min
        aabb_max = self.aabb_max
        goal = self.goal
        step_size = self.step_size

        # Near set cap (KNN)
        K_NEAR = 60

        for _ in range(self.max_iter):
            x_rand = self._sample()

            # Nearest via KDTree
            _, nn_idx = self.kdtree.query(x_rand, k=1)
            nn_idx = int(nn_idx)

            ax, ay = X[nn_idx, 0], X[nn_idx, 1]
            tx, ty = float(x_rand[0]), float(x_rand[1])
            nx, ny = steer(ax, ay, tx, ty, step_size)

            if not is_segment_valid(ax, ay, nx, ny, lo, hi, V, offsets, aabb_min, aabb_max):
                continue

            # Add node at next index (tentative)
            new_idx = self.n
            if new_idx >= self.capacity:
                break

            X[new_idx, 0] = nx
            X[new_idx, 1] = ny
            parent[new_idx] = nn_idx
            cost[new_idx] = cost[nn_idx] + dist(ax, ay, nx, ny)

            # Near nodes: KNN (capped) instead of radius explosion
            r = self._neighbor_radius()

            near = self.kdtree.query_ball_point([nx, ny], r)

            # Cap to avoid pathological huge rewiring sets (keeps speed stable)
            K_CAP = 200
            if len(near) > K_CAP:
                near_np = np.asarray(near, dtype=np.int64)
                pts = X[near_np]
                dx = pts[:, 0] - nx
                dy = pts[:, 1] - ny
                order = np.argsort(dx*dx + dy*dy)
                near_np = near_np[order[:K_CAP]]
            else:
                near_np = np.asarray(near, dtype=np.int64)

            if near_np.ndim == 0:
                near_np = near_np.reshape(1)

            near_arr = near_np

            # Fused best-parent + rewire inside Numba
            choose_parent_and_rewire_numba(
                new_idx, near_arr, X, parent, cost, lo, hi, V, offsets, aabb_min, aabb_max
            )

            # Commit node
            self.n += 1

            # Try connect to goal
            gx, gy = goal[0], goal[1]
            if dist(nx, ny, gx, gy) <= step_size:
                if is_segment_valid(nx, ny, gx, gy, lo, hi, V, offsets, aabb_min, aabb_max):
                    goal_cost = cost[new_idx] + dist(nx, ny, gx, gy)

                    goal_idx = self.n
                    if goal_idx < self.capacity:
                        X[goal_idx] = goal
                        parent[goal_idx] = new_idx
                        cost[goal_idx] = goal_cost
                        self.n += 1

                        if goal_cost < self.c_best:
                            self.c_best = goal_cost
                            self.best_goal_idx = goal_idx

            self._maybe_rebuild_kdtree()

        if self.best_goal_idx < 0:
            return None
        return self.extract_path(self.best_goal_idx)

    def extract_path(self, idx: int) -> np.ndarray:
        path = []
        while idx != -1:
            path.append(self.X[idx].copy())
            idx = int(self.parent[idx])
        path.reverse()
        return np.asarray(path, dtype=np.float64)


# ---------------------------
# Visualization + separators + walls
# ---------------------------

def poly_centroid(poly: np.ndarray) -> np.ndarray:
    return np.mean(poly, axis=0)

def build_separators(obstacles, bounds):
    """
    Returns dict with two ordered lists:
      - "oo": obstacle-centroid to obstacle-centroid segments
      - "ow": obstacle-centroid to wall (horizontal to x-bounds) segments
    """
    lo, hi = bounds
    C = [poly_centroid(o.vertices) for o in obstacles]

    oo = []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            oo.append((C[i].copy(), C[j].copy()))

    ow = []
    for c in C:
        ow.append((c.copy(), np.array([lo[0], c[1]])))
        ow.append((c.copy(), np.array([hi[0], c[1]])))

    return {"oo": oo, "ow": ow}


def separators_to_arrays(separators):
    def to_arr(lst):
        arr = np.zeros((len(lst), 2, 2), dtype=np.float64)
        for i, (c, d) in enumerate(lst):
            arr[i, 0, :] = np.asarray(c, dtype=np.float64)
            arr[i, 1, :] = np.asarray(d, dtype=np.float64)
        return arr
    return to_arr(separators.get("oo", [])), to_arr(separators.get("ow", []))


def thick_segment_to_polygon(a: np.ndarray, b: np.ndarray, thickness: float) -> np.ndarray:
    """
    Represent a line segment as a thin rectangle polygon (4 vertices).
    thickness is total wall thickness.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-12:
        t = thickness / 2.0
        return np.array([[a[0]-t, a[1]-t], [a[0]+t, a[1]-t], [a[0]+t, a[1]+t], [a[0]-t, a[1]+t]], dtype=float)

    d = d / L
    n = np.array([-d[1], d[0]])
    t = thickness / 2.0
    p1 = a + n * t
    p2 = a - n * t
    p3 = b - n * t
    p4 = b + n * t
    return np.vstack([p1, p2, p3, p4])


def extract_path_indices(planner) -> List[int]:
    idxs = []
    idx = int(planner.best_goal_idx)
    while idx != -1:
        idxs.append(idx)
        idx = int(planner.parent[idx])
    idxs.reverse()
    return idxs


def prune_subtree(planner, root_cut: int):
    """
    Remove all descendants of root_cut (including root_cut) from the planner tree.
    Keeps indices compact by rebuilding arrays and parent pointers.
    """
    n = planner.n
    parent = planner.parent[:n]

    children = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)

    to_delete = np.zeros(n, dtype=bool)
    stack = [int(root_cut)]
    while stack:
        u = stack.pop()
        if to_delete[u]:
            continue
        to_delete[u] = True
        stack.extend(children[u])

    prune_by_mask(planner, to_delete)


def prune_by_mask(planner, to_delete: np.ndarray):
    """Prune all nodes where to_delete[i]=True (and rebuild arrays compactly)."""
    n = planner.n
    to_delete = np.asarray(to_delete, dtype=bool)
    if to_delete.shape[0] != n:
        raise ValueError("to_delete mask must have length planner.n")

    keep = np.where(~to_delete)[0]
    if len(keep) == 0:
        raise RuntimeError("Prune removed everything (unexpected).")

    parent_old = planner.parent[:n]

    new_id = -np.ones(n, dtype=int)
    for newi, oldi in enumerate(keep):
        new_id[oldi] = newi

    planner.X[:len(keep)] = planner.X[keep]
    planner.cost[:len(keep)] = planner.cost[keep]

    new_parent = np.full(len(keep), -1, dtype=np.int64)
    for newi, oldi in enumerate(keep):
        p = int(parent_old[oldi])
        new_parent[newi] = -1 if p < 0 else int(new_id[p])
    planner.parent[:len(keep)] = new_parent

    planner.n = len(keep)

    if planner.best_goal_idx >= 0:
        if planner.best_goal_idx < n and to_delete[planner.best_goal_idx]:
            planner.best_goal_idx = -1
            planner.c_best = np.inf
        else:
            planner.best_goal_idx = int(new_id[planner.best_goal_idx])

    planner.kdtree = cKDTree(planner.X[:planner.n])
    planner._since_kd_rebuild = 0
    planner._last_kd_rebuild_n = planner.n


def prune_subtrees_crossing_last_added_obstacle(planner):
    """After adding a new (wall) polygon obstacle at the end of planner.V/offsets,
    prune every subtree whose *incoming edge* crosses that new polygon."""
    n = planner.n
    if n <= 1:
        return

    if planner.offsets.shape[0] < 2:
        return

    wall_start = int(planner.offsets[-2]) if planner.offsets.shape[0] >= 3 else int(planner.offsets[0])
    wall_end = int(planner.offsets[-1]) if planner.offsets.shape[0] >= 3 else int(planner.offsets[1])

    bad_mask_u8 = find_edges_crossing_last_poly_parallel(
        planner.X, planner.parent, planner.n, planner.V, wall_start, wall_end
    )
    bad_roots = np.where(bad_mask_u8 == 1)[0].tolist()
    if not bad_roots:
        return

    # Build children lists once (Python), then DFS from bad roots
    parent = planner.parent[:n]
    children = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)

    to_delete = np.zeros(n, dtype=bool)
    stack = [int(r) for r in bad_roots]
    while stack:
        u = stack.pop()
        if to_delete[u]:
            continue
        to_delete[u] = True
        stack.extend(children[u])

    prune_by_mask(planner, to_delete)


def add_wall_obstacle(planner, obstacles_list, seg_a, seg_b, wall_thickness=0.15):
    wall_poly = thick_segment_to_polygon(seg_a, seg_b, wall_thickness)
    obstacles_list.append(PolyObstacle(wall_poly))
    append_polygon_to_packed(planner, wall_poly)


def iterative_blocking_rrtstar(start, goal, bounds, obstacles, planner_kwargs,
                               wall_thickness=0.15, max_rounds=50):
    """
    Returns list of found paths. Each round:
      - plan
      - find latest crossed separator (Numba-parallel)
      - prune after safe node
      - add wall obstacle for that separator (incremental packing)
      - prune any tree edges now colliding with added wall
      - continue planning
    """
    obstacles_work = list(obstacles)
    separators = build_separators(obstacles_work, bounds)

    seps_oo, seps_ow = separators_to_arrays(separators)
    blocked_oo = np.zeros(seps_oo.shape[0], dtype=np.uint8)
    blocked_ow = np.zeros(seps_ow.shape[0], dtype=np.uint8)

    planner = InformedRRTStar2D_Fast(
        start=start, goal=goal, bounds=bounds, obstacles=obstacles_work, **planner_kwargs
    )

    paths = []
    for _round in range(max_rounds):
        path = planner.plan()
        if path is None:
            break
        paths.append(path)

        d_ref = 2.0 * planner.step_size
        found, sep_type_i, sid, edge_k, ix, iy, score = pick_crossing_latest_and_narrowest_numba_parallel(
            path,
            seps_oo, seps_ow,
            blocked_oo, blocked_ow,
            planner.V, planner.offsets,
            planner.aabb_min, planner.aabb_max,
            d_ref=d_ref, w_latest=1.0, w_narrow=2.0
        )

        if not found:
            break

        if sep_type_i == 0:
            blocked_oo[sid] = 1
            sep_type = "oo"
        else:
            blocked_ow[sid] = 1
            sep_type = "ow"

        idxs = extract_path_indices(planner)
        safe_idx = idxs[edge_k]
        cut_idx  = idxs[edge_k + 1]

        prune_subtree(planner, cut_idx)

        a, b = separators[sep_type][sid]
        add_wall_obstacle(planner, obstacles_work, a, b, wall_thickness=wall_thickness)

        prune_subtrees_crossing_last_added_obstacle(planner)

        planner.best_goal_idx = -1
        planner.c_best = np.inf

    return paths, obstacles_work, planner


def plot_tree_and_routes(start, goal, bounds, obstacles, X, parent, n, paths,
                         title="Routes", tree_alpha=0.18, tree_lw=0.7, plotTree=True):
    lo, hi = bounds
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    for obs in obstacles:
        poly = np.asarray(obs.vertices)
        ax.fill(poly[:, 0], poly[:, 1], color="lightgray", alpha=0.8, zorder=1)
        ax.plot(np.r_[poly[:, 0], poly[0, 0]],
                np.r_[poly[:, 1], poly[0, 1]],
                color="gray", lw=1.0, zorder=2)

    if plotTree:
        Xn = X[:n]
        pn = parent[:n].astype(int)
        for i in range(n):
            p = pn[i]
            if p >= 0:
                ax.plot([Xn[i, 0], Xn[p, 0]],
                        [Xn[i, 1], Xn[p, 1]],
                        lw=tree_lw, alpha=tree_alpha, color="k", zorder=0)

    ax.scatter([start[0]], [start[1]], s=90, marker="o", zorder=5)
    ax.scatter([goal[0]],  [goal[1]],  s=140, marker="*", zorder=5)

    if paths:
        cmap = plt.get_cmap("winter")  # blue->green
        m = len(paths)
        for i, path in enumerate(paths):
            if path is None or len(path) < 2:
                continue
            c = cmap(i / max(1, m - 1))
            lw = 3.0 if i == m - 1 else 2.3
            ax.plot(path[:, 0], path[:, 1], lw=lw, color=c, zorder=6, label=f"route {i+1}")

        if m <= 12:
            ax.legend(loc="best", frameon=True)

    ax.set_title(title)
    plt.show()

# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Threading: use all available threads by default
    try:
        numba.set_num_threads(numba.get_num_threads())
    except Exception:
        pass
    
    start = np.array([1.0, 1.0])
    goal  = np.array([9.0, 9.0])
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

    obstacles = [
        PolyObstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]])),
        PolyObstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]])),
        PolyObstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]])),
        PolyObstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]])),
        PolyObstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]])),
        PolyObstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]])),
        PolyObstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]])),
    ]

    planner_kwargs = dict(
        step_size=0.7,
        goal_sample_rate=0.01,
        max_iter=2500,
        rewire_factor=1.7,
        kd_rebuild_every=25,
        rng_seed=2,
    )

    t0 = time.time()
    paths, obstacles_final, planner = iterative_blocking_rrtstar(
        start, goal, bounds, obstacles, planner_kwargs,
        wall_thickness=0.18,
        max_rounds=20
    )
    t1 = time.time()

    print(f"Found {len(paths)} distinct routes before failure in {t1-t0:.2f} seconds")

    plot_tree_and_routes(start, goal, bounds, obstacles,
                         planner.X, planner.parent, planner.n,
                         paths,
                         title=f"{len(paths)} routes", plotTree=False)