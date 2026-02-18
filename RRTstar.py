"""
Informed RRT* (2D) with:
- Polygon obstacles (convex or concave, non-self-intersecting)
- Fast nearest / near queries via scipy.spatial.cKDTree (rebuilt periodically)
- Numba-accelerated collision + geometry kernels

Dependencies:
  pip install numpy matplotlib scipy numba

Run:
  python informed_rrtstar_2d_fast.py
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import cKDTree
import numba as nb


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


# ---------------------------
# Numba kernels
# ---------------------------

@nb.njit(cache=True, fastmath=True)
def sample_unit_disk(rng_u: float, rng_v: float) -> np.ndarray:
    # Uniform in unit disk: r=sqrt(u), theta=2pi*v
    theta = 2.0 * math.pi * rng_v
    r = math.sqrt(rng_u)
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)

@nb.njit(cache=True, fastmath=True)
def orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@nb.njit(cache=True, fastmath=True)
def on_segment(ax, ay, bx, by, cx, cy, eps=1e-12) -> bool:
    return (min(ax, bx) - eps <= cx <= max(ax, bx) + eps and
            min(ay, by) - eps <= cy <= max(ay, by) + eps)

@nb.njit(cache=True, fastmath=True)
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

@nb.njit(cache=True, fastmath=True)
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

@nb.njit(cache=True, fastmath=True)
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

@nb.njit(cache=True, fastmath=True)
def is_state_valid(px, py, lo, hi, V, offsets) -> bool:
    if px < lo[0] or py < lo[1] or px > hi[0] or py > hi[1]:
        return False
    # check inside any polygon
    P = offsets.shape[0] - 1
    for k in range(P):
        s = offsets[k]
        e = offsets[k + 1]
        if point_in_poly(px, py, V, s, e):
            return False
    return True

@nb.njit(cache=True, fastmath=True)
def is_segment_valid(ax, ay, bx, by, lo, hi, V, offsets) -> bool:
    if not is_state_valid(ax, ay, lo, hi, V, offsets):
        return False
    if not is_state_valid(bx, by, lo, hi, V, offsets):
        return False
    P = offsets.shape[0] - 1
    for k in range(P):
        s = offsets[k]
        e = offsets[k + 1]
        if segment_intersects_poly(ax, ay, bx, by, V, s, e):
            return False
    return True

@nb.njit(cache=True, fastmath=True)
def steer(ax, ay, tx, ty, step_size) -> Tuple[float, float]:
    dx = tx - ax
    dy = ty - ay
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= step_size:
        return tx, ty
    inv = step_size / (dist + 1e-18)
    return ax + inv * dx, ay + inv * dy

@nb.njit(cache=True, fastmath=True)
def dist2(ax, ay, bx, by) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

@nb.njit(cache=True, fastmath=True)
def dist(ax, ay, bx, by) -> float:
    return math.sqrt(dist2(ax, ay, bx, by))


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
    Uses cKDTree for nearest + radius queries (rebuilt periodically).
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

        self.step_size = float(step_size)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)
        self.rewire_factor = float(rewire_factor)
        self.kd_rebuild_every = int(kd_rebuild_every)

        self.capacity = self.max_iter + 5  # start + possible goal nodes
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

        # Safety checks
        if not is_state_valid(self.start[0], self.start[1], self.lo, self.hi, self.V, self.offsets):
            raise ValueError("Start is invalid (in collision or out of bounds).")
        if not is_state_valid(self.goal[0], self.goal[1], self.lo, self.hi, self.V, self.offsets):
            raise ValueError("Goal is invalid (in collision or out of bounds).")

    def _maybe_rebuild_kdtree(self):
        self._since_kd_rebuild += 1
        if self._since_kd_rebuild >= self.kd_rebuild_every:
            self.kdtree = cKDTree(self.X[:self.n])
            self._since_kd_rebuild = 0

    def _neighbor_radius(self) -> float:
        nV = max(self.n, 2)
        gamma = self.rewire_factor * self.step_size * 2.0
        r = gamma * math.sqrt(math.log(nV) / nV)
        return max(r, 1.5 * self.step_size)

    def _sample(self) -> np.ndarray:
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()

        if math.isfinite(self.c_best):
            # Informed: sample inside ellipse
            r1 = self.c_best / 2.0
            r2 = math.sqrt(max(self.c_best * self.c_best - self.c_min * self.c_min, 0.0)) / 2.0

            for _ in range(60):
                u = random.random()
                v = random.random()
                disk = sample_unit_disk(u, v)
                x_local = np.array([r1 * disk[0], r2 * disk[1]], dtype=np.float64)
                x = self.R @ x_local + self.x_center
                if (self.lo[0] <= x[0] <= self.hi[0]) and (self.lo[1] <= x[1] <= self.hi[1]):
                    return x
            # fallback clamp
            return clamp(x, self.lo, self.hi)

        # Before first solution: uniform in bounds
        return self.lo + np.random.rand(2) * (self.hi - self.lo)

    def plan(self) -> Optional[np.ndarray]:
        for _ in range(self.max_iter):
            x_rand = self._sample()

            # Nearest via KDTree
            _, nn_idx = self.kdtree.query(x_rand, k=1)
            nn_idx = int(nn_idx)

            ax, ay = self.X[nn_idx, 0], self.X[nn_idx, 1]
            tx, ty = float(x_rand[0]), float(x_rand[1])
            nx, ny = steer(ax, ay, tx, ty, self.step_size)

            if not is_segment_valid(ax, ay, nx, ny, self.lo, self.hi, self.V, self.offsets):
                continue

            # Near nodes via KDTree radius query
            r = self._neighbor_radius()
            near = self.kdtree.query_ball_point([nx, ny], r)

            # Choose parent: best cost among near that has collision-free edge
            best_parent = nn_idx
            best_cost = self.cost[nn_idx] + dist(ax, ay, nx, ny)

            for i in near:
                i = int(i)
                px, py = self.X[i, 0], self.X[i, 1]
                if not is_segment_valid(px, py, nx, ny, self.lo, self.hi, self.V, self.offsets):
                    continue
                c = self.cost[i] + dist(px, py, nx, ny)
                if c < best_cost:
                    best_cost = c
                    best_parent = i

            # Add node
            new_idx = self.n
            if new_idx >= self.capacity:
                # (shouldn't happen with our capacity choice)
                break
            self.X[new_idx, 0] = nx
            self.X[new_idx, 1] = ny
            self.parent[new_idx] = best_parent
            self.cost[new_idx] = best_cost
            self.n += 1

            # Rewire
            for i in near:
                i = int(i)
                if i == new_idx:
                    continue
                px, py = self.X[i, 0], self.X[i, 1]
                new_cost = self.cost[new_idx] + dist(nx, ny, px, py)
                if new_cost + 1e-12 < self.cost[i]:
                    if is_segment_valid(nx, ny, px, py, self.lo, self.hi, self.V, self.offsets):
                        self.parent[i] = new_idx
                        self.cost[i] = new_cost

            # Try connect to goal
            gx, gy = self.goal[0], self.goal[1]
            if dist(nx, ny, gx, gy) <= self.step_size:
                if is_segment_valid(nx, ny, gx, gy, self.lo, self.hi, self.V, self.offsets):
                    goal_cost = self.cost[new_idx] + dist(nx, ny, gx, gy)

                    # Add goal as a node (so rewiring can improve it)
                    goal_idx = self.n
                    if goal_idx < self.capacity:
                        self.X[goal_idx] = self.goal
                        self.parent[goal_idx] = new_idx
                        self.cost[goal_idx] = goal_cost
                        self.n += 1

                        if goal_cost < self.c_best:
                            self.c_best = goal_cost
                            self.best_goal_idx = goal_idx

            # KDTree rebuild periodically (since cKDTree is static)
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
# Visualization
# ---------------------------

def plot_scene(
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    obstacles: List[PolyObstacle],
    X: np.ndarray,
    parent: np.ndarray,
    n: int,
    path: Optional[np.ndarray],
    title: str = "Informed RRT* (2D) — fast",
):
    lo, hi = bounds
    fig, ax = plt.subplots(figsize=(7, 7))

    # Obstacles
    for obs in obstacles:
        ax.add_patch(MplPolygon(obs.vertices, closed=True,
                                facecolor="0.6", edgecolor="black", alpha=0.35))

    # Tree edges
    for i in range(n):
        p = int(parent[i])
        if p == -1:
            continue
        ax.plot([X[p, 0], X[i, 0]], [X[p, 1], X[i, 1]],
                linewidth=0.6, alpha=0.30, color="0.15")

    # Path
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], linewidth=2.5)  # default color

    # Start/Goal
    ax.scatter([start[0]], [start[1]], s=70, marker="o", color="tab:green", zorder=5)
    ax.scatter([goal[0]], [goal[1]], s=90, marker="*", color="tab:red", zorder=5)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.show()


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    start = np.array([1.0, 1.0])
    goal  = np.array([9.0, 9.0])
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

    # Any polygons (convex or concave), avoid self-intersections.
    obstacles = [
        PolyObstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]])),
        PolyObstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]])),
        PolyObstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]])),
    ]

    planner = InformedRRTStar2D_Fast(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=0.7,
        goal_sample_rate=0.03,
        max_iter=1600,
        rewire_factor=1.7,
        kd_rebuild_every=25,   # smaller = more accurate NN/near; larger = faster rebuild overhead
        rng_seed=2,
    )

    path = planner.plan()
    if path is None:
        print("No path found.")
        title = "Informed RRT* (2D) — no solution"
    else:
        print(f"Found path with {len(path)} waypoints, cost ~ {planner.c_best:.3f}")
        title = f"Informed RRT* (2D) — cost: {planner.c_best:.2f}"

    plot_scene(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        X=planner.X,
        parent=planner.parent,
        n=planner.n,
        path=path,
        title=title,
    )
