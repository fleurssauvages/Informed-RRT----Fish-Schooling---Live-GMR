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

import numpy as np
import matplotlib.pyplot as plt
import numba
import time
import pickle
import copy
from numba import njit, prange

from RRTstar_reroute import iterative_blocking_rrtstar, PolyObstacle, plot_tree_and_routes
from RL.env2D import FishGoalEnv2D, get_terminal_goals
from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

def add_cov_ellipses(ax, mu_y, Sigma_y, step=20, n_std=1.5, n_points=40, alpha=0.20):
    """
    Draw ellipses for 2D Gaussian covariances along the trajectory.
    Returns a list of Line2D objects (so you can remove them on refresh).
    """
    lines = []
    for i in range(0, mu_y.shape[0], step):
        S = Sigma_y[i]
        # ensure symmetric
        S = 0.5 * (S + S.T)

        # eig
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-12)

        ang = np.linspace(0, 2 * np.pi, n_points)
        circle = np.stack([np.cos(ang), np.sin(ang)], axis=0)  # (2,P)

        A = V @ np.diag(np.sqrt(w) * n_std)
        ell = (A @ circle).T + mu_y[i]  # (P,2)

        ln, = ax.plot(ell[:, 0], ell[:, 1], "r", lw=1.0, alpha=alpha)
        lines.append(ln)
    return lines


def remove_lines(lines):
    for ln in lines:
        try:
            ln.remove()
        except Exception:
            pass
    lines.clear()

def _merge_or_add(nodes, p, tol):
    """Return index of existing node within tol, else append and return new index."""
    if len(nodes) == 0:
        nodes.append(p)
        return 0
    pts = np.asarray(nodes)
    d2 = np.sum((pts - p)**2, axis=1)
    j = int(np.argmin(d2))
    if d2[j] <= tol * tol:
        # snap-to existing node
        return j
    nodes.append(p)
    return len(nodes) - 1


def build_goal_graph_from_paths(paths_xy, merge_tol=0.5, terminal_self_weight=1.0, scale=1.0):
    nodes = []
    edges = {}
    start_idx = None
    terminal_nodes = set()

    route_node_lists = []   # NEW: list of lists of node ids (one per route)

    for path in paths_xy:
        path = np.asarray(path, dtype=np.float32) * scale
        idxs = []
        for k in range(path.shape[0]):
            idxs.append(_merge_or_add(nodes, path[k], merge_tol))

        route_node_lists.append(idxs)  # NEW

        if start_idx is None:
            start_idx = idxs[0]
        terminal_nodes.add(idxs[-1])

        for a, b in zip(idxs[:-1], idxs[1:]):
            edges[(a, b)] = edges.get((a, b), 0.0) + 1.0

    goals = np.asarray(nodes, dtype=np.float32)
    G = goals.shape[0]
    goal_W = np.zeros((G, G), dtype=np.float32)

    out_sum = np.zeros(G, dtype=np.float32)
    for (a, b), w in edges.items():
        goal_W[a, b] += w
        out_sum[a] += w

    for a in range(G):
        if a in terminal_nodes:
            continue
        s = out_sum[a]
        if s > 1e-12:
            goal_W[a, :] /= s

    for t in terminal_nodes:
        goal_W[t, :] = 0.0
        goal_W[t, t] = float(terminal_self_weight)

    return goals, goal_W, int(start_idx if start_idx is not None else 0), route_node_lists, terminal_nodes


def resample_polyline_by_step(path, step=2.0):
    path = np.asarray(path, dtype=np.float32)
    out = [path[0]]
    acc = 0.0
    for i in range(1, len(path)):
        a = out[-1]
        b = path[i]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        while L >= step:
            a = a + (step / L) * seg
            out.append(a.copy())
            seg = b - a
            L = float(np.linalg.norm(seg))
    if np.linalg.norm(out[-1] - path[-1]) > 1e-6:
        out.append(path[-1])
    return np.asarray(out, dtype=np.float32)

def obstacles_to_segs(obstacles, scale = 1.0):
    """
    obstacles: list of PolyObstacle objects
    returns: segs array (M,4) float32
    """
    segs_all = []

    for obs in obstacles:
        # change this if your attribute name is different
        poly = np.asarray(obs.vertices, dtype=np.float32)

        if poly.shape[0] < 3:
            continue

        # edges i -> i+1 and closing edge last -> first
        p1 = poly
        p2 = np.roll(poly, -1, axis=0)

        segs = np.column_stack([
            p1[:, 0], p1[:, 1],
            p2[:, 0], p2[:, 1]
        ])  # (m,4)

        segs_all.append(segs * scale)

    if not segs_all:
        return np.zeros((0,4), dtype=np.float32)

    return np.vstack(segs_all).astype(np.float32)

def plot_heatmap(H, extent, title=""):
    H_log = np.log1p(H)

    plt.figure(figsize=(6,5))
    plt.imshow(
        H_log,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap="viridis"   # no orange 🙂
    )
    plt.colorbar(label="log(occupancy)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def compute_occupancy(boid_pos, bins=100, bounds=(0,40)):
    """
    boid_pos: (T, N, 2)
    returns:
        H: occupancy histogram
        extent: for imshow
    """
    T, N, _ = boid_pos.shape
    pts = boid_pos.reshape(T * N, 2)

    xmin, xmax = bounds
    ymin, ymax = bounds

    H, xedges, yedges = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )

    return H.T, [xmin, xmax, ymin, ymax]  # transpose for imshow

@njit(parallel=True, fastmath=True)
def _min_d2_history_per_boid(pos_TN2, hist_H2):
    """
    pos_TN2: (Ts, N, 2) float32/float64
    hist_H2: (H, 2)

    Returns:
        min_d2_HN: (H, N) where min_d2_HN[h,j] = min_t ||pos[t,j]-hist[h]||^2
    """
    Ts, N, _ = pos_TN2.shape
    H = hist_H2.shape[0]
    out = np.empty((H, N), dtype=pos_TN2.dtype)

    for j in prange(N):
        for h in range(H):
            hx = hist_H2[h, 0]
            hy = hist_H2[h, 1]
            best = 1e30
            for t in range(Ts):
                dx = pos_TN2[t, j, 0] - hx
                dy = pos_TN2[t, j, 1] - hy
                d2 = dx*dx + dy*dy
                if d2 < best:
                    best = d2
            out[h, j] = best
    return out


@njit(parallel=True, fastmath=True)
def _scores_from_min_d2(min_d2_HN, w_H):
    """
    min_d2_HN: (H,N)
    w_H: (H,)
    Returns scores (N,)
    """
    H, N = min_d2_HN.shape
    scores = np.empty(N, dtype=min_d2_HN.dtype)

    for j in prange(N):
        s = 0.0
        for h in range(H):
            s += w_H[h] * min_d2_HN[h, j]
        scores[j] = s
    return scores

@njit(parallel=True, fastmath=True)
def _ordered_scores_monotone_window(pos_TN2, hist_H2, w_H, win=64):
    """
    pos_TN2: (Ts, N, 2)
    hist_H2: (H, 2)
    w_H: (H,)
    win: lookahead window (in trajectory indices)

    For each boid j:
      t_prev = 0
      for h=0..H-1:
         search t in [t_prev, min(Ts-1, t_prev+win)] for min dist
         t_prev = argmin_t
         score += w[h] * min_dist
    """
    Ts, N, _ = pos_TN2.shape
    H = hist_H2.shape[0]
    scores = np.empty(N, dtype=pos_TN2.dtype)

    for j in prange(N):
        t_prev = 0
        s = 0.0

        for h in range(H):
            hx = hist_H2[h, 0]
            hy = hist_H2[h, 1]

            t_end = t_prev + win
            if t_end >= Ts:
                t_end = Ts - 1

            best = 1e30
            best_t = t_prev

            # search forward only
            for t in range(t_prev, t_end + 1):
                dx = pos_TN2[t, j, 0] - hx
                dy = pos_TN2[t, j, 1] - hy
                d2 = dx*dx + dy*dy
                if d2 < best:
                    best = d2
                    best_t = t

            t_prev = best_t
            s += w_H[h] * best

            # early stop: if we reached the end, remaining history points can’t advance
            if t_prev >= Ts - 1 and h < H - 1:
                # penalize remaining points as if they match last point
                lastx = pos_TN2[Ts - 1, j, 0]
                lasty = pos_TN2[Ts - 1, j, 1]
                for hh in range(h + 1, H):
                    dx = lastx - hist_H2[hh, 0]
                    dy = lasty - hist_H2[hh, 1]
                    s += w_H[hh] * (dx*dx + dy*dy)
                break

        scores[j] = s

    return scores

def select_demos_2d(
    boid_pos_TN2,
    history_points_2d,
    n_demos=15,
    time_stride=5,
    score_time_stride=6,
    max_history=120,
    decay=0.08,
    win=64,
    dtype=np.float32
):
    """
    Order-aware demo selection using monotone time matching with a forward window.
    Much faster than DTW and enforces chronological consistency.
    """

    T, N, _ = boid_pos_TN2.shape

    hist = np.asarray(history_points_2d, dtype=dtype)
    if hist.shape[0] == 0:
        return []
    if hist.shape[0] > max_history:
        hist = hist[-max_history:]
    H = hist.shape[0]

    # weights: more weight to recent history (end of hist)
    idx = np.arange(H, dtype=dtype)
    w = np.exp(-decay * (H - 1 - idx))
    w /= (w.sum() + 1e-12)

    pos = np.asarray(boid_pos_TN2[::score_time_stride], dtype=dtype)  # (Ts,N,2)

    scores = _ordered_scores_monotone_window(pos, hist, w, win=win)

    k = min(n_demos, N)
    idx_best = np.argpartition(scores, k - 1)[:k]
    idx_best = idx_best[np.argsort(scores[idx_best])]

    demos = [boid_pos_TN2[::time_stride, j, :].astype(np.float64) for j in idx_best]
    return demos

def crop_demos_forward_from_point(pos_demos, current_xy, min_len=10):
    """
    For each demo (T,2), find t* closest to current_xy and return demo[t*:].
    Drops demos that become too short.
    """
    cur = np.asarray(current_xy, dtype=float).reshape(1, 2)
    cropped = []
    for traj in pos_demos:
        d2 = np.sum((traj - cur) ** 2, axis=1)
        t0 = int(np.argmin(d2))
        tr = traj[t0:]
        if tr.shape[0] >= min_len:
            cropped.append(tr)
    return cropped

def chaikin_closed(poly, n_iters=2):
    """
    Chaikin corner-cutting for a closed polygon.
    poly: (M,2)
    returns smoother closed polyline (K,2) (not explicitly closed)
    """
    p = np.asarray(poly, dtype=float)
    for _ in range(n_iters):
        p_next = []
        for i in range(len(p)):
            p0 = p[i]
            p1 = p[(i + 1) % len(p)]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            p_next.extend([Q, R])
        p = np.asarray(p_next)
    return p

def resample_closed_polyline(polyline, n_points=16):
    """
    Evenly resample a closed polyline to n_points along arc-length.
    polyline: (M,2), closed implicitly
    returns: (n_points,2)
    """
    p = np.asarray(polyline, dtype=float)
    closed = np.vstack([p, p[0]])
    seg = closed[1:] - closed[:-1]
    L = np.linalg.norm(seg, axis=1)
    per = L.sum()
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s = np.linspace(0.0, per, n_points + 1)[:-1]

    out = np.empty((n_points, 2), dtype=float)
    for k, sk in enumerate(s):
        i = np.searchsorted(cum, sk, side="right") - 1
        i = min(i, len(seg) - 1)
        if L[i] < 1e-12:
            out[k] = closed[i]
        else:
            a = (sk - cum[i]) / L[i]
            out[k] = closed[i] + a * seg[i]
    return out

def round_obstacle(poly, n_iters=2, n_points=16):
    smooth = chaikin_closed(poly, n_iters=n_iters)
    rounded = resample_closed_polyline(smooth, n_points=n_points)
    return rounded

def _orient(a, b, c):
    # 2D cross product (b-a) x (c-a)
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a, b, c, eps=1e-9):
    # c colinear with a-b, and within bounding box
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

def _seg_intersect(p1, p2, q1, q2, eps=1e-9):
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    # general case
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and \
       (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True

    # colinear / touching cases
    if abs(o1) <= eps and _on_segment(p1, p2, q1, eps): return True
    if abs(o2) <= eps and _on_segment(p1, p2, q2, eps): return True
    if abs(o3) <= eps and _on_segment(q1, q2, p1, eps): return True
    if abs(o4) <= eps and _on_segment(q1, q2, p2, eps): return True
    return False

def segment_crosses_obstacles(p, q, segs, eps=1e-9):
    """
    segs: (M,4) array [x1,y1,x2,y2] for obstacle polygon edges
    Returns True if segment p-q intersects/touches any obstacle edge.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    for x1, y1, x2, y2 in segs:
        r1 = np.array([x1, y1], dtype=float)
        r2 = np.array([x2, y2], dtype=float)
        if _seg_intersect(p, q, r1, r2, eps=eps):
            return True
    return False

def pick_route_fraction_nodes(route_node_lists, fractions=(0.5,)):
    """
    Return for each route a list of node-ids picked at given fractions of the route length.
    fractions in [0,1], e.g. (0.5,) for middle, (1/3,2/3) for thirds.
    """
    picked = []
    for idxs in route_node_lists:
        n = len(idxs)
        if n == 0:
            picked.append([])
            continue

        nodes = []
        for f in fractions:
            f = float(f)
            k = int(round(f * (n - 1)))
            k = max(0, min(n - 1, k))
            nodes.append(idxs[k])

        # unique while keeping order
        uniq = []
        seen = set()
        for u in nodes:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        picked.append(uniq)
    return picked


def add_fractional_closest_transitions(
    goals,
    goal_W,
    route_node_lists,
    terminal_nodes,
    segs,
    X=8.0,
    fractions=(0.5,),
    transition_weight=0.25,
    eps=1e-9,
):
    """
    For each route i:
      - pick source nodes at specified fractions along route i
      - for each source node u:
          for each other route j:
              find v in route j minimizing distance ||goals[u]-goals[v]||
              if dist <= X and segment(u,v) does not cross obstacles:
                  add transition u->v

    Then renormalize nonterminal rows of goal_W.
    """
    G = goals.shape[0]
    goal_W = goal_W.copy()

    # precompute arrays for each route for fast closest-point search
    route_pos = []
    for idxs in route_node_lists:
        idxs_arr = np.asarray(idxs, dtype=int)
        if idxs_arr.size == 0:
            route_pos.append((idxs_arr, np.empty((0, 2), dtype=float)))
        else:
            route_pos.append((idxs_arr, goals[idxs_arr]))

    source_nodes_per_route = pick_route_fraction_nodes(route_node_lists, fractions=fractions)

    for i, sources in enumerate(source_nodes_per_route):
        for u in sources:
            if u in terminal_nodes:
                continue

            pu = goals[u]

            for j, (idxs_j, pos_j) in enumerate(route_pos):
                if j == i or idxs_j.size == 0:
                    continue

                # closest node on route j
                d2 = np.sum((pos_j - pu) ** 2, axis=1)
                k = int(np.argmin(d2))
                v = int(idxs_j[k])
                d = float(np.sqrt(d2[k]))

                if d > X:
                    continue

                # collision check
                if segment_crosses_obstacles(pu, goals[v], segs, eps=eps):
                    continue

                goal_W[u, v] += float(transition_weight)

    # renormalize nonterminal rows
    for a in range(G):
        if a in terminal_nodes:
            continue
        s = float(goal_W[a].sum())
        if s > 1e-12:
            goal_W[a] /= s

    # keep terminals as pure self-loop
    for t in terminal_nodes:
        goal_W[t, :] = 0.0
        goal_W[t, t] = 1.0

    return goal_W

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
        PolyObstacle(round_obstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=2, n_points=24)),
        PolyObstacle(round_obstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=2, n_points=24)),
    ]
    

    planner_kwargs = dict(
        step_size=0.7,
        goal_sample_rate=0.01,
        max_iter=12000,
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
                         title=f"Tree + {len(paths)} routes", plotTree=False)
    
    theta_path = "save/best_policy.pkl"
    action = pickle.load(open(theta_path, "rb"))['best_theta']

    scale = 4
    goals, goal_W, start_idx, route_node_lists, terminal_nodes = build_goal_graph_from_paths(paths, scale=scale)
    segs = obstacles_to_segs(obstacles, scale=scale)

    env = FishGoalEnv2D(
        boid_count=400 * len(paths),
        bound=40.0,
        max_steps=600,
        dt=1,
        start=np.array(start, dtype=np.float32),
        goals=np.array([goal]) * scale,
        goal_W=np.array([[1.0]]),
        segs=segs,
        doAnimation=False,
        returnTrajectory=True,
        avoid_r = 1.0,
        start_spread=1.0,
        goal_radius = 5.0
    )

    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos_no_RTT = np.array(info["trajectory_boid_pos"]) 

    env = FishGoalEnv2D(
        boid_count=400 * len(paths),
        bound=40.0,
        max_steps=600,
        dt=1,
        start=np.array(start, dtype=np.float32),
        goals=goals,
        goal_W=goal_W,
        segs=segs,
        doAnimation=False,
        returnTrajectory=True,
        avoid_r = 1.0,
        start_spread=1.0,
        goal_radius = 5.0
    )

    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos_RTT = np.array(info["trajectory_boid_pos"])

    goal_W = add_fractional_closest_transitions(
        goals, goal_W,
        route_node_lists=route_node_lists,
        terminal_nodes=terminal_nodes,
        segs=segs,
        X=30.0,
        fractions=(1/4, 2/4, 3/4),
        transition_weight=0.1
    )

    env = FishGoalEnv2D(
        boid_count=400 * len(paths),
        bound=40.0,
        max_steps=600,
        dt=1,
        start=np.array(start, dtype=np.float32),
        goals=goals,
        goal_W=goal_W,
        segs=segs,
        doAnimation=False,
        returnTrajectory=True,
        avoid_r = 1.0,
        start_spread=1.0,
        goal_radius = 5.0
    )

    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos_RTT_transitions = np.array(info["trajectory_boid_pos"])

    H_no, extent = compute_occupancy(boid_pos_no_RTT, bins=120)
    H_rtt, _     = compute_occupancy(boid_pos_RTT, bins=120)
    H_rtt_transitions, _     = compute_occupancy(boid_pos_RTT_transitions, bins=120)

    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.imshow(np.log1p(H_no), extent=extent, origin="lower",
            aspect="equal", cmap="viridis", vmin=0, vmax=10)
    plt.title("No RRT*")
    plt.colorbar(label="log occupancy")

    plt.subplot(1,3,2)
    plt.imshow(np.log1p(H_rtt), extent=extent, origin="lower",
            aspect="equal", cmap="viridis", vmin=0, vmax=10)
    plt.title("With RRT*")
    plt.colorbar(label="log occupancy")

    plt.subplot(1,3,3)
    plt.imshow(np.log1p(H_rtt_transitions), extent=extent, origin="lower",
            aspect="equal", cmap="viridis", vmin=0, vmax=10)
    plt.title("With RRT* and Transitions")
    plt.colorbar(label="log occupancy")

    plt.tight_layout()
    plt.show()

    # --- GMR parameters (mirrors your example :contentReference[oaicite:5]{index=5} ) ---
    boid_pos= boid_pos_RTT_transitions
    nonzero_mask = np.any(np.abs(boid_pos) > 0.5, axis=(1,2))
    boid_pos = boid_pos[nonzero_mask]

    max_steps = 600
    n_demos = 5
    time_stride = 5
    n_components = 12
    cov_type = "full"

    history_len = 36
    update_period = 0.08
    update_iters = 10
    move_eps = 1e-3

    # --- init cursor/history ---
    x = start.astype(float).copy()
    history = [x.copy()]
    path = [x.copy()]

    history_points = np.array([start] * history_len, dtype=float)
    pos_demos = select_demos_2d(boid_pos, history_points, n_demos=n_demos, time_stride=time_stride)

    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=2)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # walls
    for m in range(segs.shape[0]):
        x1, y1, x2, y2 = segs[m]
        ax.plot([x1, x2], [y1, y2], "k", lw=2.0)

    # terminal goals
    terminal_goals, terminal_idx = get_terminal_goals(goals, goal_W)
    ax.scatter(terminal_goals[:, 0], terminal_goals[:, 1], s=160, marker="*", c="r", label="Terminal goals")
    ax.scatter(start[0], start[1], s=80, c="k", label="Start")

    # demo lines
    demo_lines = []
    for _ in range(n_demos):
        ln, = ax.plot([], [], "k--", lw=1.0, alpha=0.6)
        demo_lines.append(ln)

    def set_demo_lines(demos, tidx=0):
        for i, ln in enumerate(demo_lines):
            if i < len(demos):
                d = demos[i][tidx:, :]
                ln.set_data(d[:, 0], d[:, 1])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    set_demo_lines(pos_demos)

    # model mean + cov
    mu_line, = ax.plot(mu_y[:, 0], mu_y[:, 1], "r", lw=2.5, label="GMR mean")
    cov_lines = add_cov_ellipses(ax, mu_y, Sigma_y, step=25, n_std=1.5, alpha=0.18)

    # cursor + path + history
    cursor_sc = ax.scatter([x[0]], [x[1]], s=90, c="b", label="Cursor")
    path_ln, = ax.plot([x[0]], [x[1]], lw=2.0, c="b", alpha=0.8)
    hist_sc = ax.scatter([x[0]], [x[1]], s=25, c="g", alpha=0.8, label="History")

    ax.legend(loc="upper left")

    x = start.copy()
    last_update_t = time.time()
    last_x_for_update = x.copy()
    dt = 10

    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()

    mu_new, Sigma_new, _, _ = gmr.regress(T=max_steps, pos_dim=3)
    f_cut = 3
    tau = 1/(2*np.pi*f_cut)
    dt_visu = 1/60
    blend = dt_visu / (tau + dt_visu)

    plt.ion()
    while plt.fignum_exists(fig.number):
        now = time.time()
        trans, rot, buttons = spm.read()
        trans = [-trans[1], trans[0]] 
        v = np.array(trans, dtype=float)

        x += v*dt

        # update history/path
        if np.linalg.norm(x - history[-1]) > 1e-6:
            history.append(x.copy())
            if len(history) > history_len:
                history = history[-history_len:]
            path.append(x.copy())

        cursor_sc.set_offsets(np.array([x]))
        P = np.array(path)
        path_ln.set_data(P[:, 0], P[:, 1])

        H = np.array(history)
        hist_sc.set_offsets(H)

        # throttled GMR update
        if (now - last_update_t) >= update_period and np.linalg.norm(x - last_x_for_update) > move_eps:
            last_update_t = now
            last_x_for_update = x.copy()

            pos_demos = select_demos_2d(boid_pos, np.array(history), n_demos=n_demos, time_stride=time_stride)
            pos_demos = crop_demos_forward_from_point(pos_demos, current_xy=history[-1], min_len=12)

            gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
            gmr.fit(pos_demos)
            mu_new, Sigma_new, _, _ = gmr.regress(T=max_steps, pos_dim=3)
            mu_y = (1 - blend) * mu_y + blend * mu_new
            Sigma_y = (1 - blend) * Sigma_y + blend * Sigma_new

            set_demo_lines(pos_demos, 0)
            mu_line.set_data(mu_y[0:, 0], mu_y[0:, 1])

            remove_lines(cov_lines)
            cov_lines[:] = add_cov_ellipses(ax, mu_y[0:], Sigma_y[0:], step=25, n_std=1.5, alpha=0.3)
        else:
            mu_y = (1 - blend) * mu_y + blend * mu_new
            Sigma_y = (1 - blend) * Sigma_y + blend * Sigma_new

        plt.pause(dt_visu)

        if np.linalg.norm(buttons) > 0.5: # Restart by pressing a button
            x = start.astype(float).copy()
            history = [x.copy()] 
            path = [x.copy()] 

    plt.ioff()
    plt.show()
