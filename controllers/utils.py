import numpy as np

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def compute_alpha(
    u_h: np.ndarray,
    v_ref: np.ndarray,
    Sigma: np.ndarray | float,
    *,
    alpha_max: float = 1.0,
    # alignment gate
    c0: float = 0.7,       # cosine threshold: must be "somewhat along"
    k_a: float = 10.0,     # sharpness
    u_deadzone: float = 1e-3,
    # confidence gate (scalar uncertainty s)
    s0: float = 10.0,      # uncertainty threshold (tune to your Sigma scale)
    k_s: float = 10.0,
) -> float:
    """
    Returns alpha in [0, alpha_max]. Increases when:
      - user input aligns with reference direction, AND
      - Sigma is small (confident).
    Drops when either worsens.
    """

    u_h = np.asarray(u_h, dtype=float).reshape(-1)
    v_ref = np.asarray(v_ref, dtype=float).reshape(-1)

    # --- deadzone: if user not pushing, don't assist
    if np.linalg.norm(u_h) < u_deadzone or np.linalg.norm(v_ref) < 1e-12:
        alpha_star = 0.0
    else:
        # alignment cosine in [-1, 1]
        c = float(u_h @ v_ref) / (float(np.linalg.norm(u_h) * np.linalg.norm(v_ref)) + 1e-12)
        c = max(-1.0, min(1.0, c))

        g_align = _sigmoid(k_a * (c - c0))

        # scalar uncertainty s
        if np.isscalar(Sigma):
            s = float(Sigma)
        else:
            Sigma = np.asarray(Sigma, dtype=float)
            # robust choice: sqrt(trace(Sigma))
            s = float(np.sqrt(np.trace(Sigma)))

        g_sigma = _sigmoid(k_s * (s0 - s))  # smaller s -> closer to 1

        alpha_star = alpha_max * g_sigma * g_align

    # clamp
    alpha = max(0.0, min(alpha_max, alpha_star))
    return alpha


def compute_terminal_nodes(goal_W, tol=1e-12):
    G = goal_W.shape[0]
    terminal = np.zeros(G, dtype=bool)

    for i in range(G):
        row = goal_W[i]

        # indices with outgoing weight
        outgoing = np.where(row > tol)[0]

        if len(outgoing) == 0:
            # no outgoing edges
            terminal[i] = True

        elif len(outgoing) == 1 and outgoing[0] == i:
            # only self-loop
            terminal[i] = True

    return terminal

def interpolate_traj(p0, p1, n):
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)

    t = np.linspace(0.0, 1.0, n)[:, None]   # shape (n,1)
    traj = p0 + t * (p1 - p0)

    return traj

def build_promp_library(ref_trajs, n_basis=25, ridge=1e-6, eps=1e-9):
    """
    Fit a simple ProMP-like model per reference trajectory:
    - Assume each ref_traj is a single demonstration of that trajectory hypothesis.
    - Fit basis weights for x(t) using ridge regression.
    - Store mean weight vector (w_mean) and a small covariance (W_cov) for likelihood smoothing.

    Parameters
    ----------
    ref_trajs : list of arrays, each (T,d)
    n_basis : number of RBF basis functions
    ridge : ridge regularization for weight fit

    Returns
    -------
    lib : dict containing basis settings + per-trajectory weights
    """
    # Determine dims
    d = ref_trajs[0].shape[1]
    # RBF centers in phase [0,1]
    centers = np.linspace(0.0, 1.0, n_basis)
    width = (centers[1] - centers[0]) * 1.5 if n_basis > 1 else 0.25

    def Phi(phase):
        # phase: (T,)
        phase = phase[:, None]  # (T,1)
        B = np.exp(-0.5 * ((phase - centers[None, :]) / (width + eps)) ** 2)  # (T,K)
        # normalize rows
        B = B / (B.sum(axis=1, keepdims=True) + eps)
        return B  # (T,K)

    traj_models = []
    for traj in ref_trajs:
        T, d_check = traj.shape
        assert d_check == d
        phase = np.linspace(0.0, 1.0, T)
        B = Phi(phase)  # (T,K)

        # Fit weights per dimension: x(t) ≈ B(t) @ W, W shape (K,d)
        # Ridge: (B^T B + λI) W = B^T X
        BtB = B.T @ B
        W = np.linalg.solve(BtB + ridge * np.eye(n_basis), B.T @ traj)  # (K,d)

        # Store with small covariance to avoid zero-likelihood (since single demo)
        # You can tune W_cov_scale depending on expected variability.
        W_cov_scale = 1e-3
        W_cov = W_cov_scale * np.eye(n_basis * d)

        traj_models.append({
            "T": T,
            "W_mean": W,        # (K,d)
            "W_cov": W_cov,     # (K*d,K*d)
        })

    return {
        "d": d,
        "n_basis": n_basis,
        "centers": centers,
        "width": width,
        "traj_models": traj_models,
    }

def _sample_line(p0, p1, step=0.02, min_points=10, eps=1e-9):
    """Sample a straight line from p0 to p1 with approx spacing `step`."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dist = np.linalg.norm(p1 - p0)
    n = max(min_points, int(np.ceil(dist / max(step, eps))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * p0 + t * p1  # (n,d)

def _sample_polyline(points, step=0.02, min_points_per_seg=10):
    """Sample a polyline defined by a list of points."""
    out = []
    for a, b in zip(points[:-1], points[1:]):
        seg = _sample_line(a, b, step=step, min_points=min_points_per_seg)
        if out:
            seg = seg[1:]  # avoid duplicate junction point
        out.append(seg)
    return np.vstack(out) if out else np.asarray(points, float)

def make_ref_trajs(list_of_point_lists, step=0.02, min_points_per_seg=10):
    """
    Given multiple lists of points, return one sampled straight-line
    trajectory (polyline) per list.

    Parameters
    ----------
    list_of_point_lists : list of arrays/lists, each shape (M_i, d)
    step : approximate spacing between samples
    min_points_per_seg : minimum points per segment

    Returns
    -------
    ref_trajs : list of arrays, each (T_i, d)
    """

    ref_trajs = []

    for pts in list_of_point_lists:
        pts = np.asarray(pts, float)
        assert pts.ndim == 2, "Each element must be (M_i, d)"

        traj_segments = []

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]

            dist = np.linalg.norm(p1 - p0)
            n = max(min_points_per_seg, int(np.ceil(dist / step)) + 1)

            t = np.linspace(0.0, 1.0, n)[:, None]
            seg = (1 - t) * p0 + t * p1

            if i > 0:
                seg = seg[1:]  # avoid duplicate junction points

            traj_segments.append(seg)

        if traj_segments:
            traj = np.vstack(traj_segments)
        else:
            traj = pts.copy()

        ref_trajs.append(traj)

    return ref_trajs

def segs_to_wall_mesh(segs, height=10_000.0, thickness=0.05, z0=0.0, cap=True):
    """
    segs: (N,4) with [x1,y1,x2,y2]
    Returns:
      verts: (M,3)
      faces: (K,3) int32 triangle indices
    """
    segs = np.asarray(segs, dtype=np.float32)
    verts_all = []
    faces_all = []
    v_off = 0

    for x1, y1, x2, y2 in segs:
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        d = p2 - p1
        L = float(np.linalg.norm(d))
        if L < 1e-8:
            continue

        # unit normal to the segment (2D)
        n = np.array([-d[1], d[0]], dtype=np.float32) / L
        o = 0.5 * thickness * n

        # bottom rectangle corners (counter-clockwise around the segment)
        b0 = p1 + o
        b1 = p2 + o
        b2 = p2 - o
        b3 = p1 - o

        z1 = z0 + float(height)

        # 8 vertices: bottom then top
        v = np.array([
            [b0[0], b0[1], z0],
            [b1[0], b1[1], z0],
            [b2[0], b2[1], z0],
            [b3[0], b3[1], z0],
            [b0[0], b0[1], z1],
            [b1[0], b1[1], z1],
            [b2[0], b2[1], z1],
            [b3[0], b3[1], z1],
        ], dtype=np.float32)

        # triangles (two per quad face)
        # side faces:
        f = [
            [0, 1, 5], [0, 5, 4],  # +o side
            [1, 2, 6], [1, 6, 5],  # end at p2
            [2, 3, 7], [2, 7, 6],  # -o side
            [3, 0, 4], [3, 4, 7],  # end at p1
        ]

        if cap:
            # bottom cap (z0) and top cap (z1)
            f += [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
            ]

        f = np.array(f, dtype=np.int32) + v_off

        verts_all.append(v)
        faces_all.append(f)
        v_off += v.shape[0]

    if not verts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    return verts, faces