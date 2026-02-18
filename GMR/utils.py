import numpy as np
from numba import njit, prange
import numba
numba.set_num_threads(8)

# ---------------------------------------
# History-conditioned demo selection
# ---------------------------------------
@njit(cache=True, fastmath=True, parallel=True)
def score_and_reps(pos_all, history_pts, w_min=0.05, w_max=1.0):
    """
    pos_all: (T,N,3) float32/float64 contiguous
    history_pts: (L,3)
    Returns:
      score2: (N,) weighted sum of squared min distances (min over time) for each history point
      reps:   (N,3) representative points closest (over time) to latest history point
    """
    T, N, _ = pos_all.shape
    L = history_pts.shape[0]

    # weights (linear ramp) + normalize
    # compute sum of linspace analytically to avoid allocating w
    # w_ell = w_min + (w_max-w_min)*ell/(L-1)
    # handle L==1 safely
    if L <= 1:
        wsum = 1.0
    else:
        # sum of arithmetic sequence
        wsum = 0.5 * L * (w_min + w_max)

    score2 = np.empty(N, dtype=pos_all.dtype)
    reps = np.empty((N, 3), dtype=pos_all.dtype)

    for n in prange(N):
        total = 0.0

        # also track reps for latest point inside same loop
        best_latest_d2 = 1e30
        best_latest_t = 0

        for ell in range(L):
            hx = history_pts[ell, 0]
            hy = history_pts[ell, 1]
            hz = history_pts[ell, 2]

            # min over time for this (n, ell)
            best_d2 = 1e30
            for t in range(T):
                dx = pos_all[t, n, 0] - hx
                dy = pos_all[t, n, 1] - hy
                dz = pos_all[t, n, 2] - hz
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < best_d2:
                    best_d2 = d2

                # if this ell is latest, also track best t for reps
                if ell == L - 1:
                    if d2 < best_latest_d2:
                        best_latest_d2 = d2
                        best_latest_t = t

            # weight for ell
            if L <= 1:
                w = 1.0
            else:
                w = w_min + (w_max - w_min) * (ell / (L - 1))
            w = w / (wsum + 1e-12)

            total += w * best_d2

        score2[n] = total

        # reps from best_latest_t
        reps[n, 0] = pos_all[best_latest_t, n, 0]
        reps[n, 1] = pos_all[best_latest_t, n, 1]
        reps[n, 2] = pos_all[best_latest_t, n, 2]

    return score2, reps

@njit(cache=True, fastmath=True)
def nms_by_rep_points(score, reps, k, min_rep_dist):
    N = score.shape[0]
    order = np.argsort(score)
    suppressed = np.zeros(N, dtype=np.uint8)
    selected = np.empty(k, dtype=np.int32)
    nsel = 0
    thr2 = min_rep_dist * min_rep_dist

    for oi in range(N):
        j = order[oi]
        if suppressed[j] == 1:
            continue

        selected[nsel] = j
        nsel += 1
        if nsel >= k:
            break

        rjx, rjy, rjz = reps[j, 0], reps[j, 1], reps[j, 2]
        for i in range(N):
            dx = reps[i, 0] - rjx
            dy = reps[i, 1] - rjy
            dz = reps[i, 2] - rjz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < thr2:
                suppressed[i] = 1
        suppressed[j] = 0

    return selected[:nsel]

def select_demos_by_history(
    pos_all, history_pts, k=15,
    w_min=0.05, w_max=1.0,
    min_rep_dist=0.05,
):
    # use float32 unless you truly need float64
    pos_all = np.ascontiguousarray(pos_all, dtype=np.float32)
    history_pts = np.ascontiguousarray(history_pts, dtype=np.float32)

    score2, reps = score_and_reps(pos_all, history_pts, w_min=w_min, w_max=w_max)
    idx = nms_by_rep_points(score2, reps, k=k, min_rep_dist=np.float32(min_rep_dist))
    return idx, score2[idx]

def select_demos(boids_pos, history_pts, n_demos=15, time_stride=10, min_rep_dist=0.05):
    idx, _ = select_demos_by_history(
        boids_pos, history_pts, k=n_demos, min_rep_dist=min_rep_dist
    )

    pos_demos = []
    for i in idx:
        demo = boids_pos[:, i, :]
        valid = ((demo * demo).sum(axis=1) > 1e-4)  # (norm > 1e-2)^2
        demo = demo[valid, :][::time_stride, :]
        pos_demos.append(demo)
    return pos_demos

# ----------------------------
# Plot helpers
# ----------------------------
def gaussian_ellipsoid(mean, cov, n_std=1.5, n_points=30):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    radii = n_std * np.sqrt(np.maximum(eigvals, 0))

    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T
    ellipsoid += mean
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def refresh_wireframes(ax, wireframes, mu_y, Sigma_y, step=30, n_std=1.5, n_points=18, alpha=0.25):
    for wf in wireframes:
        try:
            wf.remove()
        except Exception:
            pass
    wireframes.clear()

    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std, n_points=n_points)
        wf = ax.plot_wireframe(X, Y, Z, alpha=alpha, linewidth=0.5)
        wireframes.append(wf)

def plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=1.5, ax=None):
    wireframes = []
    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std)
        wf = ax.plot_wireframe(X, Y, Z, alpha=0.1)
        wireframes.append(wf)
    return wireframes