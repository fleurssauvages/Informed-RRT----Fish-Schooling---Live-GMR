import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

def normalize_demos_list(pos_demos):
    pos_demos = np.asarray(pos_demos, float) if not isinstance(pos_demos, list) else pos_demos
    if isinstance(pos_demos, list):
        return [d if d.shape[0] >= d.shape[1] else d.T for d in pos_demos]

    if pos_demos.shape[1] < pos_demos.shape[2]:  # (N,T,D)
        return [pos_demos[i] for i in range(pos_demos.shape[0])]
    else:                                        # (N,D,T)
        return [pos_demos[i].T for i in range(pos_demos.shape[0])]

class GMRGMM:
    """
    GMM + GMR on X=[t, y] (t scalar in [0,1]).
    Provides:
      - fit(demos)
      - update(demos, n_iter)
      - regress(T, pos_dim) -> mu_y(T,D), Sigma_y(T,D,D), gamma(T,K), loglik
    """
    def __init__(self, n_components=6, seed=0, cov_type="full", reg_covar=1e-6, n_iter=200):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,  # "full" or "diag"
            random_state=seed,
            reg_covar=reg_covar,
            max_iter=n_iter,
            init_params="kmeans",
            warm_start=False,  # we handle warm-start manually in update
        )

    @property
    def n_components(self):
        return self.model.n_components

    def _pack_demos_with_time(self, demos):
        demos = normalize_demos_list(demos)
        lengths = np.fromiter((Y.shape[0] for Y in demos), dtype=np.int64)
        N = int(lengths.sum())
        Dp = demos[0].shape[1]

        X_all = np.empty((N, 1 + Dp), dtype=np.float64)
        start = 0
        for Y, T in zip(demos, lengths):
            X_all[start:start+T, 0] = np.linspace(0.0, 1.0, T)
            X_all[start:start+T, 1:] = Y
            start += T

        return X_all, lengths.tolist()

    def fit(self, pos_demos):
        X_all, _ = self._pack_demos_with_time(pos_demos)
        self.model.max_iter = self.model.max_iter  # keep default
        self.model.warm_start = False
        self.model.fit(X_all)

    def update(self, pos_demos, n_iter=10):
        """
        Warm-start update: reuse previous parameters to continue EM.
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        X_all, _ = self._pack_demos_with_time(pos_demos)
        self.model.warm_start = True
        self.model.max_iter = n_iter

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_all)

    # -------- core GMR --------

    def regress(self, T, pos_dim=3):
        """
        Returns:
          mu_y: (T, pos_dim)
          Sigma_y: (T, pos_dim, pos_dim)
          gamma: (T, K) responsibilities over components given t
          loglik_t: scalar loglik of t-grid under p(t) (optional diagnostic)
        """
        K = self.model.n_components
        D = 1 + pos_dim
        t_grid = np.linspace(0.0, 1.0, T)
        t = t_grid[:, None]  # (T,1)

        w = self.model.weights_  # (K,)
        mu = self.model.means_   # (K,D)

        if self.model.covariance_type == "full":
            cov = self.model.covariances_  # (K,D,D)
            # partitions
            mu_t = mu[:, 0]              # (K,)
            mu_y = mu[:, 1:]             # (K,pos_dim)
            Sig_tt = cov[:, 0, 0] + 1e-12                # (K,)
            Sig_ty = cov[:, 0, 1:]                       # (K,pos_dim)
            Sig_yt = cov[:, 1:, 0]                       # (K,pos_dim)
            Sig_yy = cov[:, 1:, 1:]                      # (K,pos_dim,pos_dim)

            # conditional mean for each component: mu_y|t = mu_y + Sig_yt/Sig_tt * (t - mu_t)
            # shape (T,K,pos_dim)
            gain = Sig_yt / Sig_tt[:, None]              # (K,pos_dim)
            mu_y_t = mu_y[None, :, :] + (t - mu_t[None, :])[:, :, None] * gain[None, :, :]

            # conditional cov: Sig_yy|t = Sig_yy - Sig_yt Sig_ty / Sig_tt
            # (K,pos_dim,pos_dim)
            cond_cov = Sig_yy - (Sig_yt[:, :, None] * Sig_ty[:, None, :]) / Sig_tt[:, None, None]

        elif self.model.covariance_type == "diag":
            # covariances_: (K,D)
            var = self.model.covariances_ + 1e-12  # (K,D)
            mu_t = mu[:, 0]        # (K,)
            mu_y = mu[:, 1:]       # (K,pos_dim)
            Sig_tt = var[:, 0]     # (K,)
            # diag -> cross-cov = 0, so conditional mean is just mu_y
            mu_y_t = np.repeat(mu_y[None, :, :], T, axis=0)  # (T,K,pos_dim)
            # conditional cov is diag(var_y)
            diag_y = var[:, 1:]  # (K,pos_dim)
            I = np.eye(pos_dim)[None, :, :]                 # (1,pos_dim,pos_dim)
            cond_cov = diag_y[:, :, None] * I               # (K,pos_dim,pos_dim)

        else:
            raise ValueError("Use cov_type='full' or 'diag'.")

        # responsibilities given t: gamma(t,k) ∝ w_k * N(t | mu_tk, Sig_tt_k)
        logw = np.log(w + 1e-12)[None, :]                # (1,K)
        logNt = -0.5 * (np.log(2*np.pi*Sig_tt)[None, :] + ((t - mu_t[None, :])**2) / Sig_tt[None, :])
        logp = logw + logNt                               # (T,K)
        logZ = logsumexp(logp, axis=1, keepdims=True)
        gamma = np.exp(logp - logZ)                       # (T,K)
        loglik_t = float(np.sum(logZ))                    # diagnostic

        # mixture mean: mu(t)= sum_k gamma * mu_y|t
        mu_mix = np.einsum("tk,tkd->td", gamma, mu_y_t)    # (T,pos_dim)

        # mixture covariance: sum_k gamma [cond_cov_k + (mu_k - mu)(mu_k - mu)^T]
        # within
        Sigma_within = np.einsum("tk,kij->tij", gamma, cond_cov)  # (T,D,D)
        # between
        diff = mu_y_t - mu_mix[:, None, :]                           # (T,K,pos_dim)
        Sigma_between = np.einsum("tk,tk i,tk j->tij", gamma, diff, diff)

        Sigma_mix = Sigma_within + Sigma_between
        return mu_mix, Sigma_mix, gamma, loglik_t