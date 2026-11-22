"""Optional step H: explainability of the image and tabular models.

What the networks look at, what their representations encode, and whether the explanations
can be trusted. Everything here is descriptive; no result of step G is re-estimated.

1. Grade probes. Linear probes on the frozen features of the primary backbone
   (ResNet-50 fine-tuned on grades, ``pad``), fitted on the fine-tuning pool (never
   transferred) and evaluated on the transferred embryos (held out): accuracy, quadratic
   weighted kappa and confusion matrices against the expert grades.
2. Class activation maps (CAM). Features are the global average of the last convolutional
   block, so any linear read-out of the features (probe or outcome model) decomposes
   *exactly* over the 7x7 spatial grid: CAM = sum_k w_k A_k. Maps are signed (blue: lowers
   the score, red: raises it) and are not rectified, so negative evidence is shown too.
   - grade maps: the ridge probe for each Gardner criterion (higher = better quality);
   - outcome maps: the image part of the outcome model (step-G design, primary
     configuration, fitted on all transfers), back-projected from principal components to
     features. Exploratory: the image part does not beat Gardner (step G).
3. Localisation. Share of the absolute CAM mass that falls on the embryo (circle found by a
   Hough transform, central disc as fallback) divided by the share of the image the embryo
   covers: 1 = no preference, >1 = attention concentrated on the embryo.
4. Sanity check (Adebayo et al., NeurIPS 2018). The last block is re-initialised at random
   and the maps are recomputed with the same read-out; a trustworthy explanation must change
   (low rank correlation between trained and randomised maps).
5. What the representation encodes. Spearman correlations between the first principal
   components and grades, photo quality (brightness, contrast, sharpness), day, age and the
   outcomes; t-SNE map of all images.
6. Tabular model. Standardised ridge coefficients for each outcome (live birth, heart
   activity, positive hCG), with nested choice of the penalty, side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from . import images as G
from . import model as M
from .figures import GRID, INK, MUTED, plt

CRITERIA = ("EXP", "ICM", "TE")
LEVELS = {"EXP": ["1", "2", "3", "4", "5"], "ICM": ["A", "B", "C"], "TE": ["A", "B", "C"]}
OUTCOME_LABEL = {"LB": "live birth", "HA": "heart activity", "SS": "positive hCG"}
OUTCOME_TITLE = {"LB": "Live birth", "HA": "Heart activity", "SS": "Positive hCG"}

# Diverging (blue <-> red through a neutral gray) and sequential (one blue hue) scales.
BLUE, RED, NEUTRAL = "#1c5cab", "#c23b3a", "#f0efec"
SEQ = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap

    div = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    return div, seq


# =========================================================================== networks / CAM
def load_resnet(weights, pretrained: bool):
    import torch

    net, _ = G._backbone("resnet50", pretrained)
    if weights is not None:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return net.eval()


def randomise_last_block(net, seed: int):
    """Re-initialise every layer of the last residual block (model-randomisation test)."""
    import torch

    torch.manual_seed(seed)
    for m in net.layer4.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    return net.eval()


def cams(net, images: list, W: np.ndarray, batch: int = 32) -> np.ndarray:
    """Signed class activation maps (n, h, 7, 7) for read-out weights W (h, 2048).

    The features of the backbone are mean_{ij} A[:, i, j] (global average pooling), so for a
    linear read-out s = W f the map M[h, i, j] = sum_k W[h, k] A[k, i, j] averages exactly to
    the score (up to the intercept)."""
    import torch

    dev = G._device()
    net.to(dev)
    store = {}
    hook = net.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([G._to_tensor(im) for im in images[i:i + batch]]).to(dev)
                net(x)
                out.append(torch.einsum("hk,nkij->nhij", Wt, store["a"].float()).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(out) if out else np.empty((0, len(W), 7, 7))


def upsample(cam: np.ndarray, size: int = G.IMG_SIZE) -> np.ndarray:
    return zoom(cam, size / cam.shape[-1], order=1)


def embryo_mask(im) -> tuple[np.ndarray, bool]:
    """Boolean mask of the embryo on a padded 224-px image (Hough circle; central disc if
    no plausible circle is found). Returns (mask, found)."""
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    n = G.IMG_SIZE
    g = np.asarray(im.convert("L"), dtype=float) / 255.0
    edges = canny(g, sigma=2.0)
    radii = np.arange(45, 111, 5)
    acc = hough_circle(edges, radii)
    _, cx, cy, r = hough_circle_peaks(acc, radii, total_num_peaks=1)
    yy, xx = np.mgrid[:n, :n]
    if len(r) and 0.25 * n < cx[0] < 0.75 * n and 0.25 * n < cy[0] < 0.75 * n:
        return (xx - cx[0]) ** 2 + (yy - cy[0]) ** 2 <= r[0] ** 2, True
    return (xx - n / 2) ** 2 + (yy - n / 2) ** 2 <= 70 ** 2, False


def enrichment(cam_up: np.ndarray, mask: np.ndarray) -> float:
    """Share of |CAM| on the embryo divided by the share of the image it covers."""
    a = np.abs(cam_up)
    tot = a.sum()
    return float((a[mask].sum() / tot) / mask.mean()) if tot > 0 else np.nan


# =========================================================================== probes
def _quality(c: str, v: pd.Series) -> pd.Series:
    """Higher = better: EXP code 0-4 as is; ICM/TE code 0=A..2=C reversed (ND excluded)."""
    return v.astype(float) if c == "EXP" else 2.0 - v.astype(float)


def grade_probes(feats: pd.DataFrame, grades: pd.DataFrame, pool: list, held_out: list,
                 seed: int) -> tuple[pd.DataFrame, dict, dict]:
    """Linear probes of the expert grades on frozen features.

    Returns (metrics table, confusion matrices on held-out images, ridge read-out weights on
    the raw feature scale, one vector per criterion, higher = better quality)."""
    g = grades.set_index(["patient", "embryo"])
    mu = feats.loc[[k for k in pool if k in feats.index]].mean()
    sd = feats.loc[[k for k in pool if k in feats.index]].std(ddof=0).replace(0, 1.0)
    Z = (feats - mu) / sd
    rows, conf, W = [], {}, {}
    for c in CRITERIA:
        ok = g[c] <= (4 if c == "EXP" else 2)            # ND (code 3 for ICM/TE) excluded
        tr = [k for k in pool if k in Z.index and k in g.index and ok.get(k, False)]
        te = [k for k in held_out if k in Z.index and k in g.index and ok.get(k, False)]
        ytr, yte = g.loc[tr, c].astype(int).to_numpy(), g.loc[te, c].astype(int).to_numpy()
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=5, max_iter=3000,
                                   scoring="neg_log_loss", random_state=seed
                                   ).fit(Z.loc[tr].to_numpy(), ytr)
        pred = clf.predict(Z.loc[te].to_numpy())
        reg = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Z.loc[tr].to_numpy(),
                                                        _quality(c, g.loc[tr, c]).to_numpy())
        score = reg.predict(Z.loc[te].to_numpy())
        labels = list(range(len(LEVELS[c])))
        conf[c] = confusion_matrix(yte, pred, labels=labels)
        W[c] = reg.coef_ / sd.to_numpy()
        rows.append({"criterion": c, "train_images": len(tr), "held_out_images": len(te),
                     "accuracy": accuracy_score(yte, pred),
                     "majority_class_accuracy": float(np.mean(yte == np.bincount(ytr).argmax())),
                     "weighted_kappa": cohen_kappa_score(yte, pred, weights="quadratic"),
                     "spearman_quality_score": spearmanr(score, _quality(c, g.loc[te, c]))[0]})
    return pd.DataFrame(rows), conf, W


def outcome_readout(feats: pd.DataFrame, t_imp: pd.DataFrame, n_pcs: int, outcome: str,
                    n_knots: int, seed: int) -> tuple[np.ndarray, pd.Series]:
    """Image part of the step-G outcome model, as weights on the raw features.

    Same design as step G (patient covariates + first principal components, no grades),
    fitted on all transfers with a nested penalty. The coefficients of the components are
    back-projected through the PCA to one weight per feature. Returns (weights, image
    contribution to the log-odds for every image)."""
    mu, sd = feats.mean(), feats.std(ddof=0).replace(0, 1.0)
    Z = ((feats - mu) / sd).to_numpy()
    n = min(n_pcs, Z.shape[1], Z.shape[0] - 1)
    pca = PCA(n_components=n, random_state=seed).fit(Z)
    cols = [f"pc{i}" for i in range(n)]
    pcs = pd.DataFrame(pca.transform(Z), index=feats.index, columns=cols)
    tp = t_imp.merge(pcs, left_on=["patient", "embryo"], right_index=True, how="inner")
    y = tp[outcome].to_numpy()
    m = G.ImageModel(cols, n_knots=n_knots)
    m.C = m.inner_C(tp, y, seed)
    m.fit(tp, y)
    X = m._design(tp)
    coef = pd.Series(m.est_.coef_[0], index=X.columns)
    beta = (coef[cols] / m.sd_[cols]).to_numpy()
    w = (pca.components_.T @ beta) / sd.to_numpy()
    contrib = pd.Series(pcs.to_numpy() @ beta, index=pcs.index)
    return w, contrib


# =========================================================================== representation
def pc_correlations(pcs: pd.DataFrame, grades: pd.DataFrame, qc: pd.DataFrame,
                    t: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each principal component with interpretable variables."""
    g = grades.set_index(["patient", "embryo"])
    q = qc.set_index(["patient", "embryo"])
    tt = t.set_index(["patient", "embryo"])
    var = {}
    for c in CRITERIA:
        ok = g[c] <= (4 if c == "EXP" else 2)
        var[f"{c} (quality)"] = _quality(c, g.loc[ok, c])
    for c in ("brightness", "contrast", "sharpness"):
        var[f"photo {c}"] = q[c]
    var["day 5 (vs 4)"] = (tt["day"] == 5).astype(float)
    var["age"] = tt["age"].astype(float)
    for oc in ("SS", "HA", "LB"):
        var[OUTCOME_LABEL[oc]] = tt[oc].astype(float)
    out = {}
    for name, v in var.items():
        common = pcs.index.intersection(v.dropna().index)
        out[name] = [spearmanr(pcs.loc[common, c], v.loc[common])[0] for c in pcs.columns]
    return pd.DataFrame(out, index=[f"PC{i + 1}" for i in range(pcs.shape[1])]).T


def tabular_coefficients(t: pd.DataFrame, outcomes, n_knots: int, seed: int) -> pd.DataFrame:
    """Standardised ridge coefficients of the step-B model for each outcome."""
    t_imp = M.impute_patients(t, seed)
    cols = {}
    for oc in outcomes:
        C = M.choose_C_inner(t_imp, "full", n_knots, seed, oc)
        m = M.EmbryoModel(C=C, n_knots=n_knots).fit(t_imp, t_imp[oc].to_numpy())
        cols[OUTCOME_LABEL[oc]] = pd.Series(
            m.est_.coef_[0], index=M.design(t_imp, "full", m.knots_).columns)
    return pd.DataFrame(cols)


# =========================================================================== figures
def _heat(ax, df: pd.DataFrame, cmap, vmin, vmax, fmt="{:+.2f}", title=""):
    im = ax.imshow(df.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=0)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mid = (vmin + vmax) / 2
    span = (vmax - vmin) / 2
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if pd.notna(v):
                strong = abs(v - mid) > 0.6 * span if vmin < 0 else v > vmin + 1.2 * span
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if strong else INK)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")
    return im


def figure_probes(metrics: pd.DataFrame, conf: dict, path) -> None:
    _, seq = _cmaps()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [5, 3, 3]})
    for ax, c in zip(axes, CRITERIA, strict=True):
        cm = conf[c].astype(float)
        share = cm / np.where(cm.sum(1, keepdims=True) == 0, 1, cm.sum(1, keepdims=True))
        df = pd.DataFrame(share * 100, index=LEVELS[c], columns=LEVELS[c])
        _heat(ax, df, seq, 0, 100, "{:.0f}")
        r = metrics.set_index("criterion").loc[c]
        ax.set_title(f"{c}: accuracy {r['accuracy']:.2f}, weighted kappa "
                     f"{r['weighted_kappa']:.2f}", loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Predicted from image")
        ax.set_ylabel("Expert grade")
    fig.suptitle("The image representation recovers the expert grades of held-out embryos "
                 "(row %)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_cams(images: list, maps: np.ndarray, row_titles: list, col_titles: list,
                path, title: str) -> None:
    """Grid of signed CAM overlays: rows = read-outs, columns = images."""
    div, _ = _cmaps()
    nr, nc = maps.shape[1], len(images)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.65 * nr + 0.6), squeeze=False)
    for j, im in enumerate(images):
        g = np.asarray(im.convert("L"), dtype=float)
        for i in range(nr):
            m = upsample(maps[j, i])
            lim = np.percentile(np.abs(m), 99) or 1.0
            ax = axes[i, j]
            ax.imshow(g, cmap="gray")
            ax.imshow(m, cmap=div, vmin=-lim, vmax=lim, alpha=0.55)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=6.5, color=MUTED)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=7.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Red: raises the score; blue: lowers it; scale per image (99th "
             "percentile of |map|).", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def figure_representation(corr: pd.DataFrame, emb2d: pd.DataFrame, path) -> None:
    div, _ = _cmaps()
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes((0.08, 0.12, 0.42, 0.74))
    _heat(ax, corr.round(2), div, -1, 1, "{:+.2f}",
          "A. What the image components encode (Spearman rho)")
    ax.axhline(5.5, color="white", lw=2)
    ax.axhline(7.5, color="white", lw=2)
    panels = [("EXP", "B. t-SNE coloured by expansion"),
              ("outcome", "C. Transferred embryos by hCG result")]
    for k, (col, ttl) in enumerate(panels):
        a = fig.add_axes((0.57 + k * 0.215, 0.12, 0.19, 0.74))
        a.scatter(emb2d["x"], emb2d["y"], s=3, color=GRID, lw=0)
        if col == "EXP":
            for lv in range(5):
                d = emb2d[emb2d["EXP"] == lv]
                a.scatter(d["x"], d["y"], s=4, lw=0, color=SEQ[min(lv + 1, 5)],
                          label=LEVELS["EXP"][lv])
            a.legend(title="Expansion grade", fontsize=6, title_fontsize=6.5,
                     frameon=False, markerscale=2.5, loc="upper center", ncol=5,
                     handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.5, 0.0))
        else:
            for v, lab, colr in ((0, "hCG negative", CAT[0]), (1, "hCG positive", CAT[1])):
                d = emb2d[emb2d["SS"] == v]
                a.scatter(d["x"], d["y"], s=6, lw=0, color=colr, label=lab)
            a.legend(fontsize=6, frameon=False, markerscale=2, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, 0.0))
        a.set_title(ttl, loc="left", fontsize=8.5, color=INK, fontweight="bold")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def figure_coefficients(coefs: pd.DataFrame, path) -> None:
    div, _ = _cmaps()
    lim = float(np.nanmax(np.abs(coefs.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(4.6, 0.28 * len(coefs) + 1.2))
    _heat(ax, coefs, div, -lim, lim, "{:+.2f}",
          "Standardised ridge coefficients by outcome")
    ax.xaxis.tick_top()
    fig.text(0.01, -0.01, "Red: higher probability; blue: lower. Penalty chosen by nested "
             "cross-validation for each outcome (coefficients are shrunk).", fontsize=6.5,
             color=MUTED, va="top")
    fig.savefig(path)
    plt.close(fig)


def figure_cascade(sens: pd.DataFrame, img: pd.DataFrame | None, path) -> None:
    """E1 and image-vs-Gardner discrimination along the outcome cascade."""
    order = [("S8b", "SS"), ("S8", "HA"), ("S0", "LB")]
    s = sens.set_index("code")
    fig, axes = plt.subplots(1, 2 if img is not None else 1, figsize=(9.5, 3.0),
                             squeeze=False)
    a = axes[0, 0]
    for i, (code, _oc) in enumerate(order):
        if code not in s.index:
            continue
        lo, hi = (float(x) for x in s.loc[code, "E1_ci"].strip("[]").split(","))
        est = s.loc[code, "E1"]
        a.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=INK, capsize=3,
                   ms=5)
        a.text(hi + 0.3, i, f"{est:+.1f} [{lo:.1f}, {hi:.1f}]", va="center", fontsize=7.5,
               color=INK)
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_yticks(range(3), [OUTCOME_TITLE[o] for _, o in order])
    a.set_ylim(2.5, -0.5)
    a.set_xlabel("Best by Gardner minus random embryo (points)")
    a.set_title("A. Value of morphological selection", loc="left", fontsize=9, color=INK,
                fontweight="bold")
    a.set_xlim(right=a.get_xlim()[1] + 3)
    if img is not None:
        b = axes[0, 1]
        d = img[(img["group"] == "choice")]
        ref = d[d["config"] == "gardner_reference"].set_index("outcome")
        pri = d[d["config"].str.contains("PRIMARY")].set_index("outcome")
        series = (("Gardner + clinical", ref, CAT[0], -0.12),
                  ("Image (primary) + clinical", pri, CAT[1], 0.12))
        for lab, src, colr, off in series:
            for i, (_, oc) in enumerate(order):
                if oc not in src.index:
                    continue
                lo, hi = (float(x) for x in src.loc[oc, "auroc_ci"].strip("[]").split(","))
                v = src.loc[oc, "auroc"]
                b.errorbar(v, i + off, xerr=[[v - lo], [hi - v]], fmt="o", color=colr,
                           capsize=3, ms=5, label=lab if i == 0 else None)
        b.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        b.set_yticks(range(3), [OUTCOME_TITLE[o] for _, o in order])
        b.set_ylim(2.5, -0.5)
        b.set_xlabel("AUROC, patients with a choice (95% CI)")
        b.set_title("B. Discrimination among embryos judged transferable", loc="left",
                    fontsize=9, color=INK, fontweight="bold")
        b.legend(frameon=False, fontsize=7, loc="upper center", ncol=2,
                 bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
