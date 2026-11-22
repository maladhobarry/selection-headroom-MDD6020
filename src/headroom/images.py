"""Optional step G: can blastocyst images separate embryos that Gardner grades cannot?

Design (fixed in advance; every configuration of the grid is reported, none is selected):

1. Quality control of all images: size, mode, brightness, contrast, sharpness (variance of
   the Laplacian), exact duplicates, and a contact sheet for visual inspection.
2. Pre-processing: ``pad`` (resize the longer side to 224 px and pad to a square, so the whole
   embryo is kept; no cropping) or ``clahe`` (contrast-limited adaptive histogram
   equalisation, then ``pad``).
3. Feature extractors (frozen at outcome-modelling time):
   - ``resnet50_imagenet``: ResNet-50 pre-trained on ImageNet,
   - ``dinov2_vits14``: self-supervised DINOv2 ViT-S/14 (general-purpose visual features),
   - ``resnet50_gardner``: ResNet-50 fine-tuned to predict the three Gardner grades on images
     of embryos that were NOT transferred (labels = grades, never live birth, so no outcome
     leakage); its penultimate layer gives embryo-specific features.
4. Outcome model: ridge logistic regression on the first principal components of the image
   features (10, 20 or 50; PCA is unsupervised) + the patient covariates of step B, without
   the grades. Leak-free cross-fitting by patient (fold-wise imputation, nested penalty),
   repeated over several fold splits and averaged.
5. Metrics, overall and separately for patients with / without a choice: AUROC (bootstrap CI),
   calibration intercept and slope, Brier score, and the paired AUROC difference with the
   Gardner model evaluated on the same folds.
6. Shortcut test (primary configuration): can image features tell a transferred embryo from a
   vitrified sibling at equal grades? (trace of photo timing, or cues seen by the embryologist).

Primary configuration (pre-specified): resnet50_gardner + pad + 20 components.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import model as M

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# =========================================================================== inventory / QC
def find_images(raw: Path) -> dict[tuple[int, int], Path]:
    from .cleaning import parse_image_key

    out = {}
    for p in sorted(raw.rglob("*.png")):
        try:
            out.setdefault(parse_image_key(p.name), p)
        except ValueError:
            continue
    return out


def _laplacian_var(g: np.ndarray) -> float:
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def image_qc(paths_by_key: dict, out_csv: Path, sheet_png: Path, seed: int) -> pd.DataFrame:
    """Per-image quality metrics + flags, and a 6x5 contact sheet of random images."""
    from PIL import Image

    rows = []
    for (pt, em), p in paths_by_key.items():
        raw = p.read_bytes()
        im = Image.open(p)
        g = np.asarray(im.convert("L"), dtype=float) / 255.0
        rows.append({"patient": pt, "embryo": em, "file": p.name, "width": im.width,
                     "height": im.height, "mode": im.mode, "brightness": g.mean(),
                     "contrast": g.std(), "sharpness": _laplacian_var(g),
                     "md5": hashlib.md5(raw).hexdigest()})
    qc = pd.DataFrame(rows)
    qc["duplicate_of_other_image"] = qc.duplicated("md5", keep=False)
    for c in ("brightness", "contrast", "sharpness"):
        lo, hi = qc[c].quantile([0.01, 0.99])
        qc[f"flag_{c}"] = (qc[c] < lo) | (qc[c] > hi)
    qc["flag_any"] = qc.filter(like="flag_").any(axis=1) | qc["duplicate_of_other_image"]
    qc.to_csv(out_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(qc), size=min(30, len(qc)), replace=False)
    fig, axes = plt.subplots(5, 6, figsize=(9, 7.8))
    for ax, i in zip(axes.flat, pick, strict=False):
        r = qc.iloc[i]
        ax.imshow(Image.open(paths_by_key[(r.patient, r.embryo)]).convert("L"), cmap="gray")
        ax.set_title(f"{r.file}  {r.width}x{r.height}", fontsize=5)
        ax.axis("off")
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(sheet_png, dpi=150)
    plt.close(fig)
    return qc


def qc_summary(qc: pd.DataFrame) -> pd.DataFrame:
    sizes = qc.groupby(["width", "height"]).size().sort_values(ascending=False)
    rows = [
        {"check": "images", "value": len(qc)},
        {"check": "distinct sizes (top 3)", "value": "; ".join(
            f"{w}x{h}: {n}" for (w, h), n in sizes.head(3).items())},
        {"check": "colour modes", "value": "; ".join(
            f"{k}: {v}" for k, v in qc["mode"].value_counts().items())},
        {"check": "exact duplicate files", "value": int(qc["duplicate_of_other_image"].sum())},
        {"check": "brightness median (IQR)", "value": _iqr(qc["brightness"])},
        {"check": "contrast median (IQR)", "value": _iqr(qc["contrast"])},
        {"check": "sharpness median (IQR)", "value": _iqr(qc["sharpness"], 5)},
        {"check": "flagged (1st/99th percentile or duplicate)", "value": int(qc["flag_any"].sum())},
    ]
    return pd.DataFrame(rows)


def _iqr(x, d=3):
    q = x.quantile([0.25, 0.5, 0.75])
    return f"{q[0.5]:.{d}f} ({q[0.25]:.{d}f}-{q[0.75]:.{d}f})"


def finetune_pool(found: dict, transferred: set, qc: pd.DataFrame) -> tuple[list, str]:
    """Images used to fine-tune on grades: never transferred, no byte-identical copy of a
    transferred image (e.g. 858_01 == 835_01), one copy per file. Returns the keys and a
    short fingerprint used to name the cached weights and features."""
    md5 = {(int(r.patient), int(r.embryo)): r.md5 for r in qc.itertuples()}
    md5_transferred = {md5[k] for k in transferred if k in md5}
    pool, seen = [], set()
    for k in sorted(found):
        h = md5.get(k)
        if k in transferred or h in md5_transferred or h in seen:
            continue
        seen.add(h)
        pool.append(k)
    tag = hashlib.md5(",".join(f"{a}_{b}" for a, b in pool).encode()).hexdigest()[:8]
    return pool, tag


# =========================================================================== pre-processing
def load_image(path: Path, preprocess: str):
    """RGB PIL image, whole embryo kept: CLAHE (optional), resize longer side, pad square."""
    from PIL import Image

    im = Image.open(path).convert("L")
    if preprocess == "clahe":
        from skimage import exposure

        a = exposure.equalize_adapthist(np.asarray(im, dtype=float) / 255.0, clip_limit=0.01)
        im = Image.fromarray((a * 255).astype(np.uint8))
    elif preprocess != "pad":
        raise ValueError(preprocess)
    im.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), int(np.median(np.asarray(im))))
    canvas.paste(im, ((IMG_SIZE - im.width) // 2, (IMG_SIZE - im.height) // 2))
    return canvas.convert("RGB")


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_tensor(im, augment: bool = False, rng: np.random.Generator | None = None):
    import torch

    a = np.asarray(im, dtype=np.float32) / 255.0
    if augment:
        # Embryos have no orientation: random flips and 90-degree rotations.
        if rng.random() < 0.5:
            a = a[:, ::-1]
        if rng.random() < 0.5:
            a = a[::-1, :]
        a = np.rot90(a, int(rng.integers(0, 4)))
    a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


# =========================================================================== backbones
def _backbone(name: str, pretrained: bool):
    import torch
    from torchvision import models

    if name.startswith("resnet50"):
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=w)
        net.fc = torch.nn.Identity()
        return net, 2048
    if name == "dinov2_vits14":
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        return net, 384
    raise ValueError(name)


def finetune_on_grades(paths_by_key: dict, grades: pd.DataFrame, train_keys: list,
                       preprocess: str, cfg: dict, out_weights: Path, pretrained: bool,
                       log=None) -> dict:
    """Fine-tune ResNet-50 to predict EXP/ICM/TE on embryos that were NOT transferred.

    Only layer4 and the three heads are trained (stable with ~1,600 images). Returns the
    validation accuracy per criterion (10% of the training patients held out).
    """
    import torch

    seed = cfg["seed"]
    ft = cfg["images"]["finetune"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = grades.set_index(["patient", "embryo"])
    keys = [k for k in train_keys if k in paths_by_key and k in g.index]
    pts = np.array(sorted({k[0] for k in keys}))
    val_pts = set(rng.choice(pts, max(1, int(len(pts) * ft["val_fraction"])), replace=False))
    tr = [k for k in keys if k[0] not in val_pts]
    va = [k for k in keys if k[0] in val_pts]

    net, dim = _backbone("resnet50", pretrained)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.layer4.parameters():
        p.requires_grad = True
    heads = torch.nn.ModuleDict({c: torch.nn.Linear(dim, k)
                                 for c, k in (("EXP", 5), ("ICM", 4), ("TE", 4))})
    dev = _device()
    net.to(dev)
    heads.to(dev)
    params = [p for p in net.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=ft["lr"], weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    cache = {k: load_image(paths_by_key[k], preprocess) for k in keys}

    def batches(ks, bs, aug):
        order = rng.permutation(len(ks)) if aug else np.arange(len(ks))
        for i in range(0, len(ks), bs):
            b = [ks[j] for j in order[i:i + bs]]
            x = torch.stack([_to_tensor(cache[k], aug, rng) for k in b]).to(dev)
            y = {c: torch.tensor([int(g.loc[k, c]) for k in b]).to(dev) for c in heads}
            yield x, y

    best, best_state = -1.0, None
    for ep in range(ft["epochs"]):
        net.train()
        heads.train()
        for x, y in batches(tr, ft["batch_size"], True):
            f = net(x)
            loss = sum(loss_fn(heads[c](f), y[c]) for c in heads)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        heads.eval()
        correct = {c: 0 for c in heads}
        with torch.no_grad():
            for x, y in batches(va, ft["batch_size"], False):
                f = net(x)
                for c in heads:
                    correct[c] += int((heads[c](f).argmax(1) == y[c]).sum())
        acc = {c: correct[c] / max(len(va), 1) for c in heads}
        score = float(np.mean(list(acc.values())))
        if log:
            log.info("Fine-tune %s epoch %d: val accuracy %s", preprocess, ep + 1,
                     {c: round(v, 3) for c, v in acc.items()})
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_acc = acc
    torch.save(best_state, out_weights)
    return {"train_images": len(tr), "val_images": len(va),
            **{f"val_acc_{c}": v for c, v in best_acc.items()}}


def extract_features(paths_by_key: dict, backbone: str, preprocess: str, cache: Path,
                     pretrained: bool, weights: Path | None = None,
                     batch_size: int = 32) -> pd.DataFrame:
    """Frozen features for every image (cached)."""
    if cache.exists():
        z = np.load(cache)
        return pd.DataFrame(z["X"], index=pd.MultiIndex.from_arrays(
            [z["patient"], z["embryo"]], names=["patient", "embryo"]))
    import torch

    torch.manual_seed(0)
    net, _ = _backbone(backbone, pretrained)
    if weights is not None:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    dev = _device()
    if backbone.startswith("dinov2") and dev == "mps":
        dev = "cpu"   # DINOv2 uses bicubic upsampling, not implemented on Apple MPS
    net.to(dev).eval()
    keys = sorted(paths_by_key)
    feats = []
    with torch.no_grad():
        for i in range(0, len(keys), batch_size):
            x = torch.stack([_to_tensor(load_image(paths_by_key[k], preprocess))
                             for k in keys[i:i + batch_size]]).to(dev)
            feats.append(net(x).float().cpu().numpy())
    X = np.vstack(feats)
    np.savez_compressed(cache, X=X, patient=[k[0] for k in keys], embryo=[k[1] for k in keys])
    return pd.DataFrame(X, index=pd.MultiIndex.from_tuples(keys, names=["patient", "embryo"]))


def principal_components(feats: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Unsupervised PCA on all images (no outcome used, so no label leakage)."""
    Z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    n = min(n, Z.shape[1], Z.shape[0] - 1)
    pcs = PCA(n_components=n, random_state=seed).fit_transform(Z.to_numpy())
    return pd.DataFrame(pcs, index=feats.index, columns=[f"pc{i}" for i in range(n)])


# =========================================================================== outcome model
class ImageModel(M.EmbryoModel):
    """Step-B patient covariates + image principal components (grades optional)."""

    def __init__(self, pc_cols: list[str], with_grades: bool = False, **kw):
        super().__init__(**kw)
        self.pc_cols = pc_cols
        self.with_grades = with_grades

    def _design(self, df):
        X = M.design(df, self.feature_set, self.knots_)
        if not self.with_grades:
            X = X[[c for c in X.columns if not c.startswith(("EXP", "ICM", "TE", "ND"))]]
        return pd.concat([X, df[self.pc_cols]], axis=1) if self.pc_cols else X

    def fit(self, df, y):
        self.knots_ = M.rcs_knots(df["age"], self.n_knots)
        X = self._design(df)
        self.mean_, self.sd_ = X.mean(), X.std(ddof=0).replace(0, 1.0)
        self.est_ = LogisticRegression(C=self.C, max_iter=5000).fit(
            ((X - self.mean_) / self.sd_).to_numpy(), y)
        return self

    def predict(self, df):
        X = self._design(df)
        return self.est_.predict_proba(((X - self.mean_) / self.sd_).to_numpy())[:, 1]

    def inner_C(self, df, y, seed) -> float:
        self.knots_ = M.rcs_knots(df["age"], self.n_knots)
        X = self._design(df)
        Xs = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
        m = LogisticRegressionCV(Cs=M.INNER_CS, cv=StratifiedKFold(3, shuffle=True,
                                                                   random_state=seed),
                                 scoring="neg_log_loss", max_iter=5000).fit(Xs, y)
        return float(m.C_[0])


def crossfit_repeated(t: pd.DataFrame, make, n_repeats: int, k: int, seed: int,
                      outcome: str = "LB") -> np.ndarray:
    """Leak-free cross-fitted predictions (fold-wise imputation + nested penalty), averaged
    over ``n_repeats`` different fold splits."""
    y = t[outcome].to_numpy()
    out = np.zeros(len(t))
    for r in range(n_repeats):
        folds = M.patient_folds(t["patient"].to_numpy(), y, k, seed + 1000 * r)
        for f in range(k):
            tr, te = M.impute_fold(t[folds != f], t[folds == f], seed + 1000 * r + f, outcome)
            c = make(1.0).inner_C(tr, y[folds != f], seed + f)
            out[folds == f] += make(c).fit(tr, y[folds != f]).predict(te) / n_repeats
    return out


def metrics(y, p, n_boot, seed, ref=None) -> dict:
    """AUROC (+CI), calibration, Brier; paired AUROC difference vs ``ref`` predictions."""
    rng = np.random.default_rng(seed)
    a, b = M.calibration(y, p)
    res = {"auroc": roc_auc_score(y, p), "cal_intercept": a, "cal_slope": b,
           "brier": float(np.mean((p - y) ** 2))}
    boots, diffs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], p[i]))
        if ref is not None:
            diffs.append(boots[-1] - roc_auc_score(y[i], ref[i]))
    res["auroc_ci"] = f"[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
    if ref is not None:
        res["delta_auroc_vs_gardner"] = res["auroc"] - roc_auc_score(y, ref)
        res["delta_ci"] = f"[{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}]"
    return res


def shortcut_test(emb: pd.DataFrame, pc_cols: list[str], seed: int) -> dict:
    """Cross-validated AUROC for 'transferred vs sibling' from grades vs grades + images."""
    d = emb.dropna(subset=pc_cols)
    y = d["transferred"].to_numpy()
    grades = pd.get_dummies(d[["EXP", "ICM", "TE"]].astype(str)).astype(float)
    out = {}
    for name, X in (("grades", grades),
                    ("grades+images", pd.concat([grades, d[pc_cols]], axis=1))):
        Xs = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
        p = np.zeros(len(d))
        for tr, te in GroupKFold(10).split(Xs, y, d["patient"]):
            p[te] = LogisticRegression(C=0.1, max_iter=5000).fit(Xs[tr], y[tr]).predict_proba(
                Xs[te])[:, 1]
        out[f"auroc_{name}"] = float(roc_auc_score(y, p))
    out["excess_auroc_from_images"] = out["auroc_grades+images"] - out["auroc_grades"]
    return out
