"""Tables and figures of the article (written to results/)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, ACCENT, GRID = "#1F2A33", "#6F7B86", "#2E6B8A", "#DCE3E7"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

LABELS = {"worst": "Worst embryo", "random": "Random embryo",
          "observed": "Embryologist's choice", "gardner": "Best by Gardner",
          "model_best": "Best by model (upper bound)"}


def table1(transfers: pd.DataFrame) -> pd.DataFrame:
    t = transfers[transfers["day"] == 5].copy()
    t["Blastocysts available"] = pd.cut(t["n_blast"], [0, 1, 2, 3, 99],
                                        labels=["1", "2", "3", "4+"])

    def iqr(x):
        q = x.quantile([0.25, 0.5, 0.75])
        return f"{q[0.5]:.1f} ({q[0.25]:.1f}-{q[0.75]:.1f})"

    rows = []
    for lvl, d in [*t.groupby("Blastocysts available", observed=True), ("All", t)]:
        rows.append({
            "Blastocysts available": lvl, "Patients": len(d),
            "Age, mean (SD)": f"{d['age'].mean():.1f} ({d['age'].std():.1f})",
            "AMH ng/mL, median (IQR)": iqr(d["AMH"].dropna()),
            "AMH missing, n": int(d["AMH"].isna().sum()),
            "MII oocytes, median (IQR)": iqr(d["MII"]),
            "Endometrium mm, median (IQR)": iqr(d["Endo"].dropna()),
            "hCG positive, %": f"{d['SS'].mean() * 100:.1f}",
            "Heart activity, %": f"{d['HA'].mean() * 100:.1f}",
            "Live birth, n (%)": f"{int(d['LB'].sum())} ({d['LB'].mean() * 100:.1f})",
        })
    return pd.DataFrame(rows)


def figure1(summary: pd.DataFrame, path) -> None:
    """Expected live birth (%) by selection strategy, with 95% CI and E1-E3 brackets."""
    s = summary.set_index("quantity")
    order = ["worst", "random", "observed", "gardner", "model_best"]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, k in enumerate(order):
        r = s.loc[k]
        col = ACCENT if k in ("gardner", "observed") else INK
        ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["ci_low"]],
                                           [r["ci_high"] - r["estimate"]]],
                    fmt="o", color=col, ms=5, capsize=2, lw=1.2)
        ax.text(r["ci_high"] + 0.4, i, f"{r['estimate']:.1f}%", va="center", color=col)
    ax.set_yticks(y, [LABELS[k] for k in order])
    ax.set_xlabel("Expected live birth at first transfer (%)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    e = {k: s.loc[k] for k in ("E1", "E2", "E3")}
    note = (f"E1 best Gardner - random: {e['E1']['estimate']:+.1f} pts "
            f"[{e['E1']['ci_low']:.1f}, {e['E1']['ci_high']:.1f}]   "
            f"E2 embryologist - random: {e['E2']['estimate']:+.1f} "
            f"[{e['E2']['ci_low']:.1f}, {e['E2']['ci_high']:.1f}]   "
            f"E3 model - embryologist (upper bound): {e['E3']['estimate']:+.1f} "
            f"[{e['E3']['ci_low']:.1f}, {e['E3']['ci_high']:.1f}]")
    if "null_95th" in s.columns and pd.notna(s.loc["E3", "null_95th"]):
        note += ("\nE3 is within the noise floor (95th percentile of the same estimand with "
                 f"permuted outcomes: {s.loc['E3', 'null_95th']:.1f} pts); "
                 "it is not evidence of a better selector.")
    ax.set_title("Choosing among sibling blastocysts moves live birth by a few points",
                 loc="left", color=INK, fontsize=10, fontweight="bold")
    fig.text(0.01, -0.06, note, color=MUTED, fontsize=7, va="top")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure2(f_summary: pd.DataFrame, e7: pd.DataFrame, grid: pd.DataFrame, path) -> None:
    """A: live birth by blastocysts available, observed (adjusted) vs implied by the
    morphology predictor of each scenario. B: gain over morphology of better predictors."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    f = f_summary.set_index("quantity")
    lv = ["2", "3", "4+"]
    x = np.arange(len(lv))
    est = [f.loc[f"RD_{k}_vs_1", "estimate"] for k in lv]
    lo = [f.loc[f"RD_{k}_vs_1", "ci_low"] for k in lv]
    hi = [f.loc[f"RD_{k}_vs_1", "ci_high"] for k in lv]
    ok = e7[e7["reproduces_E1"].astype(bool)]
    for _, r in ok.iterrows():
        a.plot(x, [r[f"implied_RD_{k}_vs_1"] for k in lv], color=ACCENT, lw=0.8, alpha=0.6)
    a.plot([], [], color=ACCENT, lw=0.8, label="Implied by morphology (each σ)")
    a.errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o",
               color=INK, capsize=3, label="Observed, adjusted (95% CI)", zorder=3)
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_xticks(x, [f"{k} vs 1" for k in lv])
    a.set_xlabel("Blastocysts available")
    a.set_ylabel("Difference in live birth (points)")
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
             ncol=2)
    a.set_title("A. Value of having a choice", loc="left", fontsize=9, fontweight="bold",
                color=INK)

    g = grid[grid["reachable"].astype(bool) & (grid["delta_auroc"].astype(str) != "perfect")]
    g = g.assign(delta=g["delta_auroc"].astype(float))
    sig = sorted(g["sigma_e"].unique())
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, max(len(sig), 1)))
    for col, sigma in zip(shades, sig, strict=False):
        d = g[g["sigma_e"] == sigma].sort_values("delta")
        b.plot(d["delta"], d["gain_vs_gardner_choice_set"], marker="o", ms=3, color=col, lw=1,
               label=f"σ = {sigma}")
    b.axhline(5, color="#9A5419", lw=0.8, ls="--")
    b.text(0.003, 5.25, "+5 points: hypothesis of Illingworth et al. 2024", fontsize=6.5,
           color="#9A5419")
    b.set_xlabel("AUROC gain of the predictor over morphology")
    b.set_ylabel("Live-birth gain over morphology (points)")
    b.legend(frameon=False, fontsize=6, title="embryo-level SD", title_fontsize=6.5,
             loc="lower right", ncol=2)
    b.set_title("B. Ceiling for a better predictor", loc="left", fontsize=9,
                fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def calibration_plot(y: np.ndarray, p: np.ndarray, groups: np.ndarray, path) -> None:
    """Supplementary: calibration of cross-fitted predictions, deciles, by choice group."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for g, col in (("no_choice", MUTED), ("choice", ACCENT)):
        m = groups == g
        q = pd.qcut(p[m], 5, duplicates="drop")
        d = pd.DataFrame({"p": p[m], "y": y[m], "q": q}).groupby("q", observed=True).mean()
        ax.plot(d["p"] * 100, d["y"] * 100, "o-", color=col, ms=3, label=g.replace("_", " "))
    ax.plot([0, 70], [0, 70], color=GRID, lw=0.8)
    ax.set_xlabel("Predicted live birth (%)")
    ax.set_ylabel("Observed live birth (%)")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def to_markdown(df: pd.DataFrame, path, floatfmt: str = ".2f") -> None:
    """Write a table as Markdown (no extra dependency)."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- article extras
PRIMARY_C, OTHER_C = "#eb6834", "#2a78d6"


def flow_diagram(flow: pd.DataFrame, path) -> None:
    """Participant flow (TRIPOD / STROBE style) from results/tables/flow.csv."""
    from matplotlib.patches import FancyBboxPatch

    v = dict(zip(flow["step"].str.strip(), flow["transfers_or_rows"], strict=True))
    p = dict(zip(flow["step"].str.strip(), flow["patients"], strict=True))
    k = list(v)
    rows0, single, day5, choice, nochoice = (v[k[0]], v[k[3]], v[k[4]], v[k[5]], v[k[6]])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, strong=True):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=INK if strong else MUTED, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                color=INK if strong else MUTED)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0,
                                    shrinkB=0))

    box(3.4, 9.2, 5.2, 0.9, f"Fresh transfers in the clinical file\n{rows0} rows, "
        f"{p[k[0]]} patients (Linz, Kromp et al. 2023)")
    box(8.0, 7.75, 3.8, 1.25, f"Excluded\n{-v[k[1]]} rows: double-embryo\ntransfers "
        f"({-p[k[1]]} patients)\n{-v[k[2]]}: transferred image without grade", False)
    box(3.4, 6.3, 5.2, 0.9, f"Single-embryo transfers analysed\n{single} patients "
        "(prediction model, step B)")
    box(8.0, 4.9, 3.8, 0.8, f"Day-4 transfers: {single - day5}\n(model only; "
        "sensitivity S1a)", False)
    box(3.4, 3.5, 5.2, 0.9, f"Day-5 single-embryo transfers\n{day5} patients")
    box(1.7, 1.0, 3.2, 1.1, f"At least 2 blastocysts\n{choice} patients\n"
        "(selection estimands E1-E4)")
    box(5.9, 1.0, 3.2, 1.1, f"Single blastocyst\n{nochoice} patients\n"
        "(no-choice group, E7)")
    arrow(3.4, 8.75, 3.4, 6.75)
    arrow(3.4, 7.75, 6.1, 7.75)
    arrow(3.4, 5.85, 3.4, 3.95)
    arrow(3.4, 4.9, 6.1, 4.9)
    arrow(2.6, 3.05, 1.7, 1.55)
    arrow(4.2, 3.05, 5.9, 1.55)
    fig.savefig(path)
    plt.close(fig)


def with_primary(sens: pd.DataFrame, tables) -> pd.DataFrame:
    """Replace S0 by the primary estimate of step C (pooled over imputations, main bootstrap),
    so that every figure shows the same primary E1 as Figure 1 and the text. The S1-S10 rows
    keep the sensitivity settings (one imputation, smaller bootstrap)."""
    p = tables / "strategies.csv"
    if not p.exists():
        return sens
    e1 = pd.read_csv(p).set_index("quantity").loc["E1"]
    s = sens.copy()
    m = s["code"] == "S0"
    s.loc[m, "E1"] = e1["estimate"]
    s.loc[m, "E1_ci"] = f"[{e1['ci_low']:.1f}, {e1['ci_high']:.1f}]"
    return s


def sensitivity_forest(sens: pd.DataFrame, path) -> None:
    """E1 (best by Gardner minus random) across all sensitivity analyses."""
    d = sens.copy()
    ci = d["E1_ci"].str.strip("[]").str.split(",", expand=True).astype(float)
    d["lo"], d["hi"] = ci[0], ci[1]
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(d) + 1.0))
    prim = float(sens.loc[sens["code"] == "S0", "E1"].iloc[0])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(prim, color=ACCENT, lw=0.8, ls="--")
    for i, r in d.iterrows():
        c = ACCENT if r["code"] == "S0" else INK
        ax.errorbar(r["E1"], i, xerr=[[r["E1"] - r["lo"]], [r["hi"] - r["E1"]]], fmt="o",
                    color=c, ms=4, capsize=2, lw=1)
        ax.text(1.02, i, f"{r['E1']:+.1f} [{r['lo']:.1f}, {r['hi']:.1f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(d)), [f"{r.code}  {r.analysis}" for r in d.itertuples()],
                  fontsize=7)
    ax.set_xlabel("E1: live birth, best by Gardner minus random embryo (points, 95% CI)")
    ax.set_title("Sensitivity of the selection headroom (dashed: primary analysis)",
                 loc="left", fontsize=9, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(path)
    plt.close(fig)


def specification_curve(img: pd.DataFrame, path, group: str = "choice") -> None:
    """Every image configuration vs the Gardner model, by outcome (patients with a choice).

    Paired AUROC difference with 95% CI, sorted; the pre-specified primary configuration is
    highlighted. Shows at a glance that no choice of backbone, pre-processing or number of
    components changes the conclusion."""
    d = img[(img["group"] == group) & (img["config"] != "gardner_reference")].copy()
    ci = d["delta_ci"].str.strip("[]").str.split(",", expand=True).astype(float)
    d["lo"], d["hi"] = ci[0], ci[1]
    outs = [o for o in ("SS", "HA", "LB") if o in set(d["outcome"])]
    title = {"SS": "Positive hCG", "HA": "Heart activity", "LB": "Live birth"}
    fig, axes = plt.subplots(1, len(outs), figsize=(3.4 * len(outs), 3.6), sharey=True,
                             squeeze=False)
    for ax, oc in zip(axes[0], outs, strict=True):
        x = d[d["outcome"] == oc].sort_values("delta_auroc_vs_gardner").reset_index(drop=True)
        prim = x["config"].str.contains("PRIMARY")
        for i, r in x.iterrows():
            c = PRIMARY_C if prim[i] else OTHER_C
            ax.plot([i, i], [r["lo"], r["hi"]], color=c, lw=1, alpha=0.7)
            ax.plot(i, r["delta_auroc_vs_gardner"], "o", color=c, ms=4 if not prim[i] else 6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(x)} configurations, sorted")
        ax.set_title(title[oc], loc="left", fontsize=9, color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0, 0].set_ylabel("AUROC difference, image minus Gardner\n(95% CI)")
    from matplotlib.lines import Line2D

    axes[0, -1].legend([Line2D([], [], marker="o", color=PRIMARY_C, lw=0),
                        Line2D([], [], marker="o", color=OTHER_C, lw=0)],
                       ["Pre-specified primary", "Other configurations"], frameon=False,
                       fontsize=7, loc="lower right")
    fig.suptitle("No image configuration separates transferable embryos better than Gardner "
                 "(patients with a choice)", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
