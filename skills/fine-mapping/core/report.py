"""
report.py — Markdown report, TSV tables, and matplotlib figures for fine-mapping.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DISCLAIMER = (
    "ClawBio is a research and educational tool. It is not a medical device "
    "and does not provide clinical diagnoses. Consult a healthcare "
    "professional before making any medical decisions."
)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_markdown(
    df: pd.DataFrame,
    credible_sets: list[dict],
    method: str,
    params: dict,
    input_path: str = "sumstats",
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_variants = len(df)
    n_cs = len(credible_sets)
    lead_pip = float(df["pip"].max()) if "pip" in df.columns else 0.0
    lead_rsid = df.loc[df["pip"].idxmax(), "rsid"] if "pip" in df.columns and len(df) > 0 else "?"

    lines = [
        "# SuSiE Fine-Mapping Report",
        "",
        f"**Date**: {now}",
        f"**Skill**: fine-mapping",
        f"**Method**: {method}",
        f"**Input**: {input_path}",
        f"**Variants analysed**: {n_variants:,}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Method | {method} |",
        f"| Variants | {n_variants:,} |",
        f"| Credible sets | {n_cs} |",
        f"| Lead variant | {lead_rsid} (PIP = {lead_pip:.3f}) |",
        f"| Coverage threshold | {params.get('coverage', 0.95)*100:.0f}% |",
    ]

    if method == "SuSiE":
        lines += [
            f"| Max signals (L) | {params.get('L', '?')} |",
            f"| Min purity | {params.get('min_purity', 0.5)} |",
            f"| Converged | {params.get('converged', '?')} |",
            f"| Iterations | {params.get('n_iter', '?')} |",
        ]
    else:
        lines += [
            f"| Prior variance (W) | {params.get('w', 0.04)} |",
        ]

    lines += ["", "---", ""]

    # Per-CS tables
    lines += ["## Credible Sets", ""]
    if n_cs == 0:
        lines += ["No credible sets identified.", ""]
    else:
        for cs in credible_sets:
            purity_str = f"{cs['purity']:.3f}" if cs.get("purity") is not None else "N/A"
            pure_flag = "" if cs.get("pure", True) else " ⚠️ low purity"
            lines += [
                f"### {cs['cs_id']}{pure_flag}",
                "",
                f"- **Size**: {cs['size']} variants",
                f"- **Coverage**: {cs['coverage']*100:.1f}%",
                f"- **Lead variant**: {cs['lead_rsid']} (α = {cs['lead_alpha']:.4f})",
                f"- **Purity** (mean |r|): {purity_str}",
                "",
                "| rsID | Chr | Pos | Z | PIP | α |",
                "|------|-----|-----|---|-----|---|",
            ]
            for v in cs["variants"]:
                pos_str = f"{v['pos']:,}" if v.get("pos") else "?"
                p_str = f"{v.get('p', ''):.2e}" if v.get("p") else ""
                lines.append(
                    f"| {v['rsid']} | {v['chr']} | {pos_str} | "
                    f"{v['z']:.3f} | {v['pip']:.4f} | {v['alpha']:.4f} |"
                )
            lines.append("")

    # High-PIP variants
    lines += ["## High-PIP Variants (PIP ≥ 0.1)", ""]
    high_pip = df[df["pip"] >= 0.1].sort_values("pip", ascending=False)
    if len(high_pip) == 0:
        lines += ["No variants with PIP ≥ 0.1.", ""]
    else:
        lines += ["| rsID | Chr | Pos | Z | PIP | CS |", "|------|-----|-----|---|-----|----|"]
        for _, row in high_pip.iterrows():
            pos_str = f"{int(row['pos']):,}" if pd.notna(row.get("pos")) else "?"
            cs_label = str(row.get("cs_membership", ""))
            lines.append(
                f"| {row['rsid']} | {row.get('chr', '?')} | {pos_str} | "
                f"{row['z']:.3f} | {row['pip']:.4f} | {cs_label} |"
            )
        lines.append("")

    # Methodology
    if method == "SuSiE":
        lines += [
            "## Methodology",
            "",
            "Fine-mapping was performed using the **SuSiE** (Sum of Single Effects) algorithm "
            "(Wang et al. 2020, JRSS-B). SuSiE models the GWAS signal as a sum of L single effects, "
            "each with a sparse prior, and estimates posterior inclusion probabilities (PIPs) via "
            "Iterative Bayesian Stepwise Selection (IBSS). An LD matrix was used to account for "
            "linkage disequilibrium between variants.",
            "",
            "**Citations**:",
            "- Wang G et al. (2020) *A simple new approach to variable selection in regression, "
            "with application to genetic fine mapping*. JRSS-B. doi:10.1111/rssb.12388",
            "",
        ]
    else:
        lines += [
            "## Methodology",
            "",
            "Fine-mapping was performed using **Approximate Bayes Factors** (ABF; Wakefield 2009). "
            "ABF assumes at most one causal variant per locus and requires only z-scores (no LD matrix). "
            "PIPs are computed as the normalised ABF across all variants under a uniform prior.",
            "",
            "**Citations**:",
            "- Wakefield J (2009) *Bayes factors for genome-wide association studies: comparison "
            "with P-values*. Am J Hum Genet. doi:10.1016/j.ajhg.2008.12.010",
            "",
        ]

    lines += [
        "---",
        "",
        f"*{DISCLAIMER}*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_tables(output_dir: Path, df: pd.DataFrame, credible_sets: list[dict]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # pips.tsv
    cols = [c for c in ["rsid", "chr", "pos", "z", "beta", "se", "p", "maf", "pip", "cs_membership"] if c in df.columns]
    df[cols].to_csv(tables_dir / "pips.tsv", sep="\t", index=False, float_format="%.6g")

    # credible_sets.tsv
    if credible_sets:
        rows = []
        for cs in credible_sets:
            for v in cs["variants"]:
                rows.append({
                    "cs_id": cs["cs_id"],
                    "cs_size": cs["size"],
                    "cs_coverage": cs["coverage"],
                    "lead_rsid": cs["lead_rsid"],
                    "purity": cs.get("purity"),
                    "rsid": v["rsid"],
                    "chr": v.get("chr"),
                    "pos": v.get("pos"),
                    "z": v["z"],
                    "pip": v["pip"],
                    "alpha": v["alpha"],
                })
        pd.DataFrame(rows).to_csv(tables_dir / "credible_sets.tsv", sep="\t", index=False, float_format="%.6g")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def generate_figures(
    output_dir: Path,
    df: pd.DataFrame,
    credible_sets: list[dict],
    R: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [report] matplotlib not available — skipping figures")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_pip_locus(df, credible_sets, R, figures_dir, plt, mcolors)
    if "p" in df.columns and df["p"].notna().any():
        _plot_regional_association(df, credible_sets, figures_dir, plt)
    if R is not None:
        _plot_ld_heatmap(R, df, credible_sets, figures_dir, plt, mcolors)

    plt.close("all")


def _plot_pip_locus(df, credible_sets, R, figures_dir, plt, mcolors):
    """PIP locus plot, variants coloured by LD r² to lead variant."""
    fig, ax = plt.subplots(figsize=(10, 4))

    x = df["pos"].values if "pos" in df.columns and df["pos"].notna().all() else np.arange(len(df))
    pip = df["pip"].values

    # Compute LD r² to lead variant
    if R is not None and len(df) > 0:
        lead_idx = int(np.argmax(pip))
        r2 = R[lead_idx, :] ** 2
        r2 = np.clip(r2, 0, 1)
        colors = _r2_colors(r2, mcolors)
    else:
        colors = ["#3b528b"] * len(df)

    # Mark CS members
    cs_indices = set()
    for cs in credible_sets:
        for v in cs["variants"]:
            matches = df.index[df["rsid"] == v["rsid"]].tolist()
            cs_indices.update(matches)

    scatter = ax.scatter(x, pip, c=colors, s=30, edgecolors="none", zorder=2, alpha=0.85)

    # Highlight CS members
    if cs_indices:
        cs_list = list(cs_indices)
        ax.scatter(x[cs_list], pip[cs_list], s=60, facecolors="none",
                   edgecolors="black", linewidths=0.8, zorder=3)

    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="#D55E00", linestyle="--", linewidth=0.8, alpha=0.6, label="PIP = 0.5")
    ax.set_xlabel("Position" if "pos" in df.columns else "Variant index")
    ax.set_ylabel("Posterior Inclusion Probability (PIP)")
    ax.set_title("Fine-mapping Locus Plot (PIP)")
    ax.legend(fontsize=8)

    if R is not None:
        _add_r2_colorbar(fig, ax, mcolors)

    plt.tight_layout()
    fig.savefig(figures_dir / "pip_locus_plot.png", dpi=150)
    plt.close(fig)


def _plot_regional_association(df, credible_sets, figures_dir, plt):
    """–log10(p) regional association plot."""
    fig, ax = plt.subplots(figsize=(10, 4))

    x = df["pos"].values if "pos" in df.columns and df["pos"].notna().all() else np.arange(len(df))

    p = df["p"].values.astype(float)
    with np.errstate(divide="ignore"):
        log_p = -np.log10(np.where(p > 0, p, 1e-300))
    ax.scatter(x, log_p, s=15, c="#31688e", edgecolors="none", alpha=0.8)
    ax.set_xlabel("Position")
    ax.set_ylabel("–log₁₀(p)")
    ax.set_title("Regional Association")

    plt.tight_layout()
    fig.savefig(figures_dir / "regional_association.png", dpi=150)
    plt.close(fig)


def _plot_ld_heatmap(R, df, credible_sets, figures_dir, plt, mcolors):
    """LD r² heatmap with credible set annotations on both axes."""
    p = R.shape[0]

    # For large loci subsample to keep the figure readable (max 300 variants)
    if p > 300:
        step = p // 300
        idx = np.arange(0, p, step)
        R_plot = R[np.ix_(idx, idx)]
        labels = df["rsid"].iloc[idx].tolist() if "rsid" in df.columns else [str(i) for i in idx]
        pos_plot = df["pos"].iloc[idx].values if "pos" in df.columns else idx
        subsampled = True
    else:
        R_plot = R
        labels = df["rsid"].tolist() if "rsid" in df.columns else [str(i) for i in range(p)]
        pos_plot = df["pos"].values if "pos" in df.columns else np.arange(p)
        idx = np.arange(p)
        subsampled = False

    r2_plot = R_plot ** 2

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.cm.get_cmap("viridis")
    im = ax.imshow(r2_plot, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # Overlay credible set boundaries as coloured rectangles on the diagonal
    # Okabe-Ito colorblind-safe palette
    cs_colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00",
                 "#CC79A7", "#000000", "#F0E442"]
    cs_rsid_to_idx = {}
    for cs in credible_sets:
        if not cs.get("pure", True):
            continue
        for v in cs["variants"]:
            matches = np.where(df["rsid"].values == v["rsid"])[0]
            if len(matches):
                cs_rsid_to_idx[int(matches[0])] = cs["cs_id"]

    # Map original indices to plot indices
    idx_set = set(idx.tolist())
    for orig_idx, cs_id in cs_rsid_to_idx.items():
        if orig_idx not in idx_set:
            continue
        plot_i = int(np.where(idx == orig_idx)[0][0])
        cs_num = int(cs_id.lstrip("LABFabf_").split("_")[0].replace("CS", "")) - 1
        color = cs_colors[cs_num % len(cs_colors)]
        rect = plt.Rectangle(
            (plot_i - 0.5, plot_i - 0.5), 1, 1,
            linewidth=0, facecolor=color, alpha=0.6, zorder=2,
        )
        ax.add_patch(rect)

    # Axis tick labels: use position if available, else variant index
    n_ticks = min(10, len(pos_plot))
    tick_step = max(1, len(pos_plot) // n_ticks)
    tick_positions = list(range(0, len(pos_plot), tick_step))
    if "pos" in df.columns and df["pos"].notna().all():
        tick_labels = [f"{pos_plot[i]/1e6:.2f}Mb" for i in tick_positions]
    else:
        tick_labels = [str(idx[i]) for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LD r²", fontsize=9)

    title = "LD Matrix (r²)"
    if subsampled:
        title += f" — subsampled to {len(idx)} of {p} variants"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Variants" if "pos" not in df.columns else "Position")
    ax.set_ylabel("Variants" if "pos" not in df.columns else "Position")

    # Legend for credible sets
    seen = {}
    for orig_idx, cs_id in cs_rsid_to_idx.items():
        if cs_id not in seen:
            cs_num = int(cs_id.lstrip("LABFabf_").split("_")[0].replace("CS", "")) - 1
            seen[cs_id] = cs_colors[cs_num % len(cs_colors)]
    if seen:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=c, label=cs_id, alpha=0.7)
                          for cs_id, c in sorted(seen.items())]
        ax.legend(handles=legend_handles, title="Credible sets",
                  loc="upper right", fontsize=7, title_fontsize=7,
                  framealpha=0.8)

    plt.tight_layout()
    fig.savefig(figures_dir / "ld_heatmap.png", dpi=150)
    plt.close(fig)


def _r2_colors(r2: np.ndarray, mcolors) -> list:
    """Map r² values to viridis colour scale (colorblind-friendly)."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    return [cmap(norm(v)) for v in r2]


def _add_r2_colorbar(fig, ax, mcolors):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("viridis")
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("LD r² to lead", fontsize=8)


# ---------------------------------------------------------------------------
# Reproducibility bundle
# ---------------------------------------------------------------------------


def write_reproducibility(output_dir: Path, cmd: str, params: dict) -> None:
    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    (repro_dir / "commands.sh").write_text(f"#!/bin/bash\n{cmd}\n")

    try:
        from importlib.metadata import version
        packages = {}
        for pkg in ("numpy", "scipy", "pandas", "matplotlib"):
            try:
                packages[pkg] = version(pkg)
            except Exception:
                pass
    except Exception:
        packages = {}

    env = {
        "name": "fine-mapping",
        "channels": ["defaults"],
        "dependencies": [
            "python>=3.9",
            {"pip": [
                f"numpy=={packages.get('numpy', '>=1.24')}",
                f"scipy=={packages.get('scipy', '>=1.10')}",
                f"pandas=={packages.get('pandas', '>=1.5')}",
                f"matplotlib=={packages.get('matplotlib', '>=3.7')}",
            ]}
        ],
    }
    try:
        import yaml
        with open(repro_dir / "environment.yml", "w") as f:
            yaml.dump(env, f, default_flow_style=False)
    except ImportError:
        import json as _json
        (repro_dir / "environment.json").write_text(_json.dumps(env, indent=2))
