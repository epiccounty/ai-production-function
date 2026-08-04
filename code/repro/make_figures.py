#!/usr/bin/env python3
"""Generate Section 9 figures for Paper 1 from the frozen repro panel.

Four charts, one design system (modern-research tone, navy + coral on a light
surface). All data comes from code/repro/panels/ and code/repro/results_*.json.
No extraction, no private data; figures regenerate deterministically.

Output (release/paper1/figures/):
  figure2-quadrant.png          — W x X quadrant: output peaks when compute meets context
  figure3-factor-explanatory.png — corrected partial-R2: which factor explains output
  figure4-identification.png    — within-repo stock variation: why elasticities aren't identified
  figure5-correction.png        — the 49 floor cells and the slope they distorted vs the correction

Usage:  python3 build/make_figures.py [release/paper1 repo root]
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path.cwd()
PANELS = REPO / "code" / "repro" / "panels"
RESULTS = REPO / "code" / "repro" / "results_paper1_corrected.json"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

# ── design system (validated: navy + coral PASS all CVD/contrast checks) ──
NAVY = "#2563a6"
CORAL = "#f4845f"
INK = "#1f2937"        # primary text
MUTED = "#6b7280"      # secondary text / axis
GRID = "#e5e7eb"       # structural grid only
SURFACE = "#fcfcfb"    # background
HAIRLINE = "#d1d5db"   # thin axis lines

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "figure.dpi": 200,
})

# ── load data ──
panels = []
for p in sorted(PANELS.glob("panel_repo*.csv")):
    df = pd.read_csv(p)
    df["repo"] = int(p.stem.replace("panel_repo", ""))
    panels.append(df)
raw = pd.concat(panels, ignore_index=True)
res = json.loads(RESULTS.read_text())
obs = raw[(raw["W"] > 0) & (raw["Y"] > 0)].copy()
for col in ("C", "X", "W", "Y"):
    obs[f"log{col}"] = np.log(obs[col])

LOGFLOOR = np.log(1e-9)


def quadrant():
    """Figure 2 — output peaks when compute meets context (not compute alone)."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    # color points by output level (single-hue ramp = magnitude job)
    sc = ax.scatter(obs["logX"], obs["logW"], c=obs["logY"], cmap="Blues",
                    s=46, edgecolors="white", linewidths=0.6, vmin=obs.logY.min(),
                    vmax=obs.logY.max(), zorder=3)

    # median splits to define quadrants
    xm, wm = obs["logX"].median(), obs["logW"].median()
    ax.axvline(xm, color=HAIRLINE, lw=0.9, ls="--", zorder=1)
    ax.axhline(wm, color=HAIRLINE, lw=0.9, ls="--", zorder=1)

    # quadrant mean-output annotations — the message
    q = obs.assign(xhi=obs.logX > xm, whi=obs.logW > wm)
    means = q.groupby(["whi", "xhi"]).logY.mean()
    def quad_label(whi, xhi, dx, dy):
        m = means[(whi, xhi)]
        xpos = obs.logX.max() if xhi else obs.logX.min()
        ypos = obs.logW.max() if whi else obs.logW.min()
        ax.annotate(f"mean output\n{m:.1f}", xy=(xpos, ypos),
                    xytext=(xpos + dx, ypos + dy),
                    fontsize=8.5, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=GRID, lw=0.8))

    # Highlight the high-high quadrant: this is the paper's empirical pulse.
    ax.add_patch(plt.Rectangle((xm, wm), obs.logX.max() - xm, obs.logW.max() - wm,
                               fc=CORAL, alpha=0.10, zorder=0, ec="none"))
    ax.text(xm + (obs.logX.max() - xm) / 2, wm + (obs.logW.max() - wm) / 2 + 0.0,
            "", ha="center")

    ax.set_xlabel("Context capital  log X")
    ax.set_ylabel("Computational work  log W")
    ax.set_title("Output peaks where compute meets context capital",
                 fontsize=11, color=INK, pad=12, loc="left", weight="bold")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("log Y  (output)", fontsize=8.5, color=MUTED)
    cbar.ax.tick_params(labelsize=7.5, color=MUTED)
    cbar.outline.set_edgecolor(HAIRLINE)

    # the punchline, directly on the highlighted quadrant
    hh = means[(True, True)]
    ax.text(xm + (obs.logX.max() - xm) / 2, obs.logW.max(),
            f"high W + high X\n→ output {hh:.1f}",
            fontsize=8, color=CORAL, ha="center", va="top", weight="bold")

    ax.grid(True, alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "figure2-quadrant.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure2-quadrant.png  (high-W+high-X mean output={hh:.2f})")


def factor_explanatory():
    """Figure 3 — corrected partial R^2: context > compute > code."""
    pr2 = res["corrected"]["partial_r2"]
    factors = [("Context (X)", pr2["X"], NAVY),
               ("Compute (W)", pr2["W"], CORAL),
               ("Code (C)", pr2["C"], MUTED)]
    # rank by magnitude; narrative wants context first
    factors = sorted(factors, key=lambda f: f[1], reverse=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    names = [f[0] for f in factors]
    vals = [f[1] for f in factors]
    colors = [f[2] for f in factors]
    ypos = np.arange(len(factors))[::-1]

    bars = ax.barh(ypos, vals, color=colors, height=0.62, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlabel("Partial R$^2$ with output  (corrected sample, n=74)", color=MUTED)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(True, axis="x", alpha=0.6, zorder=0)

    # value labels at bar ends
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.02, y, f"{v:.2f}", va="center",
                fontsize=9.5, color=INK, weight="bold")

    ax.set_title("Which factor explains output? Context leads, compute is real, code is mute",
                 fontsize=10, color=INK, pad=12, loc="left", weight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "figure3-factor-explanatory.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure3-factor-explanatory.png  (X={vals[0]:.2f} W={vals[1]:.2f} C={vals[2]:.2f})")


def identification():
    """Figure 4 — within-repo stock variation: a dot strip showing every repository
    falls far short of the dispersion an elasticity would need to be identified."""
    disp = obs.groupby("repo")[["logC", "logX"]].std().fillna(0)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))

    # Two horizontal lanes; each repository is one dot in its lane.
    # A subtle stem anchors each dot to its lane so overlapping points read.
    for i, (col, color, lab) in enumerate(
        [("logC", CORAL, "Code capital  $\\sigma(\\log C)$"),
         ("logX", NAVY, "Context capital  $\\sigma(\\log X)$")]):
        lane = 1 - i  # context on top, code below
        vals = disp[col].values
        ax.scatter(vals, np.full(len(vals), lane), s=95, color=color, alpha=0.55,
                   edgecolors="white", linewidths=0.6, zorder=3)
        # faint stems so stacked points are distinguishable
        for v in vals:
            ax.plot([v, v], [lane - 0.06, lane + 0.06], color=color, lw=0.5, alpha=0.3, zorder=2)

    # The identifying scale: a within-repository log-stock swing of order 0.5 is
    # the kind of variation a coefficient needs to lean on. Show it as a band the
    # points must reach but don't.
    IDENT = 0.5
    ax.axvspan(IDENT, IDENT + 0.2, color=GRID, alpha=0.9, zorder=0)
    ax.text(IDENT + 0.1, 1.35, "identifying scale\n($\\sim$0.5 log units)",
            fontsize=8, color=MUTED, ha="center", va="top")
    ax.annotate("", xy=(IDENT, 1.25), xytext=(disp["logX"].max(), 1.25),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    ax.text((IDENT + disp["logX"].max()) / 2, 1.3,
            f"the best repository reaches {disp['logX'].max()/IDENT*100:.0f}% of this",
            fontsize=7.5, color=MUTED, ha="center", va="bottom", style="italic")

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Context  $\\sigma(\\log X)$", "Code  $\\sigma(\\log C)$"],
                       fontsize=9.5, color=INK)
    ax.set_xlim(-0.03, 0.72)
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlabel("Within-repository dispersion (log units)  —  each dot is one repository",
                  color=MUTED)
    ax.grid(True, axis="x", alpha=0.5, zorder=0)

    ax.set_title("Every repository falls far short of the dispersion that would identify a stock elasticity",
                 fontsize=9.5, color=INK, pad=12, loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure4-identification.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure4-identification.png  "
          f"(best C {disp['logC'].max():.2f} = {disp['logC'].max()/IDENT*100:.0f}% of scale, "
          f"best X {disp['logX'].max():.2f} = {disp['logX'].max()/IDENT*100:.0f}%)")


def correction():
    """Figure 5 — same data, one filter: the compute slope flips sign.
    Two side-by-side panels keep each on its natural axis: the left (defective)
    shows the 49 floor cells pinning the slope; the right (corrected) drops them
    and the slope turns positive. The headline is the paired slope callout."""
    floor = raw[~((raw["W"] > 0) & (raw["Y"] > 0))].copy()
    fy = np.log(np.where(floor["Y"] > 0, floor["Y"], 1e-9))

    g_def = res["as_shipped"]["compute_only"]["elasticity_W"]
    g_cor = res["corrected"]["compute_only"]["elasticity_W"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 4.3), sharey=True)

    # ── left: defective — floor cells included, axis spans the floor ──
    axL.scatter(obs["logW"], obs["logY"], s=20, c=NAVY, alpha=0.6,
                edgecolors="white", linewidths=0.3, zorder=3)
    axL.scatter(np.full(len(floor), LOGFLOOR), fy, s=18, c=CORAL, alpha=0.7,
                marker="x", linewidths=0.9, zorder=3)
    # defective fit pinned near the floor cluster
    xs = np.linspace(-22, 2, 50)
    axL.plot(xs, fy.mean() + g_def * (xs - LOGFLOOR), color=CORAL, lw=2.0,
             ls="--", zorder=4)
    axL.axvline(LOGFLOOR, color=MUTED, lw=0.7, ls=":", zorder=1)
    axL.text(LOGFLOOR, 5.3, " log floor\n $-20.7$", fontsize=7, color=MUTED, va="top")
    axL.set_title("Defective path  (n=123, 49 at the floor)",
                  fontsize=9.5, color=INK, loc="left", weight="bold")
    axL.set_xlabel("log W")
    axL.set_xlim(-23, 3)

    # ── right: corrected — floor cells dropped, natural axis ──
    axR.scatter(obs["logW"], obs["logY"], s=24, c=NAVY, alpha=0.65,
                edgecolors="white", linewidths=0.3, zorder=3)
    xr = np.linspace(obs["logW"].min(), obs["logW"].max(), 50)
    lw0, ly0 = obs["logW"].mean(), obs["logY"].mean()
    axR.plot(xr, ly0 + g_cor * (xr - lw0), color=NAVY, lw=2.2, zorder=4)
    axR.set_title("Corrected  (n=74, floor dropped)",
                  fontsize=9.5, color=INK, loc="left", weight="bold")
    axR.set_xlabel("log W")

    for ax in (axL, axR):
        ax.grid(True, alpha=0.55)
    axL.set_ylabel("log Y  (output)")

    # ── the headline: the slope value on each panel, sign-coloured ──
    axL.text(0.97, 0.06, f"$\\hat\\gamma_W = {g_def:+.2f}$", transform=axL.transAxes,
             ha="right", va="bottom", fontsize=11, color=CORAL, weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))
    axR.text(0.97, 0.06, f"$\\hat\\gamma_W = {g_cor:+.2f}$", transform=axR.transAxes,
             ha="right", va="bottom", fontsize=11, color=NAVY, weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))

    fig.suptitle("One filter flips the compute slope from negative to positive",
                 fontsize=10.5, color=INK, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figure5-correction.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure5-correction.png  (defective {g_def:+.2f} -> corrected {g_cor:+.2f})")


print("generating Section 9 figures:")
quadrant()
factor_explanatory()
identification()
correction()
print("done.")
