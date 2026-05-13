#!/usr/bin/env python3
"""
toy_topology_mc.py

Toy Monte Carlo for delayed temporal-topological / TDF-unraveling signatures.

This script is a signal-template generator, not a detector simulation. It compares
three timing hypotheses:

1. prompt: daughters emitted at t ~= 0, measured spread dominated by detector resolution
2. llp: parent decays at a delayed time; daughters cluster around one delayed time
3. tdf: daughters emitted across a finite topological-unraveling interval

Outputs:
- full daughter-level CSV
- event-level summary CSV
- plots comparing prompt / LLP / TDF timing templates
- optional short PDF note with plots

Dependencies:
    numpy pandas matplotlib

Run:
    python toy_topology_mc.py --n-events 3000 --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

Mode = Literal["prompt", "llp", "tdf"]


@dataclass(frozen=True)
class MCConfig:
    n_events_per_mode: int = 3000
    tau0_s: float = 1e-9
    alpha: float = 0.9
    k_mean: float = 1.2
    e_top_per_k: float = 1000.0
    detector_time_res_s: float = 200e-12
    llp_mean_lifetime_s: float = 15e-9
    llp_vertex_width_s: float = 50e-12
    prompt_vertex_width_s: float = 10e-12
    seed: int = 42
    truncate_at_tau: float = 5.0


def sample_complexity(rng: np.random.Generator, mean: float) -> int:
    """Poisson-like effective complexity index K >= 0."""
    return int(max(0, rng.poisson(mean)))


def tau_of_k(k: int, tau0_s: float, alpha: float) -> float:
    """Topological unraveling timescale tau(K)."""
    return float(tau0_s * np.exp(alpha * k))


def n_daughters_of_k(rng: np.random.Generator, k: int) -> int:
    """Toy daughter multiplicity. Higher K tends to create larger multiplicity."""
    lam = 1.0 + max(0, k)
    return int(rng.poisson(lam) + 1)


def truncated_exponential(
    rng: np.random.Generator,
    n: int,
    tau_s: float,
    t_max_s: float,
) -> np.ndarray:
    """Sample n times from Exp(tau_s), truncated by rejection at t_max_s."""
    if n <= 0:
        return np.array([], dtype=float)
    times: list[float] = []
    while len(times) < n:
        draw = rng.exponential(scale=tau_s, size=max(1, n - len(times)))
        accepted = draw[draw <= t_max_s]
        times.extend(accepted.tolist())
    return np.array(times[:n], dtype=float)


def energies_for_event(
    rng: np.random.Generator,
    k: int,
    n: int,
    e_top_per_k: float,
) -> np.ndarray:
    """
    Stochastic energy partition.

    For physical signal events, nontrivial topological energy is associated with K > 0.
    K = 0 is treated as a prompt/trivial control sector, but we still assign a small
    baseline visible energy scale for numerical plotting.
    """
    if n <= 0:
        return np.array([], dtype=float)
    total_e = e_top_per_k * max(1, k)
    parts = rng.exponential(scale=1.0, size=n)
    parts /= parts.sum()
    return parts * total_e


def emission_times_for_event(
    rng: np.random.Generator,
    mode: Mode,
    k: int,
    n: int,
    tau_s: float,
    cfg: MCConfig,
) -> np.ndarray:
    """
    Generate true daughter emission times for one event.

    - prompt: all daughters are near t=0
    - llp: parent delay sampled once; daughters cluster near common delayed time
    - tdf: daughters are spread over the unraveling interval
    """
    if n <= 0:
        return np.array([], dtype=float)

    if mode == "prompt":
        times = rng.normal(loc=0.0, scale=cfg.prompt_vertex_width_s, size=n)
        return np.clip(times, 0.0, None)

    if mode == "llp":
        t_decay = rng.exponential(scale=cfg.llp_mean_lifetime_s)
        times = t_decay + rng.normal(loc=0.0, scale=cfg.llp_vertex_width_s, size=n)
        return np.clip(times, 0.0, None)

    if mode == "tdf":
        # K=0 is a trivial-sector prompt-ish control; K>0 is topological signal.
        if k == 0:
            times = rng.normal(loc=0.0, scale=cfg.prompt_vertex_width_s, size=n)
            return np.clip(times, 0.0, None)
        return truncated_exponential(
            rng=rng,
            n=n,
            tau_s=tau_s,
            t_max_s=cfg.truncate_at_tau * tau_s,
        )

    raise ValueError(f"Unknown mode: {mode}")


def simulate_mode(mode: Mode, cfg: MCConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate daughter-level rows for one timing hypothesis."""
    rows = []

    for ev in range(cfg.n_events_per_mode):
        k = sample_complexity(rng, cfg.k_mean)
        tau_s = tau_of_k(k, cfg.tau0_s, cfg.alpha)
        n = n_daughters_of_k(rng, k)
        true_times = emission_times_for_event(rng, mode, k, n, tau_s, cfg)
        energies = energies_for_event(rng, k, n, cfg.e_top_per_k)
        measured_times = true_times + rng.normal(scale=cfg.detector_time_res_s, size=n)
        measured_times = np.clip(measured_times, 0.0, None)

        global_event_id = f"{mode}_{ev}"
        for i in range(n):
            rows.append(
                {
                    "event_id": global_event_id,
                    "mode": mode,
                    "K": int(k),
                    "tau_s": tau_s,
                    "daughter_index": int(i),
                    "true_t_s": float(true_times[i]),
                    "measured_t_s": float(measured_times[i]),
                    "energy": float(energies[i]),
                }
            )

    return pd.DataFrame(rows)


def summarize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Create event-level summary, including Δt and energy-time observables."""
    def energy_weighted_time(group: pd.DataFrame, time_col: str) -> float:
        total_e = group["energy"].sum()
        if total_e <= 0:
            return np.nan
        return float((group[time_col] * group["energy"]).sum() / total_e)

    grouped = df.groupby(["event_id", "mode"], sort=False)

    summary = grouped.agg(
        K=("K", "first"),
        tau_s=("tau_s", "first"),
        n_daughters=("daughter_index", "count"),
        total_energy=("energy", "sum"),
        t_min_true_s=("true_t_s", "min"),
        t_max_true_s=("true_t_s", "max"),
        t_min_measured_s=("measured_t_s", "min"),
        t_max_measured_s=("measured_t_s", "max"),
    ).reset_index()

    summary["delta_t_true_s"] = summary["t_max_true_s"] - summary["t_min_true_s"]
    summary["delta_t_measured_s"] = summary["t_max_measured_s"] - summary["t_min_measured_s"]

    t_energy_true = grouped.apply(lambda g: energy_weighted_time(g, "true_t_s"), include_groups=False)
    t_energy_meas = grouped.apply(lambda g: energy_weighted_time(g, "measured_t_s"), include_groups=False)
    summary["energy_weighted_true_t_s"] = t_energy_true.to_numpy()
    summary["energy_weighted_measured_t_s"] = t_energy_meas.to_numpy()

    # Early energy fraction, using event-specific tau as the cutoff.
    early_fracs = []
    for _, event in summary.iterrows():
        g = df[df["event_id"] == event["event_id"]]
        cutoff = event["tau_s"]
        total_e = g["energy"].sum()
        early_e = g.loc[g["true_t_s"] < cutoff, "energy"].sum()
        early_fracs.append(float(early_e / total_e) if total_e > 0 else np.nan)
    summary["early_energy_fraction_tau"] = early_fracs

    return summary


def save_plots(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> dict[str, Path]:
    """Generate recommended plots."""
    paths: dict[str, Path] = {}

    # Plot 1: true emission times by mode
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in ["prompt", "llp", "tdf"]:
        vals_ns = df.loc[df["mode"] == mode, "true_t_s"].to_numpy() * 1e9
        ax.hist(vals_ns, bins=150, histtype="step", label=mode, linewidth=1.2)
    ax.set_xlabel("True daughter emission time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("True emission-time templates")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["emission_time_compare"] = outdir / "emission_time_compare.png"
    fig.savefig(paths["emission_time_compare"], dpi=200)
    plt.close(fig)

    # Plot 2: measured Δt by mode
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in ["prompt", "llp", "tdf"]:
        vals_ns = summary.loc[summary["mode"] == mode, "delta_t_measured_s"].to_numpy() * 1e9
        ax.hist(vals_ns, bins=120, histtype="step", label=mode, linewidth=1.2)
    ax.set_xlabel(r"Measured intra-event time spread $\Delta t_{\rm event}$ (ns)")
    ax.set_ylabel("Number of events")
    ax.set_title(r"Prompt vs LLP vs TDF: $\Delta t_{\rm event}$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["delta_t_compare"] = outdir / "delta_t_compare.png"
    fig.savefig(paths["delta_t_compare"], dpi=200)
    plt.close(fig)

    # Plot 3: TDF Δt vs K
    fig, ax = plt.subplots(figsize=(8, 4))
    tdf = summary[summary["mode"] == "tdf"]
    ax.scatter(tdf["K"], tdf["delta_t_measured_s"] * 1e9, s=10, alpha=0.5)
    ax.set_xlabel("Topological complexity K")
    ax.set_ylabel(r"Measured $\Delta t_{\rm event}$ (ns)")
    ax.set_title("TDF template: complexity vs intra-event time spread")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["tdf_delta_t_vs_k"] = outdir / "tdf_delta_t_vs_k.png"
    fig.savefig(paths["tdf_delta_t_vs_k"], dpi=200)
    plt.close(fig)

    # Plot 4: example TDF event timeline with K>=3 if available
    candidates = summary[(summary["mode"] == "tdf") & (summary["K"] >= 3)]
    if candidates.empty:
        candidates = summary[summary["mode"] == "tdf"]
    example_event_id = candidates.sample(1, random_state=7).iloc[0]["event_id"]
    ev_rows = df[df["event_id"] == example_event_id].sort_values("true_t_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 3))
    y = np.arange(len(ev_rows))
    ax.scatter(ev_rows["true_t_s"] * 1e9, y, marker="o", label="true emission times")
    ax.scatter(ev_rows["measured_t_s"] * 1e9, y, marker="x", label="measured times")
    for idx, row in ev_rows.iterrows():
        ax.plot(
            [row["true_t_s"] * 1e9, row["measured_t_s"] * 1e9],
            [idx, idx],
            linestyle="--",
            linewidth=0.8,
        )
    ax.set_xlabel("Time (ns)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"d{i}" for i in ev_rows["daughter_index"]])
    ax.legend(loc="upper right", fontsize=8)
    k_val = int(ev_rows["K"].iloc[0])
    tau_val = ev_rows["tau_s"].iloc[0]
    ax.set_title(f"Example TDF event timeline ({example_event_id}, K={k_val}, tau={tau_val:.2e} s)")
    fig.tight_layout()
    paths["example_tdf_timeline"] = outdir / "example_tdf_timeline.png"
    fig.savefig(paths["example_tdf_timeline"], dpi=200)
    plt.close(fig)

    return paths


def save_search_note_pdf(plot_paths: dict[str, Path], outdir: Path) -> Path:
    """Create a short PDF note with plots. Optional supplementary artifact."""
    pdf_path = outdir / "search_note_topology_timing_templates.pdf"

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.86, "Search Note: Temporal-Topological Timing Templates", fontsize=16, weight="bold")
        fig.text(0.1, 0.82, "Toy MC comparison: prompt vs LLP vs TDF-unraveling hypotheses", fontsize=10)
        fig.text(0.1, 0.75, "Summary", fontsize=12, weight="bold")
        fig.text(
            0.1,
            0.71,
            (
                "This note accompanies toy_topology_mc.py. The code is not a full detector simulation.\n"
                "It generates analysis-level timing templates for delayed temporal-topological events.\n\n"
                "Key discriminator:\n"
                "  intra-event temporal spread among daughter particles.\n\n"
                "Template hypotheses:\n"
                "  prompt: daughters cluster at t ~= 0;\n"
                "  LLP: daughters cluster at one delayed parent decay time;\n"
                "  TDF: daughters are emitted across a finite unraveling interval.\n\n"
                "Recommended next step: fold these templates through experiment-specific trigger,\n"
                "acceptance, pileup, and timing-background models."
            ),
            fontsize=9,
        )
        pdf.savefig(fig)
        plt.close(fig)

        for title, path in [
            ("Emission-time comparison", plot_paths["emission_time_compare"]),
            (r"$\Delta t_{\rm event}$ comparison", plot_paths["delta_t_compare"]),
            ("TDF complexity vs time spread", plot_paths["tdf_delta_t_vs_k"]),
            ("Example TDF event timeline", plot_paths["example_tdf_timeline"]),
        ]:
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(title)
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def run(cfg: MCConfig, outdir: Path, make_pdf: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    daughter_frames = [simulate_mode(mode, cfg, rng) for mode in ["prompt", "llp", "tdf"]]
    daughters = pd.concat(daughter_frames, ignore_index=True)
    summary = summarize_events(daughters)

    daughter_csv = outdir / "toy_topology_daughters.csv"
    summary_csv = outdir / "toy_topology_event_summary.csv"
    daughters.to_csv(daughter_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plot_paths = save_plots(daughters, summary, outdir)
    plot_paths["daughter_csv"] = daughter_csv
    plot_paths["summary_csv"] = summary_csv

    if make_pdf:
        plot_paths["search_note_pdf"] = save_search_note_pdf(plot_paths, outdir)

    return daughters, summary, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy MC for temporal-topological timing templates.")
    parser.add_argument("--n-events", type=int, default=3000, help="Events per mode.")
    parser.add_argument("--tau0", type=float, default=1e-9, help="Base TDF timescale in seconds.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Complexity scaling exponent.")
    parser.add_argument("--k-mean", type=float, default=1.2, help="Poisson mean for K.")
    parser.add_argument("--e-top-per-k", type=float, default=1000.0, help="Energy per complexity unit.")
    parser.add_argument("--detector-time-res", type=float, default=200e-12, help="Detector time resolution in seconds.")
    parser.add_argument("--llp-mean-lifetime", type=float, default=15e-9, help="Mean LLP lifetime in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not create PDF note.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MCConfig(
        n_events_per_mode=args.n_events,
        tau0_s=args.tau0,
        alpha=args.alpha,
        k_mean=args.k_mean,
        e_top_per_k=args.e_top_per_k,
        detector_time_res_s=args.detector_time_res,
        llp_mean_lifetime_s=args.llp_mean_lifetime,
        seed=args.seed,
    )
    daughters, summary, paths = run(cfg, args.outdir, make_pdf=not args.no_pdf)

    print("\nSample event summary:")
    cols = ["event_id", "mode", "K", "tau_s", "n_daughters", "total_energy", "delta_t_measured_s"]
    print(summary[cols].head(12).to_string(index=False))

    print("\nProduced files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
