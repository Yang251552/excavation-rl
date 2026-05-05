"""Visualization utilities for training curves and evaluation results.

Generates publication-quality figures for:
  - Training curves (reward, loss, entropy over iterations)
  - Ablation comparison plots (multiple configs on same axes)
  - Evaluation metric bar charts
  - Reward component decomposition

Usage:
    python -m evaluation.visualize --log-dir results/training_curves
    python -m evaluation.visualize --ablation results/ablation_reports/reward
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(
    log_data: dict[str, list[float]],
    output_path: str,
    title: str = "Training Curves",
    smooth_window: int = 10,
) -> None:
    """Plot training metrics over iterations.

    Args:
        log_data: Dict mapping metric name → list of values.
        output_path: Path to save the figure.
        title: Plot title.
        smooth_window: Smoothing window size.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    metrics_to_plot = [
        ("Performance/episodic_return", "Episodic Return"),
        ("Performance/soil_transfer_ratio", "Soil Transfer Ratio"),
        ("Policy/entropy", "Policy Entropy"),
        ("Policy/value_loss", "Value Loss"),
    ]

    available = [(k, label) for k, label in metrics_to_plot if k in log_data]
    if not available:
        print("No metrics found in log data")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, available):
        data = np.array(log_data[key])
        x = np.arange(len(data))

        # Raw data (light)
        ax.plot(x, data, alpha=0.3, color="steelblue")

        # Smoothed data
        if len(data) > smooth_window:
            smoothed = smooth(data, smooth_window)
            x_smooth = np.arange(smooth_window - 1, len(data))
            ax.plot(x_smooth, smoothed, color="steelblue", linewidth=2)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ablation_comparison(
    results: dict[str, dict[str, list[float]]],
    metric_key: str,
    output_path: str,
    title: str = "Ablation Comparison",
    ylabel: str = "Episodic Return",
    smooth_window: int = 20,
) -> None:
    """Plot multiple training curves on same axes for ablation comparison.

    Args:
        results: Dict of config_name → {metric_key: [values]}.
        metric_key: Which metric to compare.
        output_path: Path to save the figure.
        title: Plot title.
        ylabel: Y-axis label.
        smooth_window: Smoothing window.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (config_name, data), color in zip(results.items(), colors):
        if metric_key not in data:
            continue

        values = np.array(data[metric_key])
        x = np.arange(len(values))

        # Light raw data
        ax.fill_between(x, values, alpha=0.1, color=color)

        # Smoothed
        if len(values) > smooth_window:
            smoothed = smooth(values, smooth_window)
            x_smooth = np.arange(smooth_window - 1, len(values))
            ax.plot(x_smooth, smoothed, label=config_name, color=color, linewidth=2)
        else:
            ax.plot(x, values, label=config_name, color=color, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ablation_comparison_with_seeds(
    results: dict[str, list[list[float]]],
    output_path: str,
    title: str = "Ablation Comparison",
    ylabel: str = "Episodic Return",
    smooth_window: int = 20,
) -> None:
    """Plot ablation comparison with mean +/- std across seeds.

    Args:
        results: Dict of config_name → list of seed curves (list of list).
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (config_name, seed_curves), color in zip(results.items(), colors):
        # Pad to same length
        max_len = max(len(c) for c in seed_curves)
        padded = np.full((len(seed_curves), max_len), np.nan)
        for i, curve in enumerate(seed_curves):
            padded[i, : len(curve)] = curve

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)
        x = np.arange(max_len)

        if max_len > smooth_window:
            mean_smooth = smooth(mean_curve, smooth_window)
            std_smooth = smooth(std_curve, smooth_window)
            x_smooth = np.arange(smooth_window - 1, max_len)
            ax.plot(x_smooth, mean_smooth, label=config_name, color=color, linewidth=2)
            ax.fill_between(
                x_smooth,
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2,
                color=color,
            )
        else:
            ax.plot(x, mean_curve, label=config_name, color=color, linewidth=2)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metric_bar_chart(
    config_names: list[str],
    values: list[float],
    errors: list[float],
    output_path: str,
    title: str = "Metric Comparison",
    ylabel: str = "Value",
) -> None:
    """Plot bar chart comparing a metric across configurations."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(config_names))
    bars = ax.bar(x, values, yerr=errors, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errors) * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualization tools")
    parser.add_argument("--log-dir", type=str, help="TensorBoard log directory")
    parser.add_argument("--ablation", type=str, help="Ablation results directory")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation:
        manifest_path = os.path.join(args.ablation, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"Loaded ablation manifest: {manifest['experiment']}")
            print(f"Configs: {manifest['configs']}")
        else:
            print(f"No manifest found at {manifest_path}")

    print(f"Figures will be saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
