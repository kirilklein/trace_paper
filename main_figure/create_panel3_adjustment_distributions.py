"""Create illustrative adjustment distribution panels for the main figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


BLUE = "#A0C4FF"
ORANGE = "#FFADAD"
OUTPUT_DIR = Path("main_figure/output")


def sample_beta_mixture(
    rng: np.random.Generator,
    n_points: int,
    components: list[tuple[float, float, float]],
    noise_scale: float = 0.015,
) -> np.ndarray:
    """Sample a smooth but slightly irregular propensity-like distribution."""
    weights = np.array([weight for weight, _, _ in components], dtype=float)
    weights /= weights.sum()

    component_index = rng.choice(len(components), size=n_points, p=weights)
    values = np.empty(n_points)

    for idx, (_, alpha, beta) in enumerate(components):
        mask = component_index == idx
        values[mask] = rng.beta(alpha, beta, size=mask.sum())

    values += rng.normal(0.0, noise_scale, size=n_points)
    return np.clip(values, 0.02, 0.98)


def kde_curve(values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.18) -> np.ndarray:
    """Evaluate a KDE on a fixed grid."""
    density = gaussian_kde(values, bw_method=bandwidth)
    return density(grid)


def draw_density(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
    grid: np.ndarray,
    *,
    alpha: float = 0.50,
    linewidth: float = 2.0,
) -> None:
    """Draw one filled smooth density."""
    density = kde_curve(values, grid)
    ax.fill_between(grid, density, color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(grid, density, color=color, linewidth=linewidth, zorder=3)


def style_axis(ax: plt.Axes, title: str) -> None:
    """Apply minimal styling for an overview-style figure panel."""
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=20, pad=16)
    ax.set_xlim(0.02, 0.98)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def build_panel(ax: plt.Axes, *, adjusted: bool, rng: np.random.Generator) -> None:
    """Create one unadjusted or adjusted overlap panel."""
    n_points = 2500
    grid = np.linspace(0.02, 0.98, 500)

    if adjusted:
        control = sample_beta_mixture(
            rng,
            n_points,
            components=[(0.62, 5.5, 2.9), (0.38, 8.5, 3.4)],
        )
        treated = sample_beta_mixture(
            rng,
            n_points,
            components=[(0.58, 5.9, 3.0), (0.42, 8.8, 3.6)],
        )
        title = "Adjusted"
    else:
        control = sample_beta_mixture(
            rng,
            n_points,
            components=[(0.60, 4.6, 3.7), (0.40, 7.5, 3.8)],
        )
        treated = sample_beta_mixture(
            rng,
            n_points,
            components=[(0.45, 6.3, 2.9), (0.55, 9.8, 3.0)],
        )
        title = "Unadjusted"

    draw_density(ax, control, BLUE, grid)
    draw_density(ax, treated, ORANGE, grid)
    style_axis(ax, title)


def create_individual_figure(*, adjusted: bool) -> plt.Figure:
    """Create a standalone density panel."""
    rng = np.random.default_rng(13 if adjusted else 11)
    fig, ax = plt.subplots(figsize=(6.1, 4.4), facecolor="white")
    build_panel(ax, adjusted=adjusted, rng=rng)
    fig.tight_layout()
    return fig


def create_combined_figure() -> plt.Figure:
    """Create a side-by-side overview of unadjusted and adjusted densities."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), facecolor="white")
    build_panel(axes[0], adjusted=False, rng=np.random.default_rng(11))
    build_panel(axes[1], adjusted=True, rng=np.random.default_rng(13))
    fig.tight_layout(w_pad=2.2)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure to PNG and PDF."""
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    """Create all panel 3 assets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_figure(create_individual_figure(adjusted=False), "panel3_unadjusted")
    save_figure(create_individual_figure(adjusted=True), "panel3_adjusted")
    save_figure(create_combined_figure(), "panel3_adjustment_comparison")


if __name__ == "__main__":
    main()
