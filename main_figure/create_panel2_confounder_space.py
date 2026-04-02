"""Create an illustrative confounder-space panel for the main figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#A0C4FF"
ORANGE = "#FFADAD"
OUTPUT_DIR = Path("main_figure/output")


def rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D point cloud."""
    angle_rad = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return points @ rotation.T


def sample_blob(
    rng: np.random.Generator,
    mean: tuple[float, float],
    cov: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Draw a diffuse local cloud."""
    return rng.multivariate_normal(mean=mean, cov=cov, size=n_points)


def sample_curve(
    rng: np.random.Generator,
    n_points: int,
    *,
    x_shift: float,
    y_shift: float,
    angle_deg: float,
    x_scale: float,
    y_scale: float,
    phase: float,
) -> np.ndarray:
    """Draw an organic curved manifold to mimic an embedding."""
    t = rng.uniform(-2.5, 2.5, size=n_points)
    x = x_scale * t
    y = y_scale * (0.95 * np.sin(1.15 * t + phase) + 0.16 * t**2 - 0.55)
    points = np.column_stack([x, y])
    points += rng.normal(scale=[0.28, 0.23], size=points.shape)
    points = rotate(points, angle_deg=angle_deg)
    points += np.array([x_shift, y_shift])
    return points


def sample_control_group(rng: np.random.Generator) -> np.ndarray:
    """Build a structured but overlapping control embedding."""
    main_curve = sample_curve(
        rng,
        210,
        x_shift=-0.55,
        y_shift=-0.05,
        angle_deg=18,
        x_scale=0.82,
        y_scale=0.92,
        phase=0.25,
    )
    dense_blob = sample_blob(
        rng,
        mean=(-1.55, 0.95),
        cov=np.array([[0.11, 0.02], [0.02, 0.16]]),
        n_points=65,
    )
    overlap_blob = sample_blob(
        rng,
        mean=(0.10, -0.15),
        cov=np.array([[0.16, -0.03], [-0.03, 0.13]]),
        n_points=40,
    )
    return np.vstack([main_curve, dense_blob, overlap_blob])


def sample_treated_group(rng: np.random.Generator) -> np.ndarray:
    """Build a structured but overlapping treated embedding."""
    main_curve = sample_curve(
        rng,
        210,
        x_shift=0.55,
        y_shift=0.18,
        angle_deg=-16,
        x_scale=0.86,
        y_scale=0.88,
        phase=-0.55,
    )
    dense_blob = sample_blob(
        rng,
        mean=(1.65, 1.05),
        cov=np.array([[0.14, -0.03], [-0.03, 0.18]]),
        n_points=65,
    )
    overlap_blob = sample_blob(
        rng,
        mean=(-0.05, 0.10),
        cov=np.array([[0.18, 0.02], [0.02, 0.15]]),
        n_points=40,
    )
    return np.vstack([main_curve, dense_blob, overlap_blob])


def create_panel2_figure() -> plt.Figure:
    """Build the confounder-space scatter panel."""
    rng = np.random.default_rng(7)

    control = sample_control_group(rng)
    treated = sample_treated_group(rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.4), facecolor="white")
    ax.set_facecolor("white")

    ax.scatter(
        control[:, 0],
        control[:, 1],
        s=110,
        c=BLUE,
        alpha=0.96,
        edgecolors="white",
        linewidths=0.55,
        zorder=2,
    )
    ax.scatter(
        treated[:, 0],
        treated[:, 1],
        s=110,
        c=ORANGE,
        alpha=0.96,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )

    all_points = np.vstack([control, treated])
    x_pad = 0.75
    y_pad = 0.75
    ax.set_xlim(all_points[:, 0].min() - x_pad, all_points[:, 0].max() + x_pad)
    ax.set_ylim(all_points[:, 1].min() - y_pad, all_points[:, 1].max() + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.tight_layout(pad=0)
    return fig


def main() -> None:
    """Create and save the panel 2 figure assets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = create_panel2_figure()
    png_path = OUTPUT_DIR / "panel2_confounder_space.png"
    pdf_path = OUTPUT_DIR / "panel2_confounder_space.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
