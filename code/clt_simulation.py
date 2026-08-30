"""Reproducible simulations for the publication-layout MS-PCA CLT paper.

Two simultaneous mean/covariance-spike configurations are studied:

* a well-separated regime with two clearly resolved outliers;
* a near-critical regime with both spikes close to the BBP threshold.

The two roots are associated by their left-eigenvector overlaps with the
population covariance and mean directions.  This avoids building the
asymptotic eigenvalue ordering into the finite-sample diagnostics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEED = 325042003
REPLICATIONS = 1000
N_VALUES = (200, 400, 800)
C = 0.5

CASES = {
    "Well-separated": {"theta2": 2.0, "ell": 3.0},
    "Near-critical": {"theta2": 0.9, "ell": 1.15},
}

ASSOCIATION_WINDOW = 8

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def spike_location(x: float) -> float:
    return (1.0 + x) * (1.0 + C / x)


def mean_variance(theta2: float) -> float:
    return 2.0 * (2.0 * theta2 + 1.0 + C) * (1.0 - C / theta2**2)


def covariance_variance(ell: float) -> float:
    return 2.0 * (1.0 + ell) ** 2 * (1.0 - C / ell**2)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def simultaneous_draws(
    rng: np.random.Generator,
    n: int,
    theta2: float,
    ell: float,
) -> tuple[float, float]:
    """Return the two overlap-associated sample outliers."""

    d = int(round(C * n))
    noise = rng.standard_normal((d, n)) / math.sqrt(n)

    # Sigma^(1/2) = diag(sqrt(1+ell),1,...,1).
    noise[0, :] *= math.sqrt(1.0 + ell)

    mean_direction = rng.standard_normal(d)
    mean_direction /= np.linalg.norm(mean_direction)
    membership = np.zeros(n)
    membership[: n // 2] = 1.0 / math.sqrt(n // 2)
    data = noise + math.sqrt(theta2) * np.outer(mean_direction, membership)

    eigenvalues, eigenvectors = np.linalg.eigh(data @ data.T)
    candidate_count = min(ASSOCIATION_WINDOW, d)
    candidate_indices = np.arange(d - candidate_count, d)
    candidate_vectors = eigenvectors[:, candidate_indices]

    covariance_scores = candidate_vectors[0, :] ** 2
    mean_scores = (mean_direction @ candidate_vectors) ** 2

    # Jointly assign two distinct eigenvectors.  Normalization gives the two
    # population directions equal weight despite their different overlap
    # magnitudes near the transition.
    covariance_normalized = covariance_scores / max(
        float(np.max(covariance_scores)), np.finfo(float).eps
    )
    mean_normalized = mean_scores / max(
        float(np.max(mean_scores)), np.finfo(float).eps
    )
    assignment_scores = (
        covariance_normalized[:, None] + mean_normalized[None, :]
    )
    np.fill_diagonal(assignment_scores, -np.inf)
    covariance_local, mean_local = np.unravel_index(
        int(np.argmax(assignment_scores)), assignment_scores.shape
    )
    covariance_index = int(candidate_indices[covariance_local])
    mean_index = int(candidate_indices[mean_local])

    covariance_outlier = float(eigenvalues[covariance_index])
    mean_outlier = float(eigenvalues[mean_index])
    return mean_outlier, covariance_outlier


def summarize(
    raw_values: np.ndarray,
    n: int,
    center: float,
    theoretical_variance: float,
) -> dict[str, float]:
    scaled = math.sqrt(n) * (raw_values - center)
    empirical_variance = float(np.var(scaled, ddof=1))
    return {
        "scaled_mean": float(np.mean(scaled)),
        "mcse_scaled_mean": float(np.std(scaled, ddof=1) / math.sqrt(scaled.size)),
        "empirical_variance": empirical_variance,
        "mcse_variance": empirical_variance * math.sqrt(2.0 / (scaled.size - 1.0)),
        "theoretical_variance": theoretical_variance,
    }


def draw_qq(path: Path, panels: list[tuple[str, np.ndarray, float, float, int]]) -> None:
    width, height = 1800, 1520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    label_font = font(21)
    tick_font = font(17)
    margin_x, top, bottom = 120, 85, 95
    gap_x, gap_y = 110, 175
    panel_width = (width - 2 * margin_x - gap_x) // 2
    panel_height = (height - top - bottom - gap_y) // 2
    normal = NormalDist()
    axis_limit = 3.6

    for panel_index, (title, raw, center, variance, n) in enumerate(panels):
        row, col = divmod(panel_index, 2)
        left = margin_x + col * (panel_width + gap_x)
        panel_top = top + row * (panel_height + gap_y)
        right = left + panel_width
        panel_bottom = panel_top + panel_height
        probabilities = (np.arange(raw.size) + 0.5) / raw.size
        theoretical = np.array([normal.inv_cdf(float(p)) for p in probabilities])
        empirical = np.sort(math.sqrt(n) * (raw - center) / math.sqrt(variance))

        def px(x: float) -> float:
            return left + (x + axis_limit) / (2 * axis_limit) * panel_width

        def py(y: float) -> float:
            return panel_bottom - (y + axis_limit) / (2 * axis_limit) * panel_height

        draw.rectangle((left, panel_top, right, panel_bottom), outline="black", width=2)
        draw.line(
            (px(-axis_limit), py(-axis_limit), px(axis_limit), py(axis_limit)),
            fill="#bb2b2b",
            width=3,
        )
        for tick in (-3, -2, -1, 0, 1, 2, 3):
            draw.line((px(tick), panel_bottom, px(tick), panel_bottom + 7), fill="black", width=2)
            draw.line((left - 7, py(tick), left, py(tick)), fill="black", width=2)
            draw.text((px(tick) - 7, panel_bottom + 10), str(tick), fill="black", font=tick_font)
            draw.text((left - 34, py(tick) - 9), str(tick), fill="black", font=tick_font)
        for x, y in zip(theoretical, empirical):
            if -axis_limit <= x <= axis_limit and -axis_limit <= y <= axis_limit:
                draw.ellipse(
                    (px(float(x)) - 2, py(float(y)) - 2, px(float(x)) + 2, py(float(y)) + 2),
                    fill="#1f5aa6",
                )
        draw.text((left + 6, panel_top - 48), title, fill="black", font=title_font)
        draw.text(
            (left + panel_width // 2 - 85, panel_bottom + 42),
            "Normal quantile",
            fill="black",
            font=label_font,
        )

    image.save(path, dpi=(200, 200))


def main() -> None:
    seed_sequence = np.random.SeedSequence(SEED)
    child_sequences = seed_sequence.spawn(len(CASES) * len(N_VALUES))
    rows: list[dict[str, object]] = []
    largest_samples: dict[tuple[str, str], np.ndarray] = {}
    child_index = 0

    for case, parameters in CASES.items():
        theta2 = parameters["theta2"]
        ell = parameters["ell"]
        mean_center = spike_location(theta2)
        covariance_center = spike_location(ell)
        mean_theoretical_variance = mean_variance(theta2)
        covariance_theoretical_variance = covariance_variance(ell)

        for n in N_VALUES:
            rng = np.random.default_rng(child_sequences[child_index])
            child_index += 1
            draws = np.array(
                [simultaneous_draws(rng, n, theta2, ell) for _ in range(REPLICATIONS)]
            )
            mean_values = draws[:, 0]
            covariance_values = draws[:, 1]

            for model, values, spike, center, variance in (
                (
                    "Mean shift",
                    mean_values,
                    theta2,
                    mean_center,
                    mean_theoretical_variance,
                ),
                (
                    "Covariance spike",
                    covariance_values,
                    ell,
                    covariance_center,
                    covariance_theoretical_variance,
                ),
            ):
                row: dict[str, object] = {
                    "case": case,
                    "model": model,
                    "n": n,
                    "d": int(round(C * n)),
                    "spike": spike,
                    "center": center,
                    "association_method": (
                        f"maximum normalized overlap among top {ASSOCIATION_WINDOW}"
                    ),
                }
                row.update(summarize(values, n, center, variance))
                rows.append(row)

            if n == N_VALUES[-1]:
                largest_samples[(case, "Mean shift")] = mean_values
                largest_samples[(case, "Covariance spike")] = covariance_values

    csv_path = ROOT / "code" / "clt_simulation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    qq_panels = []
    for case, model in (
        ("Well-separated", "Mean shift"),
        ("Well-separated", "Covariance spike"),
        ("Near-critical", "Mean shift"),
        ("Near-critical", "Covariance spike"),
    ):
        parameters = CASES[case]
        if model == "Mean shift":
            center = spike_location(parameters["theta2"])
            variance = mean_variance(parameters["theta2"])
        else:
            center = spike_location(parameters["ell"])
            variance = covariance_variance(parameters["ell"])
        qq_panels.append(
            (
                f"{case}: {model.lower()}, n={N_VALUES[-1]}",
                largest_samples[(case, model)],
                center,
                variance,
                N_VALUES[-1],
            )
        )

    draw_qq(FIGURE_DIR / "clt_qq_plot.png", qq_panels)

    for row in rows:
        print(", ".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()

