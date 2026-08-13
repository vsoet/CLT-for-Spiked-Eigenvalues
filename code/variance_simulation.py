Exit code: 0
Wall time: 2.9 seconds
Output:
"""Monte Carlo check for the two closed outlier-variance formulas.

The simulation uses NumPy only for the random-matrix calculation and Pillow
for a compact Q-Q diagnostic.  The random seed and every parameter appearing
in the manuscript are fixed below.
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
THETA2 = 2.0
ELL = 3.0

MEAN_CENTER = (1.0 + THETA2) * (1.0 + C / THETA2)
COV_CENTER = (1.0 + ELL) * (1.0 + C / ELL)
MEAN_VARIANCE = 2.0 * (THETA2**2 - C) * (2.0 * THETA2 + 1.0 + C) / THETA2**2
COV_VARIANCE = 2.0 * (1.0 + ELL) ** 2 * (1.0 - C / ELL**2)

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def largest_eigenvalue(data: np.ndarray) -> float:
    covariance = data @ data.T
    return float(np.linalg.eigvalsh(covariance)[-1])


def mean_shift_draw(rng: np.random.Generator, n: int) -> float:
    d = int(round(C * n))
    z = rng.standard_normal((d, n)) / math.sqrt(n)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    membership_size = n // 2
    w = np.zeros(n)
    w[:membership_size] = 1.0 / math.sqrt(membership_size)
    data = z + math.sqrt(THETA2) * np.outer(v, w)
    return math.sqrt(n) * (largest_eigenvalue(data) - MEAN_CENTER)


def covariance_draw(rng: np.random.Generator, n: int) -> float:
    d = int(round(C * n))
    data = rng.standard_normal((d, n)) / math.sqrt(n)
    data[0, :] *= math.sqrt(1.0 + ELL)
    return math.sqrt(n) * (largest_eigenvalue(data) - COV_CENTER)


def summarize(values: np.ndarray, theoretical_variance: float) -> dict[str, float]:
    empirical_mean = float(np.mean(values))
    empirical_variance = float(np.var(values, ddof=1))
    return {
        "empirical_mean": empirical_mean,
        "mcse_mean": float(np.std(values, ddof=1) / math.sqrt(values.size)),
        "empirical_variance": empirical_variance,
        "mcse_variance": empirical_variance * math.sqrt(2.0 / (values.size - 1.0)),
        "theoretical_variance": theoretical_variance,
    }


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_qq(path: Path, panels: list[tuple[str, np.ndarray, float]]) -> None:
    width, height = 1600, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(30)
    label_font = font(23)
    tick_font = font(18)
    margin_x, top, bottom = 105, 85, 95
    gap = 95
    panel_width = (width - 2 * margin_x - gap) // 2
    panel_height = height - top - bottom
    normal = NormalDist()
    axis_limit = 3.6

    for panel_index, (title, values, theoretical_variance) in enumerate(panels):
        left = margin_x + panel_index * (panel_width + gap)
        right = left + panel_width
        panel_bottom = top + panel_height
        probabilities = (np.arange(values.size) + 0.5) / values.size
        theoretical = np.array([normal.inv_cdf(float(p)) for p in probabilities])
        empirical = np.sort(values / math.sqrt(theoretical_variance))

        def px(x: float) -> float:
            return left + (x + axis_limit) / (2 * axis_limit) * panel_width

        def py(y: float) -> float:
            return panel_bottom - (y + axis_limit) / (2 * axis_limit) * panel_height

        draw.rectangle((left, top, right, panel_bottom), outline="black", width=2)
        draw.line((px(-axis_limit), py(-axis_limit), px(axis_limit), py(axis_limit)), fill="#cc3333", width=3)
        for tick in (-3, -2, -1, 0, 1, 2, 3):
            draw.line((px(tick), panel_bottom, px(tick), panel_bottom + 8), fill="black", width=2)
            draw.line((left - 8, py(tick), left, py(tick)), fill="black", width=2)
            draw.text((px(tick) - 8, panel_bottom + 12), str(tick), fill="black", font=tick_font)
            draw.text((left - 35, py(tick) - 10), str(tick), fill="black", font=tick_font)
        for x, y in zip(theoretical, empirical):
            if -axis_limit <= x <= axis_limit and -axis_limit <= y <= axis_limit:
                draw.ellipse((px(float(x)) - 2, py(float(y)) - 2, px(float(x)) + 2, py(float(y)) + 2), fill="#1f5aa6")
        draw.text((left + 10, 28), title, fill="black", font=title_font)
        draw.text((left + panel_width // 2 - 95, height - 50), "Normal quantile", fill="black", font=label_font)

    draw.text((18, height // 2 - 70), "Scaled empirical", fill="black", font=label_font)
    draw.text((24, height // 2 - 38), "quantile", fill="black", font=label_font)
    image.save(path, dpi=(180, 180))


def main() -> None:
    seed_sequence = np.random.SeedSequence(SEED)
    child_sequences = seed_sequence.spawn(2 * len(N_VALUES))
    rows: list[dict[str, object]] = []
    largest_n_samples: dict[str, np.ndarray] = {}

    for index, n in enumerate(N_VALUES):
        mean_rng = np.random.default_rng(child_sequences[2 * index])
        cov_rng = np.random.default_rng(child_sequences[2 * index + 1])
        mean_values = np.array([mean_shift_draw(mean_rng, n) for _ in range(REPLICATIONS)])
        cov_values = np.array([covariance_draw(cov_rng, n) for _ in range(REPLICATIONS)])

        for model, values, center, variance in (
            ("Mean shift", mean_values, MEAN_CENTER, MEAN_VARIANCE),
            ("Covariance spike", cov_values, COV_CENTER, COV_VARIANCE),
        ):
            row: dict[str, object] = {"model": model, "n": n, "d": int(round(C * n)), "center": center}
            row.update(summarize(values, variance))
            rows.append(row)
        if n == N_VALUES[-1]:
            largest_n_samples["Mean shift"] = mean_values
            largest_n_samples["Covariance spike"] = cov_values

    csv_path = ROOT / "code" / "variance_simulation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    draw_qq(
        FIGURE_DIR / "variance_qq.png",
        [
            (f"Mean shift, n={N_VALUES[-1]}", largest_n_samples["Mean shift"], MEAN_VARIANCE),
            (f"Covariance spike, n={N_VALUES[-1]}", largest_n_samples["Covariance spike"], COV_VARIANCE),
        ],
    )

    for row in rows:
        print(", ".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()

