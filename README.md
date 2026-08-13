# Numerical simulations for the spiked-eigenvalue CLTs

This repository contains the Monte Carlo experiment accompanying the proof of
the outlier central limit theorems for mean-shift and covariance spikes.  The
experiment checks the two closed-form asymptotic variances against empirical
variances at increasing matrix dimensions.

## Formulas checked

Write $c=d/n$.  For a supercritical mean-shift spike with squared strength
$\theta^2$, the simulation uses

$$
\lambda_{\mathrm{ms}}=(1+\theta^2)\left(1+\frac{c}{\theta^2}\right),
\qquad
\sigma_{\mathrm{ms}}^2
=2(2\theta^2+1+c)\left(1-\frac{c}{\theta^4}\right).
$$

For a supercritical covariance spike $\ell$, it uses

$$
\lambda_{\mathrm{cov}}=(1+\ell)\left(1+\frac{c}{\ell}\right),
\qquad
\sigma_{\mathrm{cov}}^2
=2(1+\ell)^2\left(1-\frac{c}{\ell^2}\right).
$$

The experiment studies the centered and scaled quantities
$\sqrt n(\widetilde\lambda-\lambda_*)$.

## Reproducibility settings

- Random seed: `325042003`
- Replications per model and dimension: `1000`
- Dimensions: $n\in\{200,400,800\}$ and $d/n=0.5$
- Mean-shift parameter: $\theta^2=2$
- Covariance-spike parameter: $\ell=3$
- Noise: independent standard Gaussian entries, scaled by $n^{-1/2}$

The parameter choices are supercritical: $\theta^2>\sqrt c$ and
$\ell>\sqrt c$.

## Files

- `code/variance_simulation.py`: Monte Carlo simulation and Q-Q plot generator.
- `code/variance_simulation_results.csv`: numerical summary produced with the fixed
  seed and settings above.
- `../figures/variance_qq.png`: Q-Q diagnostics at $n=800$.

## Running the experiment

From the repository root, create an environment and install the pinned
dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python code/variance_simulation.py
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code\variance_simulation.py
```

The script overwrites `code/variance_simulation_results.csv` and writes
`figures/variance_qq.png`.

## Recorded results

| Model | $n$ | Empirical variance | Monte Carlo SE | Theoretical variance |
|---|---:|---:|---:|---:|
| Mean shift | 200 | 9.6244 | 0.4306 | 9.6250 |
| Mean shift | 400 | 9.4647 | 0.4235 | 9.6250 |
| Mean shift | 800 | 9.2010 | 0.4117 | 9.6250 |
| Covariance spike | 200 | 29.6040 | 1.3246 | 30.2222 |
| Covariance spike | 400 | 30.8920 | 1.3822 | 30.2222 |
| Covariance spike | 800 | 29.2426 | 1.3084 | 30.2222 |

The differences between the empirical and theoretical variances are of the
same order as the reported Monte Carlo standard errors.

