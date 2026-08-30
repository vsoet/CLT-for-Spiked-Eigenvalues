# Numerical simulations for the spiked-eigenvalue CLTs

This repository contains the Monte Carlo experiment accompanying the proof of
the outlier central limit theorems for mean-shift and covariance spikes.  The
experiment simulates both spikes simultaneously, assigns the two sample roots
by eigenvector overlap, and checks the centered and theoretically standardized
statistics with normal Q-Q plots.

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
- Well-separated configuration: $\theta^2=2$ and $\ell=3$
- Near-critical configuration: $\theta^2=0.9$ and $\ell=1.15$
- Noise: independent standard Gaussian entries, scaled by $n^{-1/2}$

All four spikes are supercritical.  The near-critical configuration is used to
show the slow, nonuniform finite-sample convergence as the phase boundary is
approached; it is not covered by a uniform-in-the-spike version of the CLTs.

## Files

- `code/clt_simulation.py`: Monte Carlo simulation and figure generator.
- `code/clt_simulation_results.csv`: numerical summary produced with the fixed seed.
- `figures/clt_qq_plot.png`: Q-Q diagnostics at $n=800$.

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
python code/clt_simulation.py
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code\clt_simulation.py
```

The script overwrites `code/clt_simulation_results.csv` and writes
`figures/clt_qq_plot.png`.

## Recorded results

The CSV records the scaled means, empirical variances, Monte Carlo standard
errors, theoretical variances, and the overlap-association diagnostics used to
interpret the Q-Q plots.  It intentionally contains no separate eigenvalue-
position analysis.

