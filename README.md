# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xbonson/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xbonson/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An empirical Twitch concurrent viewer (CCU) launch-day forecast engine for Grand Theft Auto VI, built for the make.fun `gta6-twitch-ccu-forecast-01` challenge.

## Model Overview

Rather than assuming an uninformative random walk expanding away from the first-hour count, this forecasting engine learns the dynamic shape of major game launches from historical release-day trajectories.

### Methodology

1. **Comparable Trajectory Pool**:
   Historical release-day time series are extracted from past major launches across hours 1 to 24 (excluding pre-release observation windows).

2. **Bivariate Similarity Weighting**:
   Each past launch is assigned an exponential similarity weight comparing both live channel volume and viewer count during hour 1:
   $$\text{weight} = \exp\left(-0.25 \cdot \left|\ln\frac{c_{\text{case}} + 1}{c_{\text{ref}} + 1}\right|\right) \times \exp\left(-1.0 \cdot \left|\ln\frac{v_{\text{case}} + 1}{v_{\text{ref}} + 1}\right|\right)$$
   This ensures that titles with comparable initial broadcaster scale and viewer footprint contribute most strongly.

3. **Relevance Weight Cutoff**:
   Comparables with similarity weights below `0.05` are dropped. This removes noisy trajectories from dissimilar launches.

4. **Temporal Window Smoothing**:
   For each target hour $h \in [2, 24]$, empirical log ratios $\ln(y_t / y_1)$ are aggregated across a window of $h \pm 3$ hours, capturing diurnal wave patterns and global unlock schedules.

5. **Tail Trimming**:
   The upper and lower 10% tails of the aggregated ratio distribution are trimmed to prevent idiosyncratic outliers from warping the confidence bands.

6. **Quantile Generation & Monotonicity Contract**:
   Weighted quantiles at $\tau \in [0.05, 0.25, 0.50, 0.75, 0.95]$ are scaled by hour-1 viewers, bounded within $[0.01, 10^9]$, rounded to two decimal places, and enforced to be monotonically non-decreasing. If input data is missing or malformed, the system falls back safely to the contest baseline distribution rather than failing.

## Offline Evaluation Results

Evaluated via strict leave-one-out cross-validation across the complete 37-launch comparable pool on the competition's pinball loss metric ($2 \times \max(\tau \cdot d, (\tau - 1) \cdot d)$ with $d = \ln(y) - \ln(q)$ across 4,255 scored items):

| Model | Mean Pinball Loss | Relative Gain |
|---|---|---|
| **House Baseline** | `0.929650` | baseline |
| **Unweighted Pooled Model** | `0.713418` | +23.26% |
| **Shipped Improved Model** | **`0.404328`** | **+56.51% vs baseline (+43.33% vs pooled)** |

- **Single Case Runtime**: ~33 ms per case in test harness (actual in-process compute ~3 ms), well within the 8-second sandbox limit.
- **Deterministic**: 100% reproducible, byte-identical output across repeated runs without network access.

## Running the Forecast

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

The program processes each `*.in` case in the input directory in alphabetical order and emits corresponding `<id>.out` forecast files.

## Running the Evaluation Suite

```sh
node tests/evaluate.ts
```

The test runner:
1. Validates schema conformity, row counts, monotonicity, and bounds across practice cases.
2. Executes leave-one-out cross-validation across all 37 historical comparable launches and prints the exact measured pinball loss.

<sub>Entered on [make.fun](https://make.fun/?ref=kewdhhyh&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
