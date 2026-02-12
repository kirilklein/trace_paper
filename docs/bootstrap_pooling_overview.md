# Bootstrap Pooling: Current Approach vs Simplified Approach

## Context

Each "run" is a bootstrap resample. Within each run we observe arm-level
probability estimates (p1, p0) with 95% CIs. We want to combine these across
B bootstrap replicates to get a pooled Risk Difference (RD = p1 - p0) with
SE, CI, and p-value.

---

## Current Approach

The pipeline in `trace/statistics.py` has three stages:

### Stage 1 — Logit transform

Each run's probability estimates and CIs are transformed to the logit scale.
Within-run SEs are derived from the CI widths:

```
eta = logit(p)
SE_eta = (logit(CI_upper) - logit(CI_lower)) / (2 * 1.96)
```

### Stage 2 — Pool across runs on the logit scale

Four methods are available. All take `yi` (per-run logit estimates) and
`vi = SE_eta^2` (per-run variances):

| Method             | SE formula                    | Uses within-run SE? |
|--------------------|-------------------------------|---------------------|
| **Rubin's Rules**  | `sqrt(V + (1 + 1/m) * B)`    | Yes                 |
| **Inter-Intra**    | `sqrt((V + B) / m)`          | Yes                 |
| **Simple Mean**    | `std(yi) / sqrt(m)`          | No                  |
| **HKSJ**           | Random-effects meta-analysis  | Yes (IV weights)    |

Where V = mean within-run variance, B = between-run variance, m = number of runs.

### Stage 3 — Back-transform via delta method

Pooled logits are converted to RD with variance propagation:

```
p1 = inv_logit(eta1_pooled)
p0 = inv_logit(eta0_pooled)
RD = p1 - p0
SE_RD = sqrt((p1*(1-p1))^2 * SE_eta1^2 + (p0*(1-p0))^2 * SE_eta0^2)
```

P-values use a t-test on the logit-scale difference with df = m - 1.

---

## Why this is wrong for bootstrap

### 1. Within-run SEs are redundant

In bootstrap, variation across resamples **directly estimates the sampling
distribution**. The within-run CIs are asymptotic approximations that duplicate
what the bootstrap already captures. Rubin's Rules, Inter-Intra, and HKSJ all
incorporate this redundant information.

### 2. Division by sqrt(m) is wrong

`simple_mean` computes `std(yi) / sqrt(m)`. This is the SE of the *mean of
bootstrap replicates*, not the SE of the *estimator*. As B grows, this
converges to zero — which is nonsensical. More bootstrap replicates should
improve precision of the SE estimate, not make results more significant.

The bootstrap SE of an estimator is `std(theta*_b)`, not `std(theta*_b) / sqrt(B)`.
The `1/sqrt(B)` term only quantifies Monte Carlo error from using finite B.

The same issue applies to `inter_intra_variance` which divides by m.

### 3. Meta-analysis methods don't apply

HKSJ and Rubin's Rules treat runs as independent studies or independent
imputations. Bootstrap resamples are neither. The heterogeneity parameter tau^2
and the finite-imputation correction (1 + 1/m) have no meaningful
interpretation in the bootstrap setting.

### 4. The logit roundtrip is unnecessary

With bootstrap, compute the final statistic directly per replicate. No need for
logit transform, pooling on the logit scale, then delta-method back-transform.

### Impact on test statistics

Current `simple_mean` test statistic (the least wrong of the four):

```
t = mean(yi) / (std(yi) / sqrt(B))  =  mean(yi) * sqrt(B) / std(yi)
```

Correct bootstrap test statistic:

```
t = mean(yi) / std(yi)
```

The current code inflates the test statistic by sqrt(B). With B = 10 that's a
3.16x inflation; with B = 20 that's 4.47x. Everything looks far more
significant than it should.

---

## Simplified Bootstrap Approach

For each outcome, across bootstrap replicates b = 1, ..., B:

```python
# 1. Compute RD directly per replicate (no logit transform needed)
rd_b = p1_b - p0_b

# 2. Point estimate
theta = np.mean(rd_b)

# 3. SE = standard deviation of replicates (NOT divided by sqrt(B))
se = np.std(rd_b, ddof=1)

# 4. Inference with t-distribution (df = B - 1)
df = B - 1
t_stat = theta / se
p_value = 2 * (1 - tdist.cdf(abs(t_stat), df=df))

# 5. Confidence interval
t_crit = tdist.ppf(0.975, df=df)
ci_lo = theta - t_crit * se
ci_hi = theta + t_crit * se
```

### What this eliminates

| Removed step               | Why it's unnecessary                               |
|----------------------------|-----------------------------------------------------|
| Within-run CIs / SEs       | Bootstrap variation already captures uncertainty    |
| Logit transformation        | Pool directly on the scale of interest              |
| Delta method back-transform | RD is computed directly per replicate               |
| Rubin's Rules / HKSJ / etc. | These are for MI or meta-analysis, not bootstrap    |
| Percentile CIs (small B)   | Need B >> 100; use t-based CIs instead              |

### If you still want the logit-scale test

```python
logit_diff_b = logit(p1_b) - logit(p0_b)
theta = np.mean(logit_diff_b)
se = np.std(logit_diff_b, ddof=1)    # NOT divided by sqrt(B)
t_stat = theta / se
p_value = 2 * (1 - tdist.cdf(abs(t_stat), df=B-1))
```

Same principle: `std`, not `std / sqrt(B)`.

---

## Effect of B on inference

### B = 10

- **df = 9**, t critical value = **2.26** (vs 1.96 for normal).
- SE estimate has ~24% relative error (`1 / sqrt(2 * (B-1))`).
- CIs are about 15% wider than with normal approximation, correctly reflecting
  the imprecise SE estimate.
- Percentile CIs are not viable (2.5th percentile ≈ the minimum of 10 values).
- Smallest achievable p-value depends on the effect size relative to SE, but
  the t(9) distribution has heavier tails so extreme p-values are harder to
  reach.

### B = 20

- **df = 19**, t critical value = **2.09** (closer to 1.96).
- SE estimate has ~16% relative error — meaningfully better than B = 10.
- CIs are about 7% wider than normal — the t-correction matters less.
- The t(19) distribution is close to normal, so normal-approximation CIs would
  also be acceptable.
- Percentile CIs are still unreliable (2.5th percentile ≈ the smallest or
  second-smallest value).
- Overall: a useful improvement in SE precision over B = 10, at 2x
  computational cost.

### Summary table

| Property                 | B = 10      | B = 20      | B = 200     |
|--------------------------|-------------|-------------|-------------|
| df                       | 9           | 19          | 199         |
| t critical value (95%)   | 2.26        | 2.09        | 1.97        |
| Relative error of SE     | ~24%        | ~16%        | ~5%         |
| CI width vs normal       | +15%        | +7%         | +0.5%       |
| Percentile CIs viable?   | No          | No          | Yes         |
| t vs normal matters?     | Yes         | Somewhat    | Negligible  |

### Recommendation

B = 20 is a reasonable pragmatic choice if computation is expensive. The t(19)
distribution provides adequate correction, and the SE estimate is precise enough
for most purposes. If computation allows, B >= 200 enables percentile CIs and
makes the choice of reference distribution irrelevant.

---

## Side-by-side comparison

```
CURRENT (e.g. simple_mean)          BOOTSTRAP (simplified)
─────────────────────────           ──────────────────────
logit(p1), logit(p0)                rd = p1 - p0
    │                                   │
derive SE from CIs                  (nothing else needed)
    │                                   │
pool: mean(eta)                     theta = mean(rd_b)
SE = std(eta) / sqrt(B)  ← WRONG   SE = std(rd_b)  ← CORRECT
    │                                   │
delta method → RD, SE_RD            (already on RD scale)
    │                                   │
t-test, df = B-1                    t-test, df = B-1
```
