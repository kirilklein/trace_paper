# Std Combination

## Risk-Difference Inference Workflow

This section explains how the volcano-plot pipeline computes risk differences (RDs), their standard errors, and p-values. The implementation lives in `main/create_volcano_plot.py` and `trace/statistics.py`.

### 1. Per-run RD calculation (`compute_rd_pvalues`)

1. **Logit transformation (arm level)**
   - `add_logit_arm_metrics` converts the observed arm probabilities (`effect_1`, `effect_0`) and their 95 % confidence intervals into the logit scale.
   - The arm-level standard error on the logit scale is derived from the CI width:
     \[
     \text{SE}_{\text{logit}} = \frac{\text{logit}(\mathrm{upper}) - \text{logit}(\mathrm{lower})}{2z}, \quad z = 1.96
     \]
2. **Delta-method back-transform (probability scale)**
   - `rd_inference_from_arm_logits` maps logits back to probabilities via `inv_logit`, obtains \( p_1, p_0 \), and forms \( RD = p_1 - p_0 \).
   - The RD variance is propagated using the delta method:
     \[
     \mathrm{Var}(RD) = [p_1(1-p_1)]^2 \mathrm{SE}(\eta_1)^2 + [p_0(1-p_0)]^2 \mathrm{SE}(\eta_0)^2
     \]
     Standard error is \( \mathrm{SE}_{RD} = \sqrt{\mathrm{Var}(RD)} \).
   - Wald statistics: \( z = RD / \mathrm{SE}_{RD} \); p-values use a two-sided normal test; RD confidence bounds come from \( RD \pm z_{0.975}\cdot \mathrm{SE}_{RD} \).

The output (`df_per_run`) contains per-run RDs, SEs, z-scores, p-values, and pooled arm probabilities.

### 2. Pooled RD per method/outcome (logit-first HKSJ)

`main/create_volcano_plot.py` now calls `compute_rd_pvalues(..., group_cols=("method", "outcome"))`, which performs pooling entirely on the **logit** scale before mapping back to probabilities:

1. **Per-arm random-effects pooling**
   - For each arm, `pool_arm_logits(pooling="random_effects_hksj")` applies DerSimonian–Laird to estimate \( \tau^2 \) and uses Hartung–Knapp–Sidik–Jonkman (HKSJ) to obtain a pooled logit mean \( \hat\eta_a \) with standard error \( \mathrm{SE}_{\hat\eta_a} \).

2. **Logit-difference inference**
   - Combine the pooled arm logits to form \( \Delta_\eta = \hat\eta_1 - \hat\eta_0 \) with variance
     \(
       \mathrm{SE}_{\Delta_\eta}^2 = \mathrm{SE}_{\hat\eta_1}^2 + \mathrm{SE}_{\hat\eta_0}^2
     \) (conservative independence assumption).
   - Conduct a Wald t-test (df = m − 1 when m ≥ 2) for \( H_0: \Delta_\eta = 0 \). The resulting `p_value` and `z`/`t_logit` entries are **logit-scale**.

3. **Probability-scale reporting**
   - Apply `inv_logit` to obtain \( \hat p_1, \hat p_0 \) and produce RD and delta-method SE/CI for display (`RD`, `SE_RD`, `RD_CI95_*`).
   - Additional diagnostics include `eta_diff`, `se_eta_diff`, `p_value_logit`, and arm-specific heterogeneity (`eta1_tau2`, `eta0_tau2`).

The pooled table (`df_pooled`) still feeds the volcano plots (`RD`, `p_value`), but the p-values now reflect the logit-scale random-effects test.

### 3. Integration in `create_volcano_plot.py`

- Load arm-level estimates → filter to methods with arm CIs → `compute_rd_pvalues(..., group_cols=None)` for per-run diagnostics → `compute_rd_pvalues(..., group_cols=("method", "outcome"))` for pooled inference.
- The resulting `df_pooled` powers diagnostics and plotting (`prepare_volcano_data`, `volcano_plot_per_method`, Plotly overlays, etc.) while exposing both logit- and probability-scale summaries.

This logit-first approach keeps the modelling assumptions on the logit scale, yields conservative p-values when arm estimates are positively correlated, and still presents RD/RR metrics that are easy to interpret on the probability scale.