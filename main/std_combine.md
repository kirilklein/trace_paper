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

### 2. Pooled RD per method/outcome (`combine_random_effects_HKSJ`)

`main/create_volcano_plot.py` aggregates `df_per_run` to method/outcome level using Hartung–Knapp–Sidik–Jonkman (HKSJ) random-effects meta-analysis:

1. **DerSimonian–Laird heterogeneity**
   - Compute fixed-effect weights \( w_i = 1/\mathrm{SE}_{RD,i}^2 \) to estimate the heterogeneity statistic \( Q \) and between-run variance \( \tau^2 \).

2. **Random-effects pooling**
   - Random-effects weights \( w_i^* = 1 / (\mathrm{SE}_{RD,i}^2 + \tau^2) \).
   - Pooled RD: \( \theta_{\text{RE}} = \sum w_i^* RD_i / \sum w_i^* \).

3. **HKSJ adjustment**
   - Adjusted variance:
     \[
     \mathrm{SE}_{\text{HK}}^2 = \frac{\sum w_i^*(RD_i - \theta_{\text{RE}})^2}{(m-1)\sum w_i^*}
     \]
     (falls back to classic RE variance if the denominator degenerates).
   - Inference uses a \(t\)-statistic with \( df = m-1 \); p-values and confidence intervals rely on the \(t\)-distribution. When only one run exists, the routine falls back to the original run’s z-based inference.

The pooled table (`df_pooled`) feeds the volcano plots (`RD`, `p_value`) and exposes diagnostics like `tau2` and `I2`.

### 3. Integration in `create_volcano_plot.py`

- Load arm-level estimates → filter to methods with arm CIs → call `compute_rd_pvalues` (per run) → call `combine_random_effects_HKSJ` (method/outcome).
- The resulting `df_pooled` powers diagnostics and plotting (`prepare_volcano_data`, `volcano_plot_per_method`, Plotly overlays, etc.).

This two-stage approach (logit-scale arm pooling, then random-effects pooling of RDs) stabilizes variance estimates near probability bounds and accounts for run-level heterogeneity before visualisation.