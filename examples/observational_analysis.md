> Generated from synthetic data with `configs/demo.toml`.

# Observational IPTW analysis

> Synthetic data only. This is a portfolio study, not evidence from a real product or production system.

## Causal question

The estimand is the ATE. Treatment-minus-control effects are adjusted only for measured pre-treatment covariates.

## Propensity, overlap, and weights

- Model: unpenalized logistic regression (scikit-learn lbfgs)
- Features: is_new_user, pre_7d_sessions, pre_7d_watch_time_minutes, pre_7d_click_days, pre_experiment_activity_score, device_type, region
- Propensity range: 0.0758 to 0.9783
- Overall ESS: 20083.8
- Warnings: ['none']

## Balance

Maximum |SMD| before / IPTW / trimmed: 0.7374 / 0.0108 / 0.0498.

## Effect estimates

| Metric | Naive | IPTW ATE | 95% bootstrap CI | Trimmed IPTW |
|---|---:|---:|---:|---:|
| next_day_retention | 0.0808 | 0.0184 | [0.0084, 0.0277] | 0.0231 |
| clicked | 0.0739 | 0.0122 | [0.0030, 0.0213] | 0.0172 |
| watch_time_minutes | 4.2418 | 0.8593 | [0.2880, 1.5065] | 1.2742 |
| negative_feedback | -0.0128 | 0.0038 | [-0.0005, 0.0094] | 0.0034 |

## Assumptions and limitations

Causal interpretation requires consistency, no interference, positivity, correct propensity specification, and no unmeasured confounding. IPTW cannot remove unmeasured confounding. Weight trimming is a predeclared sensitivity analysis and may change the effective target population.

Human review is required; this report does not automate a launch decision.
