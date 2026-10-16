> Generated from synthetic data with `configs/demo.toml`.

# Synthetic randomized A/B analysis

> All data are synthetic. This is not a company production experiment or real product effect.
> The report does not automatically decide launch; human review is required.

## Randomization quality

- Data validation passed: True
- SRM passed: True
- Balance warnings: []

## Overall ITT effects

| Metric | Control | Treatment | Absolute effect | 95% CI | Raw p | Adjusted p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| next_day_retention | 0.304083 | 0.313153 | 0.009070 | [-0.002382, 0.020522] | 0.1206 | N/A | [-0.004764, 0.021378] |
| clicked | 0.198484 | 0.215415 | 0.016931 | [0.006887, 0.026976] | 0.00095189 | 0.00190378 | [0.008400, 0.027446] |
| watch_time_minutes | 18.309653 | 18.672576 | 0.362923 | [-0.175405, 0.901251] | 0.186378 | 0.186378 | [-0.170097, 0.911801] |
| negative_feedback | 0.041466 | 0.044736 | 0.003270 | [-0.001766, 0.008320] | 0.202929 | N/A | [-0.001499, 0.007895] |

## Guardrail

Status: **inconclusive** against the predeclared +0.005 absolute threshold.

## CUPED

CUPED uses only pre-experiment watch time and one theta estimated on all users.
- Theta: 0.069583
- Raw effect: 0.362923
- CUPED effect: 0.345974
- Raw/CUPED standard error: 0.274649 / 0.270646
- Variance reduction: 2.892%
- CUPED 95% CI: [-0.184508, 0.876455]
- CUPED p-value: 0.201146

## Predefined subgroup analysis

Only `is_new_user` is analyzed. Heterogeneity uses HC3 interaction tests, not a comparison of separate subgroup p-values.

| Metric | Existing-user effect | New-user effect | Interaction coefficient | Interaction p | Heterogeneity evidence |
|---|---:|---:|---:|---:|---:|
| next_day_retention | 0.005627 (n=16952) | 0.017390 (n=8048) | 0.011763 | 0.333992 | False |
| clicked | 0.021506 (n=16952) | 0.008032 (n=8048) | -0.013474 | 0.203798 | False |
| watch_time_minutes | 0.326641 (n=16952) | 0.489643 (n=8048) | 0.163002 | 0.771508 | False |
| negative_feedback | 0.003340 (n=16952) | 0.002655 (n=8048) | -0.000685 | 0.910742 | False |

## Decision support and limitations

- Primary status: positive_inconclusive
- Guardrail status: inconclusive
- Bootstrap is a sensitivity analysis and does not replace the primary tests.
- Subgroup analyses have additional multiplicity and do not trigger automatic decisions.
- Results cannot be generalized to real users or products.
- Open questions: ['Human review is required before any product decision.', 'The guardrail interval crosses the allowed-harm threshold.']
- `requires_human_review = true`
