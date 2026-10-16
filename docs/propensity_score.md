# Propensity score

The observational analysis estimates `P(treatment=1 | X)` with one logistic regression configured with `C=np.inf` (no regularization). `X` is a fixed whitelist of pre-treatment variables; numeric variables are standardized and device/region are one-hot encoded. Outcomes, treatment-derived fields, assignment probability, and synthetic truth are forbidden.

AUC is not the main criterion: prediction and covariate balance are different goals. We assess overlap, positivity, weight stability, ESS, and post-weighting SMD. Causal interpretation additionally requires correct model specification and no unmeasured confounding.
