# Stabilized IPTW

For treated users the stabilized weight is `P(T=1)/e(X)`; for controls it is `P(T=0)/(1-e(X))`. The ATE is the difference between separately normalized (Hájek) weighted outcome means. ESS is `(sum(w)^2)/sum(w^2)` and exposes information loss from variable weights.

Balance uses weighted group means and weighted population variances. Absolute SMD below 0.10 is a diagnostic convention, not proof of exchangeability. The untrimmed estimate is primary. Clipping weights at the predeclared 1st/99th percentiles is sensitivity analysis and can change the effective target population.

The user-level bootstrap refits preprocessing and the propensity model on every resample. Holding fitted scores fixed would omit propensity-estimation uncertainty.
