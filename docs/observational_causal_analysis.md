# Observational causal analysis

Unlike randomized assignment, observational treatment can share causes with outcomes. Stage 5 estimates an ATE with stabilized IPTW using measured pre-treatment covariates. Consistency, no interference, positivity, correct propensity specification, and no unmeasured confounding are required.

IPTW cannot solve unmeasured confounding. Formal analysis reads only the observational user-level mart and never reads synthetic truth. Truth is used only by a separate validation command. All data are synthetic and this portfolio project is not a production causal-inference system.
