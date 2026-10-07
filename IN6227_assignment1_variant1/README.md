# IN6227 Assignment 1 - Variant 1

Compare scikit-learn logistic regression and random forest classifiers on the supplied binary classification dataset. The target is `label` (`yes` is positive). No real-world label definition or data dictionary was supplied, so this project does not invent a business interpretation.



## Experimental decisions

1. Keep the supplied training/test split. Drop only rows with a missing target (three train, one test); never impute labels.
2. Retain all 15 input columns. Fit numeric median imputation and categorical mode imputation inside each cross-validation training fold. One-hot encode unordered categories, ignoring unseen categories at transform time. Preserve the literal category `Unknown`.
3. Standardize numeric features for logistic regression. Trees use unscaled numeric values. Keep IQR-flagged values and zeros because no domain rules establish that they are errors. Do not oversample or apply class weights; use the original prevalence and evaluate minority-class performance explicitly.
4. Use identical shuffled stratified three-fold splits for both models. Primary tuning metric: average precision (AP). Logistic regression: L2, LBFGS, C in {0.1, 1, 10}, tolerance 1e-4, maximum 2,000 iterations. Random forest: 200 bootstrap trees, Gini splits, sqrt feature subsampling, max_depth in {12, None}, min_samples_leaf in {1, 5}. Remaining parameters use scikit-learn 1.3.2 defaults, saved in the results JSON. Convergence warnings cause the experiment to fail.
5. Choose parameters by mean validation AP and refit on all labelled training rows. AP is the non-interpolated `average_precision_score`, not trapezoidal PR area. Fold SD is the sample SD, not a confidence interval; CV scores used for selection may be optimistic.
6. As a training-only diagnostic, remove `composite_rank` and repeat CV with the full-feature selected parameters and identical folds. Do not interpret this as proof about leakage or as a separately optimized feature-selection experiment. Final models retain all features.
7. Evaluate the fixed models once on the supplied labelled test rows. Use a prespecified 0.5 probability threshold. Report AP, ROC-AUC, accuracy, balanced accuracy, and positive-class precision, recall and F1. The all-no baseline is an arithmetic reference, not a third trained classifier.
8. Use 1,000 paired IID bootstrap resamples of test rows to estimate percentile 95% CIs for AP and the RF-minus-LR AP difference. These intervals are conditional on the fitted models and do not measure variability from retraining or shifts in the data source.

The data were inspected for basic quality in advance, including test summaries, but test results are not used to choose preprocessing, hyperparameters, features or thresholds. The experiment script reads test data only after all model configurations and the sensitivity check are complete. IID validation is an assumption because no entity identifiers, timestamps or provenance are supplied.

## Outputs

- `results/data_audit.json`: missingness, class counts, ranges, duplicates, correlations and category counts.
- `results/cv_results.csv`: all seven hyperparameter candidates and fold scores.
- `results/ablation.csv`: training-only feature sensitivity check.
- `results/results.json`: machine-readable results, parameters, timings, uncertainty, hashes and versions.
- `results/test_metrics.csv`: final metric table.
- `results/test_predictions.csv`: per-row probabilities and decisions, with source CSV row numbers.
- `results/comparison.png` / `.svg`: precision-recall curves and confusion matrices.


