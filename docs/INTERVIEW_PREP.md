# Interview prep: Student Support Radar

Short answers you can say aloud, grounded in this project's real numbers. Practice them in your own words; do not memorize.

## The 60-second pitch
"I built an early-warning tool for university dropout using a public dataset of 4,424 students. Instead of one model that uses everything, I trained one model per moment (at enrollment, after semester 1, after semester 2) to show how early dropout can really be seen. On a locked test set the macro-F1 goes from 0.55 to 0.67 to 0.71. I compared four models with cross-validation, did error analysis across subgroups, calibration and the most confident mistakes, and deployed it as a Streamlit app with per-student SHAP explanations, tests and CI."

## Questions and model answers

1. **Why three models instead of one?**
   A single model trained on all columns uses semester results that don't exist at enrollment, so its score can't be achieved when advisors need it. Staged models show the honest trade-off: later predictions are more accurate but less useful for prevention.

2. **Why macro-F1 and not accuracy?**
   Classes are imbalanced (about 50% Graduate, 32% Dropout, 18% Enrolled). Accuracy rewards guessing the majority (a dummy gets 49.9%). Macro-F1 averages the three outcomes equally, so the hard "Enrolled" class counts as much as the easy one.

3. **How did you avoid data leakage?**
   Stage columns are defined by when information exists. Preprocessing sits inside the model so it is fit only on training folds. Tuning and model selection used cross-validation on the training set only. The test set was evaluated once, after everything was frozen. I also held out whole economic-period groups to check cohort leakage; macro-F1 dropped about 0.02.

4. **What does the test-set interval tell you?**
   With 885 test students, the 95% interval on macro-F1 is about +/-0.03, so differences of 0.01-0.02 between models are noise.

5. **Why LightGBM if it barely beat logistic regression?**
   It scored a little higher in cross-validation and supports native categorical handling and per-student TreeSHAP explanations. I was upfront in the README that it is not decisively more accurate (it even lost slightly on the test set at enrollment).

6. **How did you handle categorical codes?**
   Course, qualifications, occupations and similar columns are integer codes without order. I treat them as categories (native in LightGBM, one-hot for the baselines) so the model does not invent a numeric ordering.

7. **What are SHAP values, and what are their limits?**
   They split a prediction into additive per-feature contributions (log-odds). I tested that contributions plus the bias reproduce the model's raw score. They describe what the model used, not what caused the student's outcome.

8. **What did the error analysis find?**
   "Enrolled" is hardest (F1 0.35 at enrollment). At enrollment, dropout recall is 0.37 for students 19 or younger but 0.82-0.85 for those 25 and over, and 0.32 for scholarship holders versus 0.67 for others. The most confident mistakes look like normal students, so the data lacks the real reasons people leave.

9. **What would you worry about before deploying this?**
   Single institution and country; unverified timing of "Tuition fees up to date" (it may be recorded after a student stops paying); uneven recall across groups; use of sensitive attributes; cohort drift. It must support human advisors, never replace them.

10. **Is the model calibrated?**
    Top-label confidence is reasonably calibrated (expected calibration error 0.018-0.030 out-of-fold). I did not verify per-class probabilities, and class weighting shifts raw probabilities, so I report probabilities as scores.

11. **How would you turn accuracy into something advisors can use?**
    By ranking. At enrollment, contacting the top 20% by predicted dropout risk reaches 48% of all eventual dropouts at 77% precision, about 2.4 times better than random.

12. **How did you test it?**
    43 tests: data quirks, nested stage columns, split reproducibility, SHAP additivity, handling of unseen categories, saved models matching golden predictions, a safe-retrain fallback when library versions differ, metric and capacity calculations, and Streamlit smoke tests for every page. CI runs lint and tests on each push.

13. **What would you do with another month?**
    Verify the fee-field timing with the data owners, run an ablation without sensitive attributes, inspect the 180 students with no enrolled units, add per-class calibration, and validate on a second institution if data were available.

14. **How would you monitor it in production?**
    Track input distribution drift, outcome rates per cohort, per-group recall once outcomes arrive, and retrain on new cohorts with the same locked-test protocol.

15. **What was the hardest part?**
    Resisting the high number. The model with second-semester data looks best, but the stage that matters for prevention is the weakest, and I chose to say that plainly.

## Honest CV bullets (edit to match your own work)
- Built and deployed a student dropout early-warning app (Python, scikit-learn, LightGBM, Streamlit) with stage-based models, cross-validated model selection, and a locked 885-student test set (macro-F1 0.55 at enrollment to 0.71 after semester 2).
- Performed error analysis across subgroups, calibration and top-confidence mistakes; found uneven recall by age and scholarship status and documented limitations in a model card.
- Added per-student SHAP explanations, 43 automated tests, Docker and GitHub Actions CI.
