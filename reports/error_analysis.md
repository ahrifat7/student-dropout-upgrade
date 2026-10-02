## Error analysis: what the numbers hide

*Method: subgroup, calibration and mistake analyses use out-of-fold predictions on the 3,539 training students, so the locked test set stays untouched. Headline scores come from the 885-student test set. Small subgroups are noisy and no multiple-comparison correction was applied, so treat gaps as leads to investigate, not proof.*

### 1. More information helps, but "Enrolled" stays hard
On the locked test set, macro-F1 rises from **0.547** (at enrollment) to **0.667** (after semester 1) to **0.706** (after semester 2). "Enrolled" is the weakest outcome at every stage (F1 0.35, 0.45, 0.52). That is expected: it means "still studying at the end of the normal duration", so it overlaps with both dropping out later and graduating late. It is not a distinct kind of student.

### 2. The early model is a usable triage tool, not a classifier
Overall accuracy at enrollment is only 59%. But ranking students by P(Dropout) works much better than the headline suggests: if advisors can contact the top 20%, the enrollment model reaches **48% of all eventual dropouts with 77% precision** (2.4x better than random). After semester 1, the same effort reaches 55% of dropouts with 88% precision. The right question for advisors is "who do we call first?", not "is this student a dropout?".

### 3. Who does the model miss or over-flag?
At the enrollment stage:

- **Young students are under-flagged.** Students aged 19 or younger have a 20.7% dropout rate but are predicted to drop out only 13.8% of the time (dropout recall 0.37).
- **Older students are over-flagged.** Students aged 25-39 drop out about 56-58% of the time but are predicted to at about 68-69%, which means many false alarms. Age is acting as a strong proxy; check whether it should be an input at all.
- **Scholarship holders who drop out are missed.** Dropout recall is 0.32 for scholarship holders versus 0.67 for others, and it only improves to 0.44 after semester 1. The group is protected by a low base rate (12.9%), but the students who do leave get little warning.
- **Courses differ.** Macro-F1 is lowest for courses 9119, 9130 and 9991 (0.43 to 0.46) and the model over-predicts dropout for 9130 and 9991 (64% and 62% predicted versus 56% and 52% actual). Group sizes are 115-217, so this is suggestive only. Course codes are not decoded in this repository.

### 4. One feature dominates, and it needs a question
"Tuition fees up to date" is the top feature at enrollment (mean |SHAP| 0.40) and second after semester 1. The 419 students with fees *not* up to date drop out 87% of the time and the model finds 98% of them. The dataset documentation lists this as known at enrollment, but a payment-status flag could also be recorded *after* a student stops paying, in which case it is a consequence of leaving rather than an early cause. **Verify with the data providers when the field is recorded** before treating the enrollment-stage numbers as a true early warning.

### 5. The most confident mistakes look like normal students
The 20 most confident errors per stage have confidence of 0.92-0.99. After semester 1, 10 of the 20 are students predicted to graduate who dropped out, and across all 20 the students had completed about 5 of 6 enrolled units on average, so their records look like those of successful students. At enrollment, the confident errors include young scholarship holders with fees paid who dropped out, and students with unpaid fees and debt who were still enrolled or graduated. The data cannot see why people leave (health, family, money, job offers), so some errors are unavoidable with these columns.

### 6. Calibration and cohort robustness
- **Calibration (top-label confidence):** expected calibration error is 0.030, 0.022 and 0.018 for the three stages, so "90% confident" is roughly right about 90% of the time in aggregate. I did not check per-class probability calibration, and class weighting shifts the raw probabilities.
- **Cohort robustness:** holding out whole economic-period groups (a proxy for enrollment year) lowers macro-F1 by about 0.02 at every stage (0.565 vs 0.588, 0.678 vs 0.702, 0.712 vs 0.730). The effect is small but consistent, so a model trained on past cohorts should be expected to score slightly worse on a new cohort.

### 7. Model choice: boosting is not clearly better
Cross-validated macro-F1 is higher for LightGBM at all three stages, but only by about 0.015-0.03, which is comparable to the fold-to-fold variation (standard deviation 0.015-0.026). On the test set the three models differ by less than the width of the test set's confidence interval (about +/-0.03), and the order even flips at enrollment (logistic regression 0.553, random forest 0.557, LightGBM 0.547). LightGBM was kept mainly because it enables per-student explanations, not because it is decisively more accurate.

### What I would do next
1. Confirm the timing of "Tuition fees up to date" and re-run the enrollment model without it.
2. Run an ablation without age, gender, nationality, marital status and special-needs fields, and compare subgroup recall.
3. Inspect the 180 students with zero enrolled units in both semesters.
4. Collect data from more than one institution before claiming the approach generalizes.
