# Model card: Student Support Radar

## Summary
Three LightGBM classifiers estimate whether a university student will **Drop out**, still be **Enrolled**, or **Graduate** at the end of the normal course duration. One model is used at each prediction moment: at enrollment, after semester 1 and after semester 2. Each only uses information available at that moment.

## Intended use
- **For:** helping academic advisors decide whom to contact first, so support can be offered earlier.
- **Not for:** admissions, grading, scholarship or funding decisions, discipline, or any automated decision about an individual student. Scores are estimates, not facts about a person.
- **Human in the loop:** a person must review any action taken from a score.

## Data
- UCI "Predict Students' Dropout and Academic Success" (dataset 697), CC BY 4.0. Realinho, Vieira Martins, Machado & Baptista (2021), https://doi.org/10.24432/C5MC89.
- 4,424 students from a single Portuguese higher-education institution (Instituto Politecnico de Portalegre), 36 features: academic path, demographics, socio-economic factors, national economic indicators, and first- and second-semester results.
- Outcomes: Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%).
- Many columns are integer codes (course, application mode, qualifications, occupations). They are treated as categories. This repository does not decode them.

## Training and evaluation
- Stratified 80/20 split with a fixed seed: 3,539 training and 885 test students. The test set is locked until all choices are made.
- Model choice and hyperparameters use 5-fold cross-validation on training data only (macro-F1). Candidates: majority-class baseline, logistic regression, random forest, LightGBM.
- Subgroup, calibration and mistake analyses use out-of-fold training predictions.
- The deployed models are refit on all 4,424 students using the chosen hyperparameters.

## Performance (locked test set, 885 students; 95% bootstrap intervals)
| Stage | Macro-F1 | Accuracy | Dropout recall | Top 20% flagged: dropouts reached (precision) |
|---|---|---|---|---|
| At enrollment | 0.547 (0.514-0.580) | 59.3% | 0.60 | 48% (77%) |
| After semester 1 | 0.667 (0.630-0.700) | 72.2% | 0.69 | 55% (88%) |
| After semester 2 | 0.706 (0.675-0.738) | 75.1% | 0.72 | 57% (92%) |

A majority-class guess scores macro-F1 0.222. "Enrolled" is the hardest outcome (F1 0.35, 0.45, 0.52 across the stages). Logistic regression and random forest score within about 0.015 of LightGBM on the test set (slightly ahead of it at enrollment), inside the noise of the test set. See `reports/error_analysis.md`.

## Known limitations
- **One institution, one country.** Performance elsewhere (other universities, countries, curricula) is untested and may be much worse.
- **Timing of "Tuition fees up to date" is unverified.** It is the strongest enrollment-stage feature. If it is recorded after a student stops paying, the enrollment-stage score overstates true early-warning ability.
- **Later stages are less useful for prevention.** The best scores need second-semester results, when many students have already left.
- **Cohort drift.** Holding out whole economic-period groups lowers macro-F1 by about 0.02 at each stage.
- **Uneven recall across groups** (enrollment stage): dropout recall is 0.37 for students aged 19 or younger versus 0.82-0.85 for those aged 25 and over, and 0.32 for scholarship holders versus 0.67 for others. Older students are also over-flagged.
- **Calibration** was checked only for top-label confidence (expected calibration error 0.018-0.030), not per class.
- The test set is small (885), so intervals are about +/-0.03 macro-F1.

## Sensitive attributes and fairness
The model uses age, gender (coded 0/1), nationality, marital status, special-needs, displaced and international flags as inputs. Subgroup performance is reported openly in the app and in the error analysis, but **fairness has not been established**. An ablation without these fields has not been run. Any real deployment would need a review of whether each field is appropriate and lawful to use, plus ongoing monitoring of outcomes by group.

## Explanations
Per-student explanations are TreeSHAP contributions from LightGBM (log-odds scale). They describe what the model used, not what caused a student's outcome.

## Privacy
The app code does not store uploaded files. Do not upload real student records to a public demo.

## Maintenance
Retrain with `python -m dropout.train`. Re-evaluate whenever the student population, courses or data collection change.
