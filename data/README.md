# Data

`student_dropout.csv` is the UCI dataset **Predict Students' Dropout and Academic Success**
(dataset 697), unmodified. It is semicolon-separated and starts with a UTF-8 byte-order mark.

- Source: https://archive.ics.uci.edu/dataset/697
- License: CC BY 4.0. Attribution is required, see the citation below.
- Content: 4,424 students of a Portuguese higher-education institution (Instituto Politecnico de
  Portalegre), 36 features, three outcomes (Dropout, Enrolled, Graduate).

## Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
*Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5MC89

Describing paper: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022).
Predicting Student Dropout and Academic Success. *Data*, 7(11), 146.
https://doi.org/10.3390/data7110146

## Notes on the file

- One header ends with a stray tab (`Daytime/evening attendance`); the loader strips it.
- `Nacionality` is spelled that way in the original; we keep it.
- Many columns are integer **codes** for categories (course, application mode, qualifications,
  occupations, marital status). They are treated as categories, never as ordered numbers.
- Code meanings are documented by the data providers. This repository does not guess them.

## Optional: readable labels in the app

Create `data/data_dictionary.json` with labels copied from the UCI documentation, and the app will
show them in the form. Format:

```json
{
  "Marital status": {"1": "single", "2": "married"},
  "Gender": {"0": "label from the UCI documentation", "1": "label from the UCI documentation"}
}
```
