"""Central configuration: paths, seeds, stage names and column groups."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_dropout.csv"
LABELS_PATH = ROOT / "data" / "data_dictionary.json"  # optional, see data/README.md
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METADATA_PATH = MODELS_DIR / "metadata.json"
SUMMARY_PATH = REPORTS_DIR / "summary.json"

TARGET = "Target"
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLASSES = ["Dropout", "Enrolled", "Graduate"]

# Integer-coded columns that are categories, not ordered numbers.
CATEGORICAL_COLUMNS = (
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",  # (sic) spelled this way in the original dataset
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
)

# National indicators that take only a few distinct combinations in the file.
# They behave like an enrollment-period (cohort) proxy, so we use them to build
# a group-wise validation check.
MACRO_COLUMNS = ("Unemployment rate", "Inflation rate", "GDP")

# Prediction moments. Each stage only uses information available at that time.
STAGES = {
    "enrollment": "At enrollment",
    "sem1": "After 1st semester",
    "sem2": "After 2nd semester",
}
