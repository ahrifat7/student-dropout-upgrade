import pytest
from streamlit.testing.v1 import AppTest

from dropout.config import METADATA_PATH, ROOT, SUMMARY_PATH

pytestmark = pytest.mark.skipif(
    not (METADATA_PATH.exists() and SUMMARY_PATH.exists()),
    reason="run `python -m dropout.train` first",
)

PAGES = ["Overview", "Predict a student", "Score a cohort", "Model comparison", "Error analysis", "About and limits"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_prediction_changes_when_inputs_change():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    app.sidebar.radio[0].set_value("Predict a student").run()
    app.radio(key="predict-stage").set_value("sem2").run()
    assert not app.exception
    baseline = app.metric[0].value
    app.number_input(key="sem2-Curricular units 2nd sem (approved)").set_value(0).run()
    app.number_input(key="sem2-Curricular units 1st sem (approved)").set_value(0).run()
    assert not app.exception
    assert baseline in {"Dropout", "Enrolled", "Graduate"}
    assert app.metric[0].value == "Dropout"
