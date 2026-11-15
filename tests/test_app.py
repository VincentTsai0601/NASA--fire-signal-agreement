import pandas as pd
import pytest
from html.parser import HTMLParser
from streamlit.testing.v1 import AppTest
from pipeline import write_outputs

@pytest.fixture
def app_db(tmp_path,monkeypatch):
    weekly=pd.DataFrame({'week_start':['2023-01-02','2025-01-06'],'terra_count':[2,5],'aqua_count':[3,7],'viirs_count':[8,13]})
    weekly.attrs['metadata']={'schema_version':1,'bounds':{'south':37,'north':42,'west':-125,'east':-119}}
    predictions=weekly.iloc[[1]].assign(linear_estimate=[10],linear_gap=[3])
    path=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,{'linear':{'status':'available','mae':3,'signed_mean_error':-3},'xgboost':{'status':'unavailable','mae':None,'signed_mean_error':None}},path)
    monkeypatch.setenv('FIRE_DB',str(path))
    return path


def test_dashboard_loads_and_year_filter(app_db):
    app=AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert app.title[0].value=='Fire Signal Agreement'
    assert app.metric[0].value=='5'
    assert any('Verified answers' in item.label for item in app.expander)
    app.sidebar.selectbox[0].select(2023).run()
    assert not app.exception
    assert app.metric[0].value=='2'
    assert app.sidebar.selectbox[0].value==2023


def test_calendar_does_not_color_hidden_adjacent_month_dates(app_db):
    class Cells(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'td':
                self.cells[attrs['title']] = attrs.get('style', '')

    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    tables = [item.value for item in app.markdown if '<table' in item.value]
    january = Cells()
    january.feed(tables[0])
    assert 'background:' in january.cells['2025-01-06']
    # January's last Monday appears again as an empty February padding cell.
    # Give that week a real observation so its color would leak without a guard.
    import sqlite3
    with sqlite3.connect(app_db) as conn:
        conn.execute("INSERT INTO weekly VALUES ('2025-01-27', 1, 1, 20)")
    app.run()
    tables = [item.value for item in app.markdown if '<table' in item.value]
    february = Cells()
    february.feed(next(table for table in tables if '2025-02-01' in table and '2025-01-27' in table and '2025-01-06' not in table))
    assert 'background:' not in february.cells['2025-01-27']


def test_missing_database_explains_build_command(tmp_path,monkeypatch):
    monkeypatch.setenv('FIRE_DB',str(tmp_path/'missing.sqlite'))
    app=AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert 'Build the data first' in app.info[0].value


def test_analyst_unavailable_preserves_dashboard(app_db,monkeypatch):
    monkeypatch.delenv('VLLM_MODEL',raising=False)
    app=AppTest.from_file('app.py').run(timeout=30)
    app.text_input[0].input('Why do sensor counts differ?').run()
    app.button[0].click().run()
    assert not app.exception
    assert 'Analyst unavailable' in app.info[0].value
    assert app.metric[0].value=='5'
