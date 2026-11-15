import pandas as pd
import pytest
from html.parser import HTMLParser
from streamlit.testing.v1 import AppTest
from pipeline import calibrate, write_outputs

@pytest.fixture
def app_db(tmp_path,monkeypatch):
    weekly=pd.DataFrame({'week_start':['2023-01-02','2025-01-06'],'terra_count':[2,5],'aqua_count':[3,7],'viirs_count':[8,13]})
    weekly.attrs['metadata']={'schema_version':1,'bounds':{'south':37,'north':42,'west':-125,'east':-119}}
    predictions=weekly.iloc[[1]].assign(linear_estimate=[10],linear_gap=[3])
    path=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,{'linear':{'status':'available','mae':3,'signed_mean_error':-3},'xgboost':{'status':'unavailable','mae':None,'signed_mean_error':None}},path,harmonized=calibrate(weekly))
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
    app.selectbox(key='calendar_layer').select('Observed VIIRS').run()
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


def test_calibrated_calendar_selection_provenance_and_shared_year_scale(app_db):
    import sqlite3
    with sqlite3.connect(app_db) as conn:
        conn.execute('UPDATE harmonized SET linear_estimate=7.25')
        conn.execute("UPDATE harmonized SET linear_estimate=100 WHERE week_start LIKE '2023%'")
    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    app.selectbox(key='calendar_model').select('Linear regression').run()
    tables = [item.value for item in app.markdown if '<table' in item.value]
    assert '7.2' in tables[0]
    assert any('held-out' in c.value.lower() for c in app.caption)
    scale = next(c.value for c in app.caption if 'Color scale:' in c.value)
    assert '100.0' in scale
    assert any('Download harmonized CSV' == b.label for b in app.get('download_button'))
    frame = next(f.value for f in app.dataframe if 'evaluation_status' in f.value)
    assert {'reference_sensor', 'train_start', 'train_end', 'linear_estimate'} <= set(frame)
    app.sidebar.selectbox[0].select(2023).run()
    assert any('training-period fit' in c.value.lower() for c in app.caption)
    assert next(c.value for c in app.caption if 'Color scale:' in c.value) == scale
    app.selectbox(key='calendar_layer').select('Observed VIIRS').run()
    tables = [item.value for item in app.markdown if '<table' in item.value]
    assert '<b>8</b>' in tables[0]


def test_unavailable_model_is_explicit_and_legacy_database_keeps_observations(app_db):
    import sqlite3
    with sqlite3.connect(app_db) as conn:
        columns = [row[1] for row in conn.execute('PRAGMA table_info(harmonized)')]
        if 'xgboost_estimate' in columns:
            conn.execute('ALTER TABLE harmonized DROP COLUMN xgboost_estimate')
    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert app.selectbox(key='calendar_model').options == ['Linear regression']
    assert any('XGBoost' in item.value and 'unavailable' in item.value for item in app.info)
    with sqlite3.connect(app_db) as conn:
        conn.execute('DROP TABLE harmonized')
    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert app.selectbox(key='calendar_layer').options == ['Observed VIIRS']
    assert any('rebuild' in item.value.lower() for item in app.info)


def test_calendar_midrange_numeric_label_has_readable_contrast(app_db):
    import re
    import sqlite3
    with sqlite3.connect(app_db) as conn:
        conn.execute("UPDATE harmonized SET linear_estimate=100 WHERE week_start LIKE '2023%'")
        conn.execute("UPDATE harmonized SET linear_estimate=60 WHERE week_start LIKE '2025%'")
    app = AppTest.from_file('app.py').run(timeout=30)
    app.selectbox(key='calendar_model').select('Linear regression').run()
    table = next(m.value for m in app.markdown if '<table' in m.value)
    match = re.search(r'background:rgb\((\d+), (\d+), (\d+)\);color:([^";]+)[^>]*>6<br><b>60.0</b>', table)
    assert match, table
    channels = [int(v)/255 for v in match.groups()[:3]]
    linear = [v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4 for v in channels]
    light = sum(v*w for v,w in zip(linear, [0.2126,0.7152,0.0722]))
    text_light = 1 if match[4] == 'white' else 0
    ratio = (max(light,text_light)+0.05)/(min(light,text_light)+0.05)
    assert ratio >= 4.5
