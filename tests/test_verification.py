import json
import sqlite3
import pandas as pd
import pytest
from pipeline import build_weekly, calibrate, evaluate, write_outputs
from verify_project import verify_outputs

@pytest.fixture
def outputs(tmp_path):
    paths=[]
    for name,satellite,version in [('m','Terra','61.03'),('v','SNPP','2')]:
        path=tmp_path/f'{name}.csv'
        pd.DataFrame({'latitude':[39,39,36], 'longitude':[-121]*3, 'acq_date':['2023-01-02','2025-08-04','2025-08-04'], 'satellite':[satellite]*3, 'version':[version]*3}).to_csv(path,index=False)
        paths.append(path)
    weekly=build_weekly(*paths)
    predictions,metrics=evaluate(weekly)
    db=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,metrics,db,harmonized=calibrate(weekly))
    return db,*paths


def test_reconciles_raw_archives_and_metrics(outputs):
    report=verify_outputs(*outputs)
    assert report['offline_passed']
    assert report['checks']['raw_reconciliation']['actual']=={'terra_count':1,'aqua_count':0,'viirs_count':1}
    assert report['checks']['weekly_calendar']['train_weeks']==104
    assert report['checks']['heldout_metrics']['status']=='passed'
    assert not report['full_plan_complete']


def test_detects_tampered_source_hash(outputs):
    db,m,v=outputs
    with m.open('a') as file: file.write('\n')
    report=verify_outputs(db,m,v)
    assert not report['offline_passed']
    assert report['checks']['source_integrity']['status']=='failed'


def test_detects_corrupt_weekly_count(outputs):
    db,m,v=outputs
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE weekly SET terra_count=999 WHERE date(week_start)='2025-08-04'")
    report=verify_outputs(db,m,v)
    assert not report['offline_passed']
    assert report['checks']['raw_reconciliation']['status']=='failed'


def test_detects_corrupt_published_score(outputs):
    db,m,v=outputs
    with sqlite3.connect(db) as conn: conn.execute("UPDATE metrics SET mae=999 WHERE model='linear'")
    report=verify_outputs(db,m,v)
    assert not report['offline_passed']
    assert report['checks']['heldout_metrics']['status']=='failed'


def test_missing_database_does_not_create_one(tmp_path):
    db=tmp_path/'missing.sqlite'
    report=verify_outputs(db,tmp_path/'m.csv',tmp_path/'v.csv')
    assert not report['offline_passed']
    assert not db.exists()


def test_excluded_week_is_not_reconciled_as_zero(outputs):
    report=verify_outputs(*outputs,week_start='2024-12-30')
    assert not report['offline_passed']
    assert report['checks']['raw_reconciliation']['status']=='failed'


@pytest.mark.parametrize('statement', [
    "UPDATE harmonized SET linear_estimate=999 WHERE week_start LIKE '2023%'",
    "UPDATE harmonized SET evaluation_status='held_out' WHERE week_start LIKE '2023%'",
    "UPDATE harmonized SET reference_sensor='MODIS'",
    "DELETE FROM harmonized WHERE week_start LIKE '2025-08-04%'",
    "UPDATE metadata SET json=json_set(json, '$.harmonization.train_end', '2025-12-31')",
])
def test_detects_corrupt_harmonized_calendar_or_provenance(outputs, statement):
    db, m, v = outputs
    with sqlite3.connect(db) as conn:
        conn.execute(statement)
    report = verify_outputs(db, m, v)
    assert not report['offline_passed']
    assert report['checks']['harmonized_calendar']['status'] == 'failed'
