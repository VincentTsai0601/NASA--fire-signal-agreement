import pandas as pd
import pytest
from pipeline import build_weekly, evaluate


def archive(tmp_path, name, satellite, dates, **extra):
    df = pd.DataFrame(dict(latitude=[39]*len(dates), longitude=[-121]*len(dates), acq_date=dates, satellite=satellite, version=['2' if satellite == 'SNPP' else '61.03']*len(dates)))
    for key, value in extra.items(): df[key] = value
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_week_boundaries_counts_and_renamed_archives(tmp_path):
    m = archive(tmp_path, 'renamed.csv', ['Terra','Aqua','Terra','Aqua'], ['2023-01-01','2023-01-02','2024-12-30','2025-01-06'])
    v = archive(tmp_path, 'other.csv', 'SNPP', ['2023-01-02','2025-01-06'])
    w = build_weekly(m,v)
    assert str(w.iloc[0].week_start.date()) == '2023-01-02'
    assert w.iloc[0].aqua_count == 1
    assert w.iloc[0].terra_count == 0
    assert len(w) == 155
    assert pd.Timestamp('2024-12-30') not in set(w.week_start)
    assert w.viirs_count.sum() == 2


@pytest.mark.parametrize('field,value', [('satellite','Aqua'),('version','2.0NRT'),('latitude',100),('acq_date','bad')])
def test_invalid_archive(tmp_path, field, value):
    m = archive(tmp_path,'m.csv','Terra',['2023-01-02'])
    v = archive(tmp_path,'v.csv','SNPP',['2023-01-02'], **{field:[value]}) if field != 'satellite' else archive(tmp_path,'v.csv',value,['2023-01-02'])
    with pytest.raises(ValueError): build_weekly(m,v)


def test_no_test_target_leakage():
    dates = pd.date_range('2023-01-02','2025-12-22',freq='W-MON')
    w = pd.DataFrame({'week_start':dates, 'terra_count':range(len(dates)), 'aqua_count':2, 'viirs_count':range(len(dates))})
    p, scores = evaluate(w)
    changed = w.copy()
    changed.loc[changed.week_start.dt.year == 2025,'viirs_count'] += 10000
    q, other = evaluate(changed)
    pd.testing.assert_frame_equal(p.filter(like='_estimate'),q.filter(like='_estimate'))
    assert scores['linear']['mae'] != other['linear']['mae']
    assert (p.filter(like='_estimate') >= 0).all().all()

@pytest.mark.parametrize('version', [None,'garbage'])
def test_missing_or_unrecognized_processing_version(tmp_path,version):
    m=archive(tmp_path,'m.csv','Terra',['2023-01-02'])
    v=archive(tmp_path,'v.csv','SNPP',['2023-01-02'],version=[version])
    with pytest.raises(ValueError,match='version'): build_weekly(m,v)


def test_box_edges_and_training_year_boundary(tmp_path):
    m=archive(tmp_path,'m.csv',['Terra']*4,['2023-12-31']*4,latitude=[37,42,36.99,42.01],longitude=[-125,-119,-121,-121])
    v=archive(tmp_path,'v.csv','SNPP',['2023-12-31'])
    w=build_weekly(m,v)
    row=w[w.week_start==pd.Timestamp('2023-12-25')].iloc[0]
    assert row.terra_count==2
    assert row.viirs_count==1


def test_metadata_records_model_configuration(tmp_path):
    m=archive(tmp_path,'m.csv','Terra',['2023-01-02'])
    v=archive(tmp_path,'v.csv','SNPP',['2023-01-02'])
    meta=build_weekly(m,v).attrs['metadata']
    assert meta['evaluation']['features']==['terra_count','aqua_count']
    assert meta['evaluation']['signed_mean_error']=='estimate - observed'
    assert meta['evaluation']['xgboost_parameters']['random_state']==42
    assert meta['sources'][0]['included_rows']==1


def test_standard_content_can_have_misleading_filename(tmp_path):
    m=archive(tmp_path,'renamed_nrt_copy.csv','Terra',['2023-01-02'])
    v=archive(tmp_path,'renamed_viirs.csv','SNPP',['2023-01-02'])
    assert build_weekly(m,v).iloc[0].terra_count==1
