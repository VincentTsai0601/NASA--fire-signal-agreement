import pandas as pd
from evidence import Evidence
from pipeline import write_outputs
from reports import demo_answers


def test_demo_answers_use_largest_absolute_error_and_unavailable_model(tmp_path):
    weekly = pd.DataFrame({'week_start':['2025-01-06','2025-01-13'],'terra_count':[1,2],'aqua_count':[2,3],'viirs_count':[10,30]})
    weekly.attrs['metadata'] = {}
    predictions = weekly.assign(linear_estimate=[30,25],linear_gap=[-20,5])
    db=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,{'linear':{'status':'available','mae':12.5,'signed_mean_error':7.5},'xgboost':{'status':'unavailable','mae':None,'signed_mean_error':None}},db)
    answers=demo_answers(Evidence(db))
    assert len(answers)==3
    assert '2025-01-06' in answers[0]['answer']
    assert '20.00' in answers[0]['answer']
    assert '-20.00' in answers[0]['answer']
    assert 'unavailable' in answers[1]['answer'].lower()
    assert 'beat' not in answers[1]['answer'].lower()
    assert 'https://firms.modaps.eosdis.nasa.gov' in answers[2]['answer']


def test_demo_reports_xgboost_loss(tmp_path):
    weekly=pd.DataFrame({'week_start':['2025-01-06'],'viirs_count':[2]})
    weekly.attrs['metadata']={}
    predictions=weekly.assign(linear_estimate=[2],xgboost_estimate=[7])
    db=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,{'linear':{'status':'available','mae':0,'signed_mean_error':0},'xgboost':{'status':'available','mae':5,'signed_mean_error':5}},db)
    answers=demo_answers(Evidence(db))
    assert 'No' in answers[1]['answer']
    assert '5.00' in answers[1]['answer']
    assert 'xgboost' in answers[0]['answer'].lower()
