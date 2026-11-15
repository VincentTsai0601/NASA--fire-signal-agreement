import pandas as pd
import pytest
from pipeline import write_outputs
from evidence import Evidence
from assistant import answer

@pytest.fixture
def evidence(tmp_path):
    w = pd.DataFrame({'week_start':['2025-01-06'],'terra_count':[2],'aqua_count':[3],'viirs_count':[8]})
    w.attrs['metadata'] = {'schema_version':1}
    path = tmp_path/'fire.sqlite'
    write_outputs(w,w,{'linear':{'status':'available','mae':2,'signed_mean_error':-2}},path)
    return Evidence(path)


def test_exact_and_empty_windows(evidence):
    assert evidence.get_weekly_counts('2025-01-01','2025-01-31')[0]['viirs_count'] == 8
    assert evidence.get_weekly_counts('2024-01-01','2024-01-31') == []
    assert evidence.get_model_metrics()[0]['mae'] == 2

@pytest.mark.parametrize('start,end', [('bad','2025-01-01'),('2025-02-01','2025-01-01'),('2022-01-01','2025-01-01')])
def test_invalid_dates(evidence,start,end):
    with pytest.raises(ValueError): evidence.get_weekly_counts(start,end)


def test_geography(evidence):
    with pytest.raises(ValueError): evidence.get_weekly_counts('2025-01-01','2025-01-31',region='Nevada')


def test_retrieval(evidence):
    assert 'hotspot' in str(evidence.search_docs('hotspots versus fires')).lower()
    assert 'VIIRS' in str(evidence.search_docs('MODIS versus VIIRS'))


def test_tool_allowlist_and_budget(evidence):
    class Stub:
        def __init__(self): self.calls = 0
        def complete(self,messages):
            self.calls += 1
            return {'tool':'get_model_metrics','arguments':{}}
    client = Stub()
    result = answer('Did XGBoost beat linear regression?',evidence,client)
    assert client.calls == 4
    assert 'limit' in result.lower()


def test_unknown_tool(evidence):
    class Stub:
        def complete(self,messages): return {'tool':'execute_sql','arguments':{'sql':'DROP TABLE weekly'}}
    assert 'unsupported' in answer('question',evidence,Stub()).lower()
    assert evidence.get_model_metrics()


def test_accepted_iso_date_is_normalized_before_sql(evidence):
    assert evidence.get_weekly_counts('20250101','20250131')[0]['viirs_count']==8
