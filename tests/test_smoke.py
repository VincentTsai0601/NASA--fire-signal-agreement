import pandas as pd
from evidence import Evidence
from pipeline import write_outputs
from smoke_test import run_smoke


def make_evidence(tmp_path):
    weekly=pd.DataFrame({'week_start':['2025-01-06'],'terra_count':[2],'aqua_count':[3],'viirs_count':[8]})
    weekly.attrs['metadata']={}
    predictions=weekly.assign(linear_estimate=[30])
    path=tmp_path/'fire.sqlite'
    write_outputs(weekly,predictions,{'linear':{'status':'available','mae':22,'signed_mean_error':22},'xgboost':{'status':'unavailable','mae':None,'signed_mean_error':None}},path)
    return Evidence(path)


def test_smoke_fails_on_rejected_answers(tmp_path):
    class Client:
        def health(self): return True
        def complete(self,messages): return {'answer':'I made up this answer.'}
    report=run_smoke(make_evidence(tmp_path),Client())
    assert not report['passed']
    assert len(report['cases'])==3
    assert all(not case['passed'] for case in report['cases'])
    assert all(case['model_replies'] == [{'answer':'I made up this answer.'}]
               for case in report['cases'])


def test_smoke_records_live_endpoint_failure(tmp_path):
    class Client:
        def health(self): raise ValueError('model not served')
    report=run_smoke(make_evidence(tmp_path),Client())
    assert not report['passed']
    assert 'model not served' in report['error']


def test_smoke_checks_actual_table_answers(tmp_path):
    class Client:
        def health(self): return True
        def complete(self,messages):
            question=messages[1]['content']
            if 'largest' in question or 'worst miss' in question:
                return {'answer':'','kind':'data','citations':['predictions'],'claims':[{'source':'predictions','operation':'largest_absolute_error','model':'linear'}]}
            if 'XGBoost' in question:
                if len(messages)==3: return {'tool':'get_model_metrics','arguments':{}}
                return {'answer':'','kind':'data','citations':['metrics'],'claims':[{'source':'metrics','operation':'compare_mae'}]}
            if len(messages)==3: return {'tool':'search_docs','arguments':{'query':question}}
            return {'answer':'VIIRS has finer spatial resolution.','kind':'method','citations':['docs:1']}
    report=run_smoke(make_evidence(tmp_path),Client())
    assert report['passed']
    assert all(case['passed'] for case in report['cases'])
    assert [len(case['model_replies']) for case in report['cases']] == [1, 2, 2]
    assert report['cases'][1]['model_replies'][0]['tool'] == 'get_model_metrics'


def test_smoke_preserves_structured_routing_and_trace(tmp_path):
    class Client:
        supports_task_routing = True
        def health(self): return True
        def route(self, question):
            if 'largest' in question or 'worst miss' in question: return {'task': 'largest_error', 'model': 'linear'}
            if 'XGBoost' in question: return {'task': 'compare_mae'}
            return {'task': 'explain'}
        def explain(self, question, sources):
            assert all(ref.startswith('docs:') for ref in sources)
            return {'answer': 'VIIRS has finer spatial resolution.', 'kind': 'method', 'citations': ['docs:1']}
        def complete(self, messages):
            raise AssertionError('Live routing must be preserved by the recorder')
    report = run_smoke(make_evidence(tmp_path), Client())
    assert report['passed'], report
    assert report['cases'][0]['model_replies'] == [{'stage': 'route', 'response': {'task': 'largest_error', 'model': 'linear'}}]
    assert [reply['stage'] for reply in report['cases'][2]['model_replies']] == ['route', 'explain']


def test_smoke_reports_scope_checks_without_model_calls(tmp_path):
    class Client:
        supports_task_routing = True
        def health(self): return True
        def route(self, question):
            assert 'Texas' not in question and '2026' not in question
            if 'largest' in question or 'worst miss' in question: return {'task': 'largest_error', 'model': 'linear'}
            if 'XGBoost' in question: return {'task': 'compare_mae'}
            return {'task': 'explain'}
        def explain(self, question, sources):
            return {'answer': '', 'kind': 'method', 'citations': ['docs:1']}
    report = run_smoke(make_evidence(tmp_path), Client())
    assert report['passed']
    checks = report['scope_checks']
    assert len(checks) >= 2
    assert all(case['passed'] and not case['model_replies'] for case in checks)
    assert any('Texas' in case['question'] for case in checks)
    assert any('2026' in case['question'] for case in checks)


class BenchmarkClient:
    supports_task_routing = True
    model = 'test-model'
    def health(self): return True
    def route(self, question):
        if 'largest' in question or 'worst miss' in question:
            return {'task': 'largest_error', 'model': 'linear'}
        if 'XGBoost' in question:
            return {'task': 'compare_mae'}
        return {'task': 'explain'}
    def explain(self, question, sources):
        return {'answer': '', 'kind': 'method', 'citations': ['docs:1']}


def test_five_case_benchmark_matches_data_definition_and_scope(tmp_path):
    report = run_smoke(make_evidence(tmp_path), BenchmarkClient())
    assert report['passed'], report
    benchmark = report['benchmark_cases']
    assert len(benchmark) == 5
    assert report['benchmark_summary'] == {'passed': 5, 'total': 5}
    assert all(case['passed'] for case in benchmark)
    assert report['model'] == 'test-model'
    assert benchmark[0]['expected'] in benchmark[0]['answer']
    assert benchmark[1]['expected'] in benchmark[1]['answer']
    assert 'not a count of distinct fires' in benchmark[2]['answer']
    assert all(not case['model_replies'] for case in benchmark[3:])


def test_failing_paraphrase_fails_overall_benchmark(tmp_path):
    class Client(BenchmarkClient):
        def route(self, question):
            if 'worst miss' in question:
                return {'task': 'compare_mae'}
            return super().route(question)
    report = run_smoke(make_evidence(tmp_path), Client())
    assert all(case['passed'] for case in report['cases'])
    assert not report['passed']
    assert not report['benchmark_cases'][0]['passed']


def test_definition_benchmark_rejects_irrelevant_valid_citation(tmp_path):
    class Client(BenchmarkClient):
        def explain(self, question, sources):
            if '100' in question:
                # A real retrieved passage can still fail to answer the question.
                ref = next(ref for ref, passage in sources.items()
                           if passage['section'] != 'MODIS pixel definition')
                return {'answer': '', 'kind': 'method', 'citations': [ref]}
            return super().explain(question, sources)
    report = run_smoke(make_evidence(tmp_path), Client())
    assert all(case['passed'] for case in report['cases'])
    assert not report['passed']
    assert not report['benchmark_cases'][2]['passed']
