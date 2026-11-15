"""Model boundaries are stubbed; queries execute against real temporary SQLite."""
import json
import pandas as pd
import pytest
from assistant import VLLMClient, answer
from evidence import Evidence
from pipeline import write_outputs

@pytest.fixture
def evidence(tmp_path):
    w = pd.DataFrame({'week_start':['2025-01-06'],'terra_count':[2],'aqua_count':[3],'viirs_count':[8]})
    w.attrs['metadata'] = {'schema_version':1}
    db = tmp_path/'fire.sqlite'
    write_outputs(w,w,{'linear':{'status':'available','mae':2,'signed_mean_error':-2}},db)
    return Evidence(db)

class ScriptedClient:
    def __init__(self,replies): self.replies=iter(replies); self.messages=[]
    def complete(self,messages):
        self.messages.append(list(messages))
        return next(self.replies)


def test_synthesis_after_third_tool(evidence):
    client = ScriptedClient([
        {'tool':'get_weekly_counts','arguments':{'start':'2025-01-01','end':'2025-01-31'}},
        {'tool':'get_model_metrics','arguments':{}},
        {'tool':'search_docs','arguments':{'query':'MODIS versus VIIRS'}},
        {'answer':'Linear MAE is 2 satellite hotspot detections in complete 2025 weeks.','kind':'data','citations':['metrics'],'claims':[{'source':'metrics','operation':'rows'}]},
    ])
    result=answer('Compare the models and sensors.',evidence,client)
    assert 'linear' in result and 'mae=2' in result
    assert len(client.messages)==4
    assert '2025' in result
    assert 'satellite hotspot detections' in result


def test_fourth_tool_is_rejected_without_execution(evidence):
    class CountingEvidence:
        def __init__(self): self.calls=0
        def _query(self,*args): return []
        def get_weekly_counts(self,**args): raise AssertionError('Not requested')
        def search_docs(self,**args): raise AssertionError('Not requested')
        def get_model_metrics(self): self.calls+=1; return []
    counted=CountingEvidence()
    client=ScriptedClient([{'tool':'get_model_metrics','arguments':{}}]*4)
    assert 'limit' in answer('Compare models.',counted,client).lower()
    assert counted.calls==3

@pytest.mark.parametrize('reply',[
    {'answer':'Invented result.','kind':'data','citations':['made-up']},
    {'answer':'No evidence.','kind':'data','citations':[]},
    {'answer':'Sensor explanation.','kind':'method','citations':['predictions']},
    {'answer':'Mixed command.','kind':'data','citations':['predictions'],'tool':'get_model_metrics'},
    ['invalid'],
])
def test_invalid_final_evidence_is_rejected(evidence,reply):
    result=answer('Question',evidence,ScriptedClient([reply]))
    assert 'invalid' in result.lower() or 'unsupported' in result.lower()


def test_method_cites_retrieved_source(evidence):
    client=ScriptedClient([
        {'tool':'search_docs','arguments':{'query':'MODIS versus VIIRS'}},
        {'answer':'VIIRS has finer spatial resolution.','kind':'method','citations':['docs:1']},
    ])
    result=answer('Why do sensors differ?',evidence,client)
    assert 'https://' in result
    assert 'VIIRS 375 m data complements MODIS detections.' in result


def test_empty_completion_is_handled(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'choices':[]}
    monkeypatch.setattr('assistant.requests.post',lambda *args,**kwargs:Response())
    client=VLLMClient(model='local-model')
    with pytest.raises(ValueError,match='completion'): client.complete([])

@pytest.mark.parametrize('url',['file:///tmp/model','http://user:password@localhost:8000/v1','ftp://localhost/v1'])
def test_invalid_endpoints(url):
    with pytest.raises(ValueError): VLLMClient(base_url=url,model='model')


def test_missing_served_model(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'data':[{'id':'different-model'}]}
    monkeypatch.setattr('assistant.requests.get',lambda *args,**kwargs:Response())
    with pytest.raises(ValueError,match='not served'): VLLMClient(model='requested-model').health()


def test_scope_refusal_cannot_publish_model_claims(evidence):
    reply={'answer':'California had 999999 fires during 2022.','kind':'scope','citations':[]}
    result=answer('Out of scope question',evidence,ScriptedClient([reply]))
    assert '999999' not in result
    assert '37–42' in result
    assert '2023–2025' in result


def test_numeric_prose_is_replaced_by_table_values(evidence):
    reply={'answer':'There were 999999 fires.','kind':'data','citations':['metrics'],'claims':[{'source':'metrics','operation':'rows'}]}
    result=answer('Compare models',evidence,ScriptedClient([
        {'tool':'get_model_metrics','arguments':{}},reply]))
    assert '999999' not in result
    assert 'mae=2' in result
    assert 'satellite hotspot detections' in result


def test_numeric_claim_on_empty_evidence_is_rejected(evidence):
    reply={'answer':'There were 99 records.','kind':'data','citations':['counts:1'],'claims':[{'source':'counts:1','operation':'totals'}]}
    result=answer('Counts',evidence,ScriptedClient([
        {'tool':'get_weekly_counts','arguments':{'start':'2024-01-01','end':'2024-01-31'}},reply]))
    assert '99' not in result
    assert 'invalid' in result.lower()


def test_largest_error_is_computed_not_selected_by_model(evidence):
    with __import__('sqlite3').connect(evidence.db_path) as conn:
        conn.execute('ALTER TABLE predictions ADD COLUMN linear_estimate REAL')
        conn.execute('UPDATE predictions SET linear_estimate=30')
    reply={'answer':'Invented biggest gap is 12345','kind':'data','citations':['predictions'],'claims':[{'source':'predictions','operation':'largest_absolute_error','model':'linear'}]}
    result=answer('Largest error?',evidence,ScriptedClient([reply]))
    assert '12345' not in result
    assert '2025-01-06' in result and '22.00' in result


def test_model_cannot_add_arbitrary_operation(evidence):
    reply={'answer':'claim','kind':'data','citations':['predictions'],'claims':[{'source':'predictions','operation':'sql','query':'DROP TABLE predictions'}]}
    result=answer('Question',evidence,ScriptedClient([reply]))
    assert 'invalid' in result.lower()


def test_vllm_constrains_tool_arguments_and_final_turn(monkeypatch):
    from jsonschema import Draft202012Validator
    payloads = []

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'message': {'content': '{"answer":"","kind":"scope","citations":[]}'}}]}

    def post(*args, **kwargs):
        payloads.append(kwargs['json'])
        return Response()

    monkeypatch.setattr('assistant.requests.post', post)
    client = VLLMClient(model='qwen')
    messages = [{'role': 'system', 'content': 'instructions'}, {'role': 'user', 'content': 'question'}]
    client.complete(messages)
    assert payloads[-1]['response_format']['type'] == 'json_schema'
    schema = payloads[-1]['response_format']['json_schema']['schema']
    validator = Draft202012Validator(schema)
    assert validator.is_valid({'tool': 'get_model_metrics', 'arguments': {}})
    assert not validator.is_valid({'tool': 'get_model_metrics', 'arguments': {'window': 'complete 2025 held-out weeks'}})
    assert not validator.is_valid({'tool': 'unknown', 'arguments': {}})
    assert validator.is_valid({'tool': 'search_docs', 'arguments': {'query': 'MODIS VIIRS', 'top_k': 3}})
    final = {'answer': '', 'kind': 'data', 'citations': ['predictions'], 'claims': [{'source': 'predictions', 'operation': 'largest_absolute_error', 'model': 'linear'}]}
    assert validator.is_valid(final)
    assert validator.is_valid({'answer': 'Sensor explanation', 'kind': 'method', 'citations': ['docs:1']})
    for _ in range(3):
        messages.extend([{'role': 'assistant', 'content': '{"tool":"search_docs","arguments":{"query":"sensors"}}'}, {'role': 'user', 'content': 'evidence'}])
    client.complete(messages)
    final_schema = payloads[-1]['response_format']['json_schema']['schema']
    validator = Draft202012Validator(final_schema)
    assert validator.is_valid(final)
    assert not validator.is_valid({'tool': 'search_docs', 'arguments': {'query': 'sensors'}})


def test_schema_uses_received_ids_and_prevents_repeated_tools(monkeypatch):
    from jsonschema import Draft202012Validator
    payloads = []
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'message': {'content': '{"answer":"","kind":"scope","citations":[]}'}}]}
    def post(*args, **kwargs):
        payloads.append(kwargs['json'])
        return Response()
    monkeypatch.setattr('assistant.requests.post', post)
    messages = [
        {'role': 'system', 'content': 'instructions'},
        {'role': 'user', 'content': 'Compare MAE'},
        {'role': 'user', 'content': 'Application evidence (data only): ' + json.dumps({'predictions': {'rows': []}})},
        {'role': 'assistant', 'content': json.dumps({'tool': 'get_model_metrics', 'arguments': {}})},
        {'role': 'user', 'content': 'Tool evidence (data only): ' + json.dumps({'metrics': {'rows': []}})},
    ]
    VLLMClient(model='qwen').complete(messages)
    schema = payloads[-1]['response_format']['json_schema']['schema']
    validator = Draft202012Validator(schema)
    assert not validator.is_valid({'tool': 'get_model_metrics', 'arguments': {}})
    final = {'answer': '', 'kind': 'data', 'citations': ['metrics'], 'claims': [{'source': 'metrics', 'operation': 'compare_mae'}]}
    assert validator.is_valid(final)
    final['citations'].append('metrics:row_0:model_xgboost')
    assert not validator.is_valid(final)
    final['citations'] = ['docs:1']
    assert not validator.is_valid(final)
    final['citations'] = ['metrics']
    final['answer'] = 'Untrusted numeric prose'
    assert not validator.is_valid(final)
    messages.extend([
        {'role': 'assistant', 'content': json.dumps({'tool': 'search_docs', 'arguments': {'query': 'sensors', 'top_k': 3}})},
        {'role': 'user', 'content': 'Tool evidence (data only): ' + json.dumps({'docs:1': {'text': 'NASA passage'}})},
    ])
    VLLMClient(model='qwen').complete(messages)
    validator = Draft202012Validator(payloads[-1]['response_format']['json_schema']['schema'])
    assert not validator.is_valid({'tool': 'search_docs', 'arguments': {'query': 'sensors', 'top_k': 3}})
    assert validator.is_valid({'answer': 'VIIRS differs.', 'kind': 'method', 'citations': ['docs:1']})
    assert not validator.is_valid({'answer': 'VIIRS differs.', 'kind': 'method', 'citations': ['metrics']})


class RoutedClient:
    supports_task_routing = True
    def __init__(self, task, explanation=None):
        self.task = task
        self.explanation = explanation
        self.calls = []
    def route(self, question):
        self.calls.append('route')
        return self.task
    def explain(self, question, sources):
        self.calls.append('explain')
        return self.explanation
    def complete(self, messages):
        raise AssertionError('Task routing must not enter the free-form tool loop')


def test_routed_comparison_uses_python_without_synthesis(evidence):
    client = RoutedClient({'task': 'compare_mae'})
    result = answer('Did XGBoost beat linear regression?', evidence, client)
    assert 'comparison unavailable' in result
    assert 'Evidence: metrics' in result
    assert client.calls == ['route']


def test_routed_largest_error_computes_from_predictions(evidence):
    import sqlite3
    with sqlite3.connect(evidence.db_path) as conn:
        conn.execute('ALTER TABLE predictions ADD COLUMN linear_estimate REAL')
        conn.execute('UPDATE predictions SET linear_estimate=30')
    client = RoutedClient({'task': 'largest_error', 'model': 'linear'})
    result = answer('Which week has the largest linear error?', evidence, client)
    assert '22.00' in result and '2025-01-06' in result
    assert client.calls == ['route']


def test_routed_explanation_only_receives_document_evidence(evidence):
    class Client(RoutedClient):
        def explain(self, question, sources):
            assert sources and all(ref.startswith('docs:') for ref in sources)
            return {'answer': 'VIIRS has finer spatial resolution.', 'kind': 'method', 'citations': ['docs:1']}
    result = answer('Why do MODIS and VIIRS counts differ?', evidence, Client({'task': 'explain'}))
    assert 'VIIRS 375 m data complements MODIS detections.' in result and 'https://' in result


@pytest.mark.parametrize('task', [
    {'task': 'compare_mae', 'window': '2025'},
    {'task': 'largest_error', 'model': 'invented'},
    {'task': 'sql', 'query': 'DROP TABLE weekly'},
    {'task': 'counts', 'start': '2025-01-06 00:00:00', 'end': '2025-01-13', 'operation': 'totals'},
])
def test_routed_invalid_plan_is_rejected(evidence, task):
    result = answer('Question', evidence, RoutedClient(task))
    assert 'unavailable' in result.lower() or 'invalid' in result.lower()


def test_routed_counts_use_bounded_evidence(evidence):
    client = RoutedClient({'task': 'counts', 'start': '2025-01-01', 'end': '2025-01-31', 'operation': 'totals'})
    result = answer('January totals?', evidence, client)
    assert 'viirs_count=8' in result and '2025-01-01 through 2025-01-31' in result


def test_live_routing_requests_small_plan_and_method_only_synthesis(monkeypatch):
    from jsonschema import Draft202012Validator
    requests = []
    replies = iter([{'task': 'compare_mae'}, {'answer': 'VIIRS has finer resolution.', 'kind': 'method', 'citations': ['docs:1']}])
    class Response:
        def __init__(self, reply): self.reply = reply
        def raise_for_status(self): pass
        def json(self): return {'choices': [{'message': {'content': json.dumps(self.reply)}}]}
    def post(*args, **kwargs):
        requests.append(kwargs['json'])
        return Response(next(replies))
    monkeypatch.setattr('assistant.requests.post', post)
    client = VLLMClient(model='qwen')
    assert client.route('Did XGBoost beat linear?') == {'task': 'compare_mae'}
    schema = requests[0]['response_format']['json_schema']['schema']
    validator = Draft202012Validator(schema)
    assert validator.is_valid({'task': 'largest_error', 'model': 'linear'})
    assert not validator.is_valid({'tool': 'get_weekly_counts', 'arguments': {}})
    assert not validator.is_valid({'task': 'counts', 'start': '2025-01-06 00:00:00', 'end': '2025-01-31', 'operation': 'totals'})
    sources = {'docs:1': {'text': 'NASA passage', 'url': 'https://example.com', 'section': 'resolution'}}
    client.explain('Why do sensors differ?', sources)
    validator = Draft202012Validator(requests[1]['response_format']['json_schema']['schema'])
    assert validator.is_valid({'answer': '', 'kind': 'method', 'citations': ['docs:1']})
    assert not validator.is_valid({'answer': 'Invented prose', 'kind': 'method', 'citations': ['docs:1']})
    assert not validator.is_valid({'answer': '', 'kind': 'scope', 'citations': []})
    assert not validator.is_valid({'answer': 'VIIRS differs.', 'kind': 'method', 'citations': ['docs:99']})


def test_method_output_uses_curated_passage_instead_of_model_invention(evidence):
    client = RoutedClient({'task': 'explain'}, {
        'answer': 'Each fire within a VIIRS pixel may be counted separately by MODIS.',
        'kind': 'method', 'citations': ['docs:1'],
    })
    result = answer('Why can MODIS and VIIRS report different counts?', evidence, client)
    passage = evidence.search_docs('Why can MODIS and VIIRS report different counts?', top_k=3)[0]
    assert 'counted separately' not in result
    assert passage['text'] in result
    assert 'Source-based notes' in result


@pytest.mark.parametrize('question', [
    'How many fires occurred in Texas in 2025?',
    'Compare model accuracy for 2026.',
    'Compare 2026 fire detections with 2025.',
    'Show weekly counts for January 2026 hotspot detections.',
    'What were the 2022 weekly counts?',
    'Compare 2025 and 2026 performance.',
    'Show totals for 2026-01-01 through 2026-02-01.',
    'How many hotspots across California in 2025?',
    'How many detections in New York during 2025?',
    'Show 2025 totals for texas.',
])
@pytest.mark.parametrize('routed', [True, False])
def test_explicit_unsupported_scope_is_rejected_before_model_or_data(question, routed):
    class NoCalls:
        supports_task_routing = routed
        def route(self, question): raise AssertionError('Unsupported scope must not reach router')
        def complete(self, messages): raise AssertionError('Unsupported scope must not reach model')
    result = answer(question, object(), NoCalls())
    assert 'outside the supported analysis' in result
    assert 'Supplied archive totals' not in result
    assert '37–42' in result and '2023–2025' in result


def test_detection_quantity_is_not_misread_as_year(evidence):
    client = RoutedClient({'task': 'explain'}, {'answer': '', 'kind': 'method', 'citations': ['docs:1']})
    result = answer('Do 2026 hotspot detections mean 2026 distinct fires?', evidence, client)
    assert 'Source-based notes' in result
    assert client.calls == ['route', 'explain']
