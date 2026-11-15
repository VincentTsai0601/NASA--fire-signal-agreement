"""Check live endpoint, demo answers and a five-case analyst benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from assistant import VLLMClient, answer
from claims import render_claim
from evidence import Evidence

QUESTIONS = [
    'For linear regression, which 2025 week had the largest absolute calibration error?',
    'Did XGBoost beat linear regression on 2025 held-out MAE?',
    'Why can MODIS and VIIRS sensors report different counts? Cite NASA documentation.',
]
SCOPE_QUESTIONS = [
    'How many fires occurred in Texas in 2025?',
    'Compare model accuracy for 2026.',
    'How many hotspots occurred across California in 2025?',
    'Show weekly counts for 2022.',
]
BENCHMARK_QUESTIONS = [
    "Which week was linear regression's worst miss in 2025?",
    'Which performed better, XGBoost or linear?',
    'Do 100 hotspot detections mean 100 separate fires?',
    SCOPE_QUESTIONS[0],
    SCOPE_QUESTIONS[1],
]


class RecordingClient:
    """Retain completion JSON for diagnosis without recording evidence prompts."""
    def __init__(self, client):
        self.client = client
        self.replies = []
        self.supports_task_routing = getattr(client, 'supports_task_routing', False)

    def route(self, question):
        reply = self.client.route(question)
        self.replies.append({'stage': 'route', 'response': reply})
        return reply

    def explain(self, question, sources):
        reply = self.client.explain(question, sources)
        self.replies.append({'stage': 'explain', 'response': reply})
        return reply

    def complete(self, messages):
        reply = self.client.complete(messages)
        self.replies.append(reply)
        return reply


def run_smoke(evidence, client):
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': False, 'cases': [],
              'model': getattr(client, 'model', None), 'benchmark_version': 1}
    try:
        client.health()
        sources = {
            'predictions': {'rows': evidence._query('SELECT * FROM predictions ORDER BY week_start LIMIT 52')},
            'metrics': {'rows': evidence.get_model_metrics()},
        }
        expected = [
            render_claim({'source': 'predictions', 'operation': 'largest_absolute_error', 'model': 'linear'}, sources),
            render_claim({'source': 'metrics', 'operation': 'compare_mae'}, sources),
        ]
        for index, question in enumerate(QUESTIONS):
            recorded = RecordingClient(client)
            result = answer(question, evidence, recorded)
            if index < 2:
                passed = expected[index] in result and 'Evidence:' in result
            else:
                # Source linkage is checked here; a human must still assess the explanation.
                passed = ('Evidence:' in result and 'https://' in result
                          and 'firms' in result.lower() and 'viirs' in result.lower())
            report['cases'].append({'question': question, 'answer': result, 'passed': passed,
                                    'model_replies': recorded.replies,
                                    'expected': expected[index] if index < 2 else 'Retrieved NASA source citation and sensor explanation'})
        report['scope_checks'] = []
        for question in SCOPE_QUESTIONS:
            recorded = RecordingClient(client)
            result = answer(question, evidence, recorded)
            passed = ('This question is outside the supported analysis:' in result
                      and not recorded.replies)
            report['scope_checks'].append({'question': question, 'answer': result,
                                           'passed': passed, 'model_replies': recorded.replies})
        report['benchmark_cases'] = []
        definition = next((passage for passage in evidence.corpus
                           if passage['section'] == 'MODIS pixel definition'), None)
        for index, question in enumerate(BENCHMARK_QUESTIONS):
            if index >= 3:
                case = next(case for case in report['scope_checks'] if case['question'] == question)
                report['benchmark_cases'].append({**case, 'expected': 'Unsupported-scope refusal without model dispatch'})
                continue
            recorded = RecordingClient(client)
            result = answer(question, evidence, recorded)
            if index < 2:
                reference = expected[index]
                passed = reference in result and 'Evidence:' in result
            else:
                reference = definition['text'] if definition else 'Required MODIS pixel-definition passage unavailable'
                passed = bool(definition and reference in result and definition['url'] in result and 'Evidence:' in result)
            report['benchmark_cases'].append({'question': question, 'answer': result,
                                               'passed': passed, 'expected': reference,
                                               'model_replies': recorded.replies})
        report['passed'] = all(case['passed'] for case in
                               report['cases'] + report['scope_checks'] + report['benchmark_cases'])
        report['benchmark_summary'] = {'passed': sum(case['passed'] for case in report['benchmark_cases']),
                                       'total': len(report['benchmark_cases'])}
        report['review_note'] = ('Numeric answers are checked against SQLite; the definition requires the curated pixel passage. '
                                 'Scope refusals test application guards, not model scope reasoning. '
                                 'These examples do not establish general routing accuracy.')
    except Exception as exc:
        report['error'] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/fire.sqlite')
    parser.add_argument('--output', default='data/live-smoke.json')
    args = parser.parse_args()
    try:
        report = run_smoke(Evidence(args.db), VLLMClient())
    except Exception as exc:
        report = {'passed': False, 'error': str(exc), 'cases': []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
