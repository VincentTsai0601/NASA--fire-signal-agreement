"""Verified reference answers computed from published tables, without an LLM."""
import pandas as pd

DEMO_QUESTIONS = [
    'Which 2025 week had the largest calibration error?',
    'Did XGBoost beat linear regression?',
    'Why can the sensors report different counts?',
]


def demo_answers(evidence):
    predictions = pd.DataFrame(evidence._query('SELECT * FROM predictions ORDER BY week_start LIMIT 52'))
    metrics = {row['model']: row for row in evidence.get_model_metrics()}
    errors = []
    for model in ['median', 'linear', 'xgboost']:
        column = f'{model}_estimate'
        if column not in predictions or predictions.empty:
            continue
        valid = predictions.dropna(subset=[column, 'viirs_count'])
        if valid.empty:
            continue
        differences = valid.viirs_count - valid[column]
        row = valid.loc[differences.abs().idxmax()]
        gap = float(row.viirs_count - row[column])
        errors.append(
            f"{model}: week starting {str(row.week_start)[:10]} UTC, observed VIIRS "
            f"{row.viirs_count:,.0f}, estimated VIIRS {row[column]:,.2f}; "
            f"absolute error {abs(gap):,.2f}, signed gap (observed − estimated) {gap:,.2f}."
        )
    error_answer = ' '.join(errors) if errors else 'No held-out predictions are available.'
    error_answer += ' Window: complete 2025 weeks. Metric: satellite hotspot detections. Source: SQLite predictions; ties use the earliest week.'

    linear, xgb = metrics.get('linear'), metrics.get('xgboost')
    if not linear or not xgb or linear['mae'] is None or xgb['mae'] is None:
        comparison = 'Comparison unavailable: both linear and XGBoost held-out MAE values are required.'
    else:
        prefix = 'Yes' if xgb['mae'] < linear['mae'] else ('No' if xgb['mae'] > linear['mae'] else 'They tied')
        comparison = f"{prefix}. XGBoost MAE {xgb['mae']:,.2f}; linear regression MAE {linear['mae']:,.2f}. Lower MAE is better."
    comparison += ' Window: complete 2025 held-out weeks. Metric: satellite hotspot detections. Source: SQLite metrics.'

    passages = evidence.search_docs('MODIS versus VIIRS resolution hotspot fires', top_k=3)
    nasa = [p for p in passages if p['url'].startswith('https://')]
    explanation = '\n\n'.join(f"{p['text']} [{p['section']}]({p['url']})" for p in nasa)
    if not explanation:
        explanation = 'No NASA source passage was retrieved; no methodological explanation is available.'
    return [{'question': question, 'answer': result} for question, result in zip(DEMO_QUESTIONS, [error_answer, comparison, explanation])]
