"""Reconcile published outputs against raw archives and record verification gaps."""
import argparse
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from evidence import Evidence
from pipeline import calibrate, FEATURES, XGBOOST_PARAMETERS


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _expected_weeks():
    result = []
    week = date(2023, 1, 2)
    while week + timedelta(days=6) <= date(2025, 12, 31):
        if not (week < date(2025, 1, 1) <= week + timedelta(days=6)):
            result.append(week.isoformat())
        week += timedelta(days=7)
    return result


def _raw_counts(paths, start, end):
    counts = {'terra_count': 0, 'aqua_count': 0, 'viirs_count': 0}
    mapping = {'Terra': 'terra_count', 'Aqua': 'aqua_count', 'SNPP': 'viirs_count'}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=['latitude', 'longitude', 'acq_date', 'satellite'], chunksize=100000):
            selected = chunk[chunk.latitude.between(37, 42) & chunk.longitude.between(-125, -119)
                             & chunk.acq_date.between(start, end)]
            for satellite, count in selected.groupby('satellite').size().items():
                counts[mapping[satellite]] += int(count)
    return counts


def _verify_metrics(weekly, predictions, metrics):
    errors = []
    expected = weekly[weekly.week_start.str[:4] == '2025'].reset_index(drop=True)
    if not predictions[expected.columns].reset_index(drop=True).equals(expected):
        errors.append('Prediction observations differ from held-out weekly counts')
    scores = {row['model']: row for row in metrics}
    for model in ['median', 'linear', 'xgboost']:
        score = scores.get(model)
        estimate = f'{model}_estimate'
        if not score:
            errors.append(f'Missing {model} status')
            continue
        if score['mae'] is None or score['signed_mean_error'] is None:
            if model != 'xgboost' or estimate in predictions or score['status'] == 'available':
                errors.append(f'Incomplete {model} scores')
            continue
        if score['status'] != 'available' or estimate not in predictions:
            errors.append(f'{model} status and predictions disagree')
            continue
        values = predictions[estimate]
        if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
            errors.append(f'Invalid {model} predictions')
            continue
        error = values - predictions.viirs_count
        for field, actual in [('mae', error.abs().mean()), ('signed_mean_error', error.mean())]:
            if not math.isclose(actual, score[field], rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f'{model} {field} differs from predictions')
        gap = f'{model}_gap'
        if gap not in predictions or not ((predictions[gap] + error).abs() < 1e-9).all():
            errors.append(f'{model} weekly gap sign or values differ')
    return {'status': 'failed' if errors else 'passed', 'errors': errors, 'models': metrics}


def _verify_harmonized(evidence, weekly, predictions, metadata):
    errors = []
    try:
        stored = pd.DataFrame(evidence._query('SELECT * FROM harmonized ORDER BY week_start'))
        expected = calibrate(weekly)
        expected.week_start = pd.to_datetime(expected.week_start).dt.strftime('%Y-%m-%d')
        stored.week_start = stored.week_start.str[:10]
        if set(stored) != set(expected) or len(stored) != len(expected):
            raise ValueError('Calibrated calendar columns or number of weeks differ')
        for column in expected:
            if column.endswith('_estimate'):
                values = stored[column].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, expected[column], rtol=1e-6, atol=1e-6):
                    errors.append(f'{column} does not reproduce training-only calibration')
                held = stored[stored.evaluation_status == 'held_out'][column].reset_index(drop=True)
                if column not in predictions or len(held) != len(predictions) or not np.allclose(held, predictions[column], rtol=1e-6, atol=1e-6):
                    errors.append(f'{column} differs from published held-out predictions')
            elif not stored[column].reset_index(drop=True).equals(expected[column].reset_index(drop=True)):
                errors.append(f'{column} differs from observations or calibration provenance')
        config = metadata.get('harmonization', {})
        for field, value in {
            'reference_sensor': 'SNPP VIIRS', 'metric': 'VIIRS-equivalent hotspot detections',
            'features': FEATURES, 'train_start': '2023-01-01', 'train_end': '2024-12-31',
            'negative_estimates': 'clip to zero',
            'models': [column.removesuffix('_estimate') for column in expected if column.endswith('_estimate')],
        }.items():
            if config.get(field) != value:
                errors.append(f'Harmonization metadata {field} differs')
        if config.get('parameters', {}).get('xgboost') != XGBOOST_PARAMETERS or not config.get('package_versions'):
            errors.append('Missing or inconsistent calibration configuration')
    except (sqlite3.Error, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(str(exc))
    return {'status': 'failed' if errors else 'passed', 'errors': errors}


def verify_outputs(db_path, modis_path, viirs_path, week_start='2025-08-04'):
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'offline_passed': False,
              'full_plan_complete': False, 'checks': {}}
    checks = report['checks']
    try:
        evidence = Evidence(db_path)
        weekly = pd.DataFrame(evidence._query('SELECT * FROM weekly ORDER BY week_start'))
        predictions = pd.DataFrame(evidence._query('SELECT * FROM predictions ORDER BY week_start'))
        metrics = evidence.get_model_metrics()
        metadata = json.loads(evidence._query('SELECT json FROM metadata')[0]['json'])
        expected = _expected_weeks()
        starts = weekly.week_start.str[:10].tolist()
        train = sum(start < '2025-01-01' for start in starts)
        test = sum(start >= '2025-01-01' for start in starts)
        checks['weekly_calendar'] = {'status': 'passed' if starts == expected else 'failed',
                                     'train_weeks': train, 'test_weeks': test, 'total_weeks': len(starts)}
        metadata_ok = (metadata.get('schema_version') == 1
                       and metadata.get('bounds') == {'south': 37, 'north': 42, 'west': -125, 'east': -119}
                       and metadata.get('train_weeks') == train and metadata.get('test_weeks') == test
                       and metadata.get('metric') == 'satellite hotspot detections'
                       and metadata.get('evaluation', {}).get('features') == ['terra_count', 'aqua_count'])
        checks['metadata'] = {'status': 'passed' if metadata_ok else 'failed'}
        recorded = metadata.get('sources', [])
        hashes = [_digest(path) for path in [modis_path, viirs_path]]
        checks['source_integrity'] = {'status': 'passed' if hashes == [item.get('sha256') for item in recorded] else 'failed',
                                      'actual_sha256': hashes}
        if week_start not in expected:
            checks['raw_reconciliation'] = {'status': 'failed', 'error': 'Choose an included complete Monday–Sunday week'}
        else:
            end = (date.fromisoformat(week_start) + timedelta(days=6)).isoformat()
            actual = _raw_counts([modis_path, viirs_path], week_start, end)
            rows = evidence.get_weekly_counts(week_start, week_start)
            published = {field: rows[0][field] for field in actual} if len(rows) == 1 else {}
            checks['raw_reconciliation'] = {'status': 'passed' if actual == published else 'failed',
                                           'start': week_start, 'end': end, 'actual': actual, 'published': published}
        checks['heldout_metrics'] = _verify_metrics(weekly, predictions, metrics)
        checks['harmonized_calendar'] = _verify_harmonized(evidence, weekly, predictions, metadata)
        report['offline_passed'] = all(check['status'] == 'passed' for check in checks.values())
        xgb = next((row for row in metrics if row['model'] == 'xgboost'), {})
        checks['xgboost_execution'] = {'status': 'passed' if xgb.get('status') == 'available' and checks['heldout_metrics']['status'] == 'passed' else 'unavailable',
                                       'detail': xgb.get('status', 'No published XGBoost result')}
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        checks['data_validation'] = {'status': 'failed', 'error': str(exc)}
        report['offline_passed'] = False
    for item in ['live_model_validation', 'browser_rendering', 'walkthrough_recording', 'challenge_brief']:
        checks[item] = {'status': 'not_run', 'detail': 'Requires separate integration verification or supplied input'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/fire.sqlite')
    parser.add_argument('--modis', default='fire_archive_M-C61_848110.csv')
    parser.add_argument('--viirs', default='fire_archive_SV-C2_848111.csv')
    parser.add_argument('--week', default='2025-08-04')
    parser.add_argument('--output', default='data/verification.json')
    args = parser.parse_args()
    report = verify_outputs(args.db, args.modis, args.viirs, args.week)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['offline_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
