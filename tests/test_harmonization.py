import builtins
import sqlite3

import numpy as np
import pandas as pd
import pytest

import pipeline


@pytest.fixture
def weeks():
    # The exact linear relation should be recovered without using 2025 targets.
    return pd.DataFrame({
        'week_start': pd.to_datetime(['2023-01-02', '2023-01-09', '2024-01-01', '2025-01-06']),
        'terra_count': [1, 2, 3, 4], 'aqua_count': [0, 0, 0, 0],
        'viirs_count': [3, 5, 7, 900],
    })


def test_calibration_covers_training_and_heldout_without_target_leakage(weeks):
    result = pipeline.calibrate(weeks)
    assert result.linear_estimate.tolist() == pytest.approx([3, 5, 7, 9])
    assert result.evaluation_status.tolist() == ['training_fit'] * 3 + ['held_out']
    assert result.reference_sensor.unique().tolist() == ['SNPP VIIRS']
    assert result.metric.unique().tolist() == ['VIIRS-equivalent hotspot detections']
    assert result.train_start.unique().tolist() == ['2023-01-01']
    assert result.train_end.unique().tolist() == ['2024-12-31']
    changed = weeks.copy()
    changed.loc[3, 'viirs_count'] = 90000
    other = pipeline.calibrate(changed)
    pd.testing.assert_frame_equal(result.filter(like='_estimate'), other.filter(like='_estimate'))
    predictions, _ = pipeline.evaluate(weeks)
    for column in result.filter(like='_estimate'):
        np.testing.assert_allclose(result.loc[[3], column], predictions[column])


def test_missing_xgboost_is_not_fabricated(weeks, monkeypatch):
    original = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('optional package absent')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_xgboost)
    result = pipeline.calibrate(weeks)
    assert 'linear_estimate' in result
    assert 'xgboost_estimate' not in result


def test_persistence_keeps_calibrated_outputs_separate(weeks, tmp_path):
    weeks.attrs['metadata'] = {'schema_version': 1}
    predictions, metrics = pipeline.evaluate(weeks)
    path = tmp_path / 'fire.sqlite'
    pipeline.write_outputs(weeks, predictions, metrics, path, harmonized=pipeline.calibrate(weeks))
    with sqlite3.connect(path) as conn:
        result = pd.read_sql('SELECT * FROM harmonized ORDER BY week_start', conn)
        observed = pd.read_sql('SELECT * FROM weekly ORDER BY week_start', conn)
    assert len(result) == 4
    assert result.linear_estimate.iloc[-1] == pytest.approx(9)
    assert observed.viirs_count.iloc[-1] == 900
    assert 'linear_estimate' not in observed
