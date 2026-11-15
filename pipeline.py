"""Reproducible FIRMS archive aggregation and held-out calibration."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BOUNDS = {'south':37, 'north':42, 'west':-125, 'east':-119}
FEATURES = ['terra_count', 'aqua_count']
XGBOOST_PARAMETERS = dict(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42, n_jobs=1, objective='reg:squarederror')


def package_versions():
    result = {}
    for package in ['pandas', 'numpy', 'scikit-learn', 'xgboost']:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def read_archive(path, sensor):
    df = pd.read_csv(path)
    required = {'latitude','longitude','acq_date','satellite','version'}
    if not required <= set(df): raise ValueError(f'{path}: missing columns {required-set(df)}')
    if df.empty: raise ValueError('Empty archive')
    if df.version.astype(str).str.contains('NRT',case=False).any():
        raise ValueError('NRT inputs are unsupported; supply a standard archive')
    standard_version = 61.03 if sensor == 'modis' else 2.0
    versions = pd.to_numeric(df.version, errors='coerce')
    if not versions.eq(standard_version).all():
        raise ValueError(f'Expected standard archive version {standard_version} for {sensor}')
    allowed = {'Terra','Aqua'} if sensor == 'modis' else {'SNPP'}
    if not set(df.satellite) <= allowed: raise ValueError(f'Expected satellites {allowed}')
    for col, low, high in [('latitude',-90,90),('longitude',-180,180)]:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not df[col].between(low,high).all(): raise ValueError(f'Invalid {col}')
    try: df['date'] = pd.to_datetime(df.acq_date,format='%Y-%m-%d',errors='raise')
    except (ValueError, TypeError) as exc: raise ValueError('Invalid acquisition dates') from exc
    if df.date.isna().any(): raise ValueError('Missing acquisition dates')
    return df


def build_weekly(modis_path, viirs_path):
    sources = [read_archive(modis_path,'modis'),read_archive(viirs_path,'viirs')]
    weeks = pd.date_range('2023-01-02','2025-12-22',freq='W-MON')
    weeks = weeks[(weeks.year == (weeks + pd.Timedelta(days=6)).year) | ((weeks.year == 2023) & ((weeks + pd.Timedelta(days=6)).year == 2024))]
    out = pd.DataFrame(index=weeks)
    out.index.name = 'week_start'
    details = []
    for df, path in zip(sources,[modis_path,viirs_path]):
        selected = df[df.latitude.between(37,42) & df.longitude.between(-125,-119)].copy()
        selected['week_start'] = selected.date - pd.to_timedelta(selected.date.dt.dayofweek,unit='D')
        for sat, column in [('Terra','terra_count'),('Aqua','aqua_count'),('SNPP','viirs_count')]:
            if sat in set(df.satellite):
                out[column] = selected[selected.satellite == sat].groupby('week_start').size().reindex(weeks,fill_value=0)
        details.append({'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'rows':len(df),'study_box_rows':len(selected),'included_rows':int(selected.week_start.isin(weeks).sum()),'min_date':str(df.date.min().date()),'max_date':str(df.date.max().date())})
    for col in FEATURES+['viirs_count']:
        if col not in out: out[col] = 0
    out = out[FEATURES+['viirs_count']].reset_index()
    out.attrs['metadata'] = {'schema_version':1,'evaluation':{'features':FEATURES,'train_start':'2023-01-01','train_end':'2024-12-31','test_start':'2025-01-01','test_end':'2025-12-31','signed_mean_error':'estimate - observed','signed_gap':'observed - estimate','negative_estimates':'clip to zero','xgboost_parameters':XGBOOST_PARAMETERS,'package_versions':package_versions()},'bounds':BOUNDS,'sources':details,'week_start_min':str(weeks.min().date()),'week_start_max':str(weeks.max().date()),'train_weeks':int((weeks.year < 2025).sum()),'test_weeks':int((weeks.year == 2025).sum()),'metric':'satellite hotspot detections','zero_meaning':'Zero supplied archive records; observing coverage and absence of fire are unknown.'}
    return out


def evaluate(weekly):
    w = weekly.copy()
    dates = pd.to_datetime(w.week_start)
    complete = (dates.dt.dayofweek == 0) & ((dates+pd.Timedelta(days=6)).dt.year == dates.dt.year)
    # Weeks spanning 2023/2024 remain wholly within the training period.
    train_mask = (dates >= '2023-01-01') & (dates+pd.Timedelta(days=6) < pd.Timestamp('2025-01-01')) & (dates.dt.dayofweek == 0)
    test_mask = complete & (dates.dt.year == 2025)
    train, test = w[train_mask], w[test_mask]
    if train.empty or test.empty: raise ValueError('Both training and held-out weeks are required')
    result = test.copy().reset_index(drop=True)
    estimates = {'median':np.full(len(test),train.viirs_count.median()),'linear':LinearRegression().fit(train[FEATURES],train.viirs_count).predict(test[FEATURES])}
    scores = {}
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(**XGBOOST_PARAMETERS)
        estimates['xgboost'] = model.fit(train[FEATURES],train.viirs_count).predict(test[FEATURES])
    except ImportError:
        scores['xgboost'] = {'status':'unavailable: install xgboost','mae':None,'signed_mean_error':None}
    for name, values in estimates.items():
        result[f'{name}_estimate'] = np.maximum(values,0)
        error = result[f'{name}_estimate']-result.viirs_count
        result[f'{name}_gap'] = -error
        scores[name] = {'status':'available','mae':float(abs(error).mean()),'signed_mean_error':float(error.mean())}
    return result, scores


def write_outputs(weekly, predictions, metrics, db_path):
    Path(db_path).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        weekly.to_sql('weekly',conn,if_exists='replace',index=False)
        predictions.to_sql('predictions',conn,if_exists='replace',index=False)
        pd.DataFrame([{'model':k,**v} for k,v in metrics.items()]).to_sql('metrics',conn,if_exists='replace',index=False)
        pd.DataFrame([{'json':json.dumps(weekly.attrs['metadata'])}]).to_sql('metadata',conn,if_exists='replace',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modis',required=True)
    parser.add_argument('--viirs',required=True)
    parser.add_argument('--output',default='data/fire.sqlite')
    args = parser.parse_args()
    weekly = build_weekly(args.modis,args.viirs)
    predictions, metrics = evaluate(weekly)
    write_outputs(weekly,predictions,metrics,args.output)
    print(json.dumps({'metadata':weekly.attrs['metadata'],'metrics':metrics},indent=2))

if __name__ == '__main__': main()
