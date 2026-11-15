"""Render allowlisted numeric statements from application evidence only."""


def render_claim(claim, sources):
    if not isinstance(claim, dict):
        raise ValueError('claim must be an object')
    source_id, operation = claim.get('source'), claim.get('operation')
    if not isinstance(source_id, str) or source_id not in sources:
        raise ValueError('unknown claim source')
    rows = sources[source_id].get('rows')
    if not rows:
        raise ValueError('no records support this claim')
    allowed_keys = {'source', 'operation', 'model'} if operation == 'largest_absolute_error' else {'source', 'operation'}
    if set(claim) != allowed_keys:
        raise ValueError('unexpected claim fields')
    if operation == 'largest_absolute_error' and source_id == 'predictions':
        model = claim['model']
        if model not in ('median', 'linear', 'xgboost'):
            raise ValueError('unsupported model')
        column = f'{model}_estimate'
        valid = [row for row in rows if row.get(column) is not None and row.get('viirs_count') is not None]
        if not valid:
            raise ValueError('model predictions unavailable')
        row = max(valid, key=lambda r: abs(r['viirs_count'] - r[column]))
        gap = row['viirs_count'] - row[column]
        return (f"{model}: largest absolute error in complete 2025 held-out weeks: week starting "
                f"{str(row['week_start'])[:10]} UTC; observed VIIRS={row['viirs_count']:,.0f}, "
                f"estimated VIIRS={row[column]:,.2f}, absolute error={abs(gap):,.2f}, "
                f"signed gap (observed − estimated)={gap:,.2f}. Ties use the earliest week.")
    if operation == 'compare_mae' and source_id == 'metrics':
        models = {row['model']: row for row in rows}
        a, b = models.get('linear'), models.get('xgboost')
        if not a or not b or a.get('mae') is None or b.get('mae') is None:
            return 'XGBoost versus linear comparison unavailable: both held-out MAE values are required.'
        decision = 'XGBoost has lower MAE' if b['mae'] < a['mae'] else ('Linear regression has lower MAE' if a['mae'] < b['mae'] else 'The models tie on MAE')
        return f"{decision}. Linear MAE={a['mae']:,.2f}; XGBoost MAE={b['mae']:,.2f}. Lower MAE is better."
    if operation == 'totals' and source_id.startswith('counts:'):
        fields = ['terra_count', 'aqua_count', 'viirs_count']
        return 'Supplied archive totals: ' + ', '.join(f'{field}={sum(row[field] for row in rows):,}' for field in fields) + f'; {len(rows)} complete weeks returned.'
    if operation == 'rows' and (source_id in ('metrics', 'predictions') or source_id.startswith('counts:')):
        return '\n'.join('; '.join(f'{key}={value if value is not None else "unavailable"}' for key, value in row.items()) for row in rows)
    raise ValueError('unsupported claim operation for this source')
