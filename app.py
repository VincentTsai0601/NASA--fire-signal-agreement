"""Fire Signal Agreement dashboard. Run: streamlit run app.py."""
import calendar
import html
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from evidence import Evidence
from assistant import VLLMClient, answer
from reports import demo_answers


def calendar_style(fraction):
    """Sequential color with black/white text chosen from relative luminance."""
    fraction = min(max(fraction, 0), 1)
    rgb = tuple(round(low + (high-low)*fraction) for low, high in zip((239,246,255),(30,64,175)))
    channels = [value/255 for value in rgb]
    linear = [value/12.92 if value <= 0.04045 else ((value+0.055)/1.055)**2.4 for value in channels]
    luminance = sum(value*weight for value,weight in zip(linear, (0.2126,0.7152,0.0722)))
    foreground = 'white' if luminance < 0.179 else '#000'
    return f'background:rgb{rgb};color:{foreground}'


st.set_page_config(page_title='Fire Signal Agreement',page_icon='🔥',layout='wide')
st.title('Fire Signal Agreement')
st.caption('Study box · 37–42°N, 125–119°W · satellite hotspot detections')
db = Path(os.getenv('FIRE_DB','data/fire.sqlite'))
if not db.exists():
    st.info('Build the data first. See README.md for the pipeline command.')
    st.stop()
evidence = Evidence(db)
weekly = pd.DataFrame(evidence._query('SELECT * FROM weekly ORDER BY week_start'))
weekly.week_start = pd.to_datetime(weekly.week_start)
predictions = pd.DataFrame(evidence._query('SELECT * FROM predictions ORDER BY week_start'))
predictions.week_start = pd.to_datetime(predictions.week_start)
metadata = json.loads(evidence._query('SELECT json FROM metadata')[0]['json'])
has_harmonized = evidence._query("SELECT name FROM sqlite_master WHERE type='table' AND name='harmonized'")
harmonized = pd.DataFrame(evidence._query('SELECT * FROM harmonized ORDER BY week_start')) if has_harmonized else pd.DataFrame()
if not harmonized.empty:
    harmonized.week_start = pd.to_datetime(harmonized.week_start)
st.warning('A zero means zero records in the supplied archive. It does not establish observing coverage or absence of fire. Bounds extend beyond California.')
year = st.sidebar.selectbox('Year',[2023,2024,2025],index=2)
view = weekly[weekly.week_start.dt.year == year]
a,b,c = st.columns(3)
for column,label,field in [(a,'Terra','terra_count'),(b,'Aqua','aqua_count'),(c,'SNPP VIIRS','viirs_count')]:
    column.metric(label,f'{int(view[field].sum()):,}')
st.caption('Totals include only complete Monday–Sunday UTC weeks in the selected split.')
st.subheader('Weekly calendar')
model_columns = {label: column for label, column in [('XGBoost', 'xgboost_estimate'), ('Linear regression', 'linear_estimate')]
                 if column in harmonized}
layers = ['Calibrated MODIS (VIIRS-equivalent)', 'Observed VIIRS'] if model_columns else ['Observed VIIRS']
layer = st.selectbox('Calendar layer', layers, key='calendar_layer')
model_column = None
if model_columns:
    model_label = st.selectbox('Calibration model', list(model_columns), key='calendar_model', disabled=layer == 'Observed VIIRS')
    model_column = model_columns[model_label]
    if 'xgboost_estimate' not in harmonized:
        st.info('XGBoost calendar estimates are unavailable. Linear regression is labeled separately; no XGBoost values are substituted.')
else:
    st.info('Calibrated calendar unavailable in this database. Rebuild the data with the updated pipeline to enable it.')
calibrated_layer = layer != 'Observed VIIRS'
calendar_data = harmonized if calibrated_layer else weekly
value_column = model_column if calibrated_layer else 'viirs_count'
calendar_view = calendar_data[calendar_data.week_start.dt.year == year]
lookup = {row.week_start.date(): float(getattr(row, value_column)) for row in calendar_view.itertuples()}
maximum = max(float(calendar_data[value_column].max()), 1.0)
st.caption('Each colored Monday represents the entire UTC week. Missing and excluded weeks show —. Color and numeric labels show the selected layer.')
if calibrated_layer:
    status = 'Held-out 2025 estimates; 2025 VIIRS targets were excluded from fitting.' if year == 2025 else 'Training-period fit (2023–2024); these estimates are not independent validation.'
    st.caption(f'{model_label} · {status}')
    st.caption('MODIS-derived expected VIIRS-equivalent hotspot detections. Fractional estimates and nonzero estimates for zero MODIS records are possible. These are not unique fires, burned area, forecasts, or coverage-corrected activity.')
st.caption(f'Color scale: 0–{maximum:,.1f} · fixed across 2023–2025 for this layer and model. Light = lower activity; dark = higher activity.')
swatches = []
for fraction in [0, 0.5, 1]:
    swatches.append(f'<span style="display:inline-block;padding:6px;{calendar_style(fraction)}">{maximum*fraction:,.1f}</span>')
st.markdown(' '.join(swatches), unsafe_allow_html=True)
cols = st.columns(3)
for month in range(1,13):
    with cols[(month-1)%3]:
        rows = []
        for week in calendar.Calendar().monthdatescalendar(year,month):
            cells = []
            for day in week:
                value = lookup.get(day) if day.month == month else None
                fraction = min(max((value or 0)/maximum, 0), 1)
                style = calendar_style(fraction) if value is not None else 'opacity:0.5'
                number = f'{value:,.1f}' if calibrated_layer and value is not None else f'{value:,.0f}' if value is not None else '—'
                label = f'{day.day}<br><b>{number}</b>' if value is not None else f'{day.day}<br>—'
                cells.append(f'<td style="padding:4px;text-align:center;{style}" title="{html.escape(str(day))}">{label if day.month == month else ""}</td>')
            rows.append('<tr>'+''.join(cells)+'</tr>')
        st.markdown(f'**{calendar.month_name[month]}**')
        st.markdown('<table style="width:100%;font-size:11px"><tr>'+''.join(f'<th>{d}</th>' for d in ['M','T','W','T','F','S','S'])+'</tr>'+''.join(rows)+'</table>',unsafe_allow_html=True)
st.subheader('Weekly sensor comparison')
estimates = harmonized if model_columns else predictions
merged = view.merge(estimates[['week_start']+[x for x in estimates if x.endswith('_estimate')]],on='week_start',how='left')
st.line_chart(merged.set_index('week_start'))
st.caption('Y axis: observed hotspot detections and MODIS-derived VIIRS-equivalent estimates. 2023–2024 estimates are training-period fits; only 2025 is held out.')
st.subheader('Held-out model comparison · 2025')
st.dataframe(pd.DataFrame(evidence.get_model_metrics()),hide_index=True)
st.caption('MAE: mean absolute error. Signed mean error = estimate − observed. Models use Terra and Aqua counts only; VIIRS is the target, not a feature.')
if year == 2025:
    gaps = [x for x in predictions if x.endswith('_gap')]
    st.bar_chart(predictions.set_index('week_start')[gaps])
    st.caption('Signed gap = observed VIIRS − estimated VIIRS; positive means underestimation.')
with st.expander('Verified answers to the demo questions'):
    st.caption('Computed from the published result tables and retrieved NASA notes; available without a model server.')
    for item in demo_answers(evidence):
        st.markdown('**' + item['question'] + '**')
        st.markdown(item['answer'])
st.subheader('Ask the analyst')
st.caption('Optional vLLM service. Charts work independently. Only aggregate weekly evidence is sent to the configured endpoint.')
question = st.text_input('Question',placeholder='Which 2025 week had the largest calibration error?')
if st.button('Ask',disabled=not question):
    try:
        client = VLLMClient()
        client.health()
        with st.spinner('Consulting evidence…'): st.write(answer(question,evidence,client))
    except Exception as exc: st.info(f'Analyst unavailable: {exc}. Use the evidence below.')
with st.expander('Raw evidence and provenance'):
    st.dataframe(view,hide_index=True)
    st.dataframe(predictions,hide_index=True)
    st.json(metadata)
    st.download_button('Download weekly CSV',weekly.to_csv(index=False),'weekly.csv','text/csv')
    if not harmonized.empty:
        exported = harmonized.copy()
        config = metadata.get('harmonization', {})
        for field in ['models', 'features', 'parameters', 'package_versions', 'negative_estimates', 'interpretation']:
            exported[field] = json.dumps(config.get(field))
        exported['study_bounds'] = json.dumps(metadata.get('bounds'))
        exported['source_sha256'] = json.dumps([source.get('sha256') for source in metadata.get('sources', [])])
        st.dataframe(exported, hide_index=True)
        st.download_button('Download harmonized CSV',exported.to_csv(index=False),'harmonized.csv','text/csv')
    for passage in evidence.corpus:
        st.markdown(f"[{passage['section']}]({passage['url']}): {passage['text']}")
