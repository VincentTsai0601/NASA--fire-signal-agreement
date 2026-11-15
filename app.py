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
st.warning('A zero means zero records in the supplied archive. It does not establish observing coverage or absence of fire. Bounds extend beyond California.')
year = st.sidebar.selectbox('Year',[2023,2024,2025],index=2)
view = weekly[weekly.week_start.dt.year == year]
a,b,c = st.columns(3)
for column,label,field in [(a,'Terra','terra_count'),(b,'Aqua','aqua_count'),(c,'SNPP VIIRS','viirs_count')]:
    column.metric(label,f'{int(view[field].sum()):,}')
st.caption('Totals include only complete Monday–Sunday UTC weeks in the selected split.')
st.subheader('Weekly calendar')
st.caption('Each colored Monday represents the entire week. Missing and excluded weeks show —. Color indicates observed VIIRS count.')
lookup = {r.week_start.date():int(r.viirs_count) for r in view.itertuples()}
maximum = max(lookup.values(),default=1) or 1
cols = st.columns(3)
for month in range(1,13):
    with cols[(month-1)%3]:
        rows = []
        for week in calendar.Calendar().monthdatescalendar(year,month):
            cells = []
            for day in week:
                value = lookup.get(day) if day.month == month else None
                shade = 0.08+0.7*((value or 0)/maximum)**0.4
                style = f'background:rgba(245,115,40,{shade})' if value is not None else 'opacity:0.5'
                label = f'{day.day}<br><b>{value:,}</b>' if value is not None else f'{day.day}<br>—'
                cells.append(f'<td style="padding:4px;text-align:center;{style}" title="{html.escape(str(day))}">{label if day.month == month else ""}</td>')
            rows.append('<tr>'+''.join(cells)+'</tr>')
        st.markdown(f'**{calendar.month_name[month]}**')
        st.markdown('<table style="width:100%;font-size:11px"><tr>'+''.join(f'<th>{d}</th>' for d in ['M','T','W','T','F','S','S'])+'</tr>'+''.join(rows)+'</table>',unsafe_allow_html=True)
st.subheader('Weekly sensor comparison')
merged = view.merge(predictions[['week_start']+[x for x in predictions if x.endswith('_estimate')]],on='week_start',how='left')
st.line_chart(merged.set_index('week_start'))
st.caption('Y axis: satellite hotspot detections. Estimates are available only for held-out 2025 weeks.')
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
    for passage in evidence.corpus:
        st.markdown(f"[{passage['section']}]({passage['url']}): {passage['text']}")
