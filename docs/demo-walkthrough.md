# Fire Signal Agreement walkthrough

Target length: about 2–3 minutes. Keep the vLLM server running in its own terminal.

## Start the dashboard with Qwen configured

In the terminal running the old Streamlit process, stop Streamlit with Ctrl+C. Leave the separate vLLM terminal running. Then use the project Python environment:

```bash
cd ~/Labtory/NASA
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
streamlit run app.py
```

Open http://localhost:8501 in Chrome. If Streamlit prints a different local port, use that address. Start your screen recorder on the dashboard window once it is ready.

## Recording sequence and narration

1. Select **2025**. Show the totals and calendar.

   Say: “This dashboard compares satellite hotspot detections within 37–42 degrees north and 125–119 degrees west. It is a study box that extends beyond California. The 2025 totals are Terra 1,881, Aqua 2,104 and SNPP VIIRS 16,620. Each colored Monday represents a complete UTC week. Zero records do not establish the absence of fire or observing coverage.”

2. Scroll to **Weekly sensor comparison** and **Held-out model comparison · 2025**.

   Say: “The models use Terra and Aqua counts to estimate VIIRS detections. They train on 2023–2024 and are evaluated on 51 complete weeks in 2025. XGBoost MAE is about 80.81 detections, versus 163.96 for linear regression. This result concerns these supplied archives and this study box.”

3. Scroll to **Ask the analyst**. Paste each question and press **Ask**. Pause for the complete answer before continuing.

   - `For linear regression, which 2025 week had the largest absolute calibration error?`

     Check: week starting **2025-07-07**; observed **1,631**; estimate **760.72**; absolute error **870.28**. Say: “Python computes the numeric answer directly from the saved evidence tables.”

   - `Did XGBoost beat linear regression on 2025 held-out MAE?`

     Check: XGBoost **80.81**, linear **163.96**; lower is better.

   - `Why can MODIS and VIIRS sensors report different counts? Cite NASA documentation.`

     Check: the answer displays **Source-based notes (curated paraphrases)** and NASA links. VIIRS uses nominal 375 m pixels; MODIS uses nominal 1 km hotspot pixels. Hotspot detections are not distinct-fire counts. Say: “Qwen routes the question and selects relevant source passages. The application displays curated NASA notes rather than generated methodological claims.”

4. Expand **Raw evidence and provenance** briefly. Show the data tables and metadata. Stop the recording and play it back to confirm all three answers and their evidence are readable.

## Completion record

The live CLI smoke test passed all three questions on 2026-11-15 at 08:06:39 UTC. In-process Streamlit replay is recorded separately in `data/dashboard-replay.json`; replay does not establish a live browser run.

After recording, provide the video path and confirm whether the three browser answers match the checks above. Do not mark browser verification or recording complete until they have actually been performed. The separate NASA challenge brief is still needed to complete corpus coverage and challenge alignment.
