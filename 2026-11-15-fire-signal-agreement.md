# Fire Signal Agreement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development to work through the checkboxes.

**Goal** — Compare MODIS and VIIRS hotspot detections, test cross-sensor calibration, and let users ask questions grounded in the results and NASA documentation.

**Architecture** — A Python batch job prepares weekly counts and held-out predictions. Streamlit reads those outputs for a calendar and an evidence panel. A bounded analyst agent uses typed data queries and a small retrieval index, with vLLM serving the language model.

**Tech stack** — Python, pandas, scikit-learn, XGBoost, Streamlit, SQLite, pytest, and a vLLM endpoint.

**Scope for review** — Start with the two uploaded NASA FIRMS archive CSVs and the study box at 37–42°N, 125–119°W. Add an ML comparison and a read-only analyst for the same area. Use the study-box label in the UI because the bounds extend beyond California. Build from the NASA data and challenge brief. Keep raw data local by default.

## Files and interfaces

| File | Responsibility |
|---|---|
| `pipeline.py` | Validate archives, build weekly tables, train and score calibration models |
| `evidence.py` | Query weekly SQLite results and retrieve source passages |
| `assistant.py` | Run typed tools and synthesize a cited answer via vLLM |
| `app.py` | Calendar, model comparison, and analyst panel |
| `docs/sources.md` | NASA links, data provenance, and explanation of limitations |
| `tests/` | Regression tests using small synthetic inputs |
| `README.md` | Setup, commands, local model configuration, and demo steps |

## 1. Prepare weekly counts

- [x] Accept explicit paths for `fire_archive_M-C61_848110.csv` and `fire_archive_SV-C2_848111(1).csv`. Validate coordinates, acquisition dates, satellite IDs and required columns. Test that an NRT input or wrong satellite is rejected and that a renamed copy works when supplied explicitly.
- [x] Implement `build_weekly(modis_path, viirs_path) -> pandas.DataFrame`. Filter to the study box. Use complete Monday–Sunday UTC weeks wholly within 2023–2025, and exclude weeks that cross the train/test boundary. Output `week_start`, `terra_count`, `aqua_count`, and `viirs_count` in chronological order. Validate filtering, date boundaries, and separate Terra/Aqua counts with a fixture. A zero represents zero records in the supplied archive for that week. The archive does not establish observing coverage or absence of fire.
- [x] Run against the real inputs and record row counts, weeks per split, and the actual min/max dates. Do a direct reconciliation of at least one weekly count against the raw CSV. Save versioned schema, bounds, sources, and date range in the output metadata.

## 2. Evaluate calibration

- [x] Implement `evaluate(weekly) -> tuple[pandas.DataFrame, dict]`. Train on weeks wholly inside 2023–2024. Use a training median benchmark, linear regression and a small XGBoost regressor. Use the same MODIS inputs for both regressors and clip negative estimates to zero. Keep VIIRS observations out of model features.
- [x] Score only complete 2025 weeks with MAE and signed mean error. Report all scores even if XGBoost loses. Test that modifying 2025 targets changes the scores while leaving the trained models' predictions identical. Write weekly counts, held-out predictions, and metrics to SQLite.
- [x] Produce a working Streamlit calendar plus a weekly comparison chart showing Terra, Aqua, observed VIIRS and estimated VIIRS where available. Show the signed gap between observed and estimated VIIRS for held-out weeks. Label the underlying metric “satellite hotspot detections.” Show actual published metrics and mark unavailable values clearly. Verify on the uploaded archives before adding chat.

## 3. Add the analyst agent, RAG, and vLLM

- [x] Implement read-only tools `get_weekly_counts(start, end)`, `get_model_metrics()` and `search_docs(query, top_k=3)`. Bind query parameters and cap the requested date range and result size. Test exact answers against SQLite, empty windows, invalid dates and unsupported geography.
- [ ] Build a small corpus from official NASA FIRMS dataset definitions, MODIS/VIIRS notes and the challenge brief. Record source URL and section for each passage. Use BM25 or TF-IDF retrieval for the first version and test that queries about hotspots versus fires and MODIS versus VIIRS retrieve the relevant source.
- [x] Implement `answer(question, evidence, client) -> str`. Use an explicit allowlist and validate tool calls. Let the agent call at most three tools per question. A numeric answer must identify the data window and metric or count used. A methodological answer must cite the retrieved source. For out-of-scope questions, state the supported region and dates. Use records and retrieved excerpts as evidence, with fixed application instructions controlling execution.
- [x] Use a configurable vLLM OpenAI-compatible endpoint with a locally available instruction model. Validate endpoint, model ID and task outputs with a live smoke test. Use structured routing if native tool calls are unavailable. If the model server is unavailable, surface that status and keep the charts and raw evidence panel usable. Test with a stub client and report live integration separately. Start with the smallest model that passes the examples. Keep the model outside of the ingest and evaluation pipeline.

## 4. Verify and prepare the demo

- [ ] Run `python -m pytest -q`, rebuild the outputs from the original archives, and launch `streamlit run app.py`. Check the calendar, chart labels, year filtering and held-out metrics.
- [x] Try three questions in the running app. “Which 2025 week had the largest calibration error?” “Did XGBoost beat linear regression?” “Why can the sensors report different counts?” Verify the first two against the generated tables and the third against NASA documentation. Record test output and actual model results in the README.
- [ ] Record a short walkthrough that starts with the calendar, shows one held-out result, then asks the agent to explain the biggest gap. The first runnable milestone is the data pipeline and calendar. If time is tight, complete that milestone and report agent capabilities only after they are verified.

## Continuation status — 2026-11-15

Verified data and calibration tasks are checked above. See `docs/progress.md` for earlier implementation history and `data/verification.json` for the current integrity report.

- Rebuilt from both original archives with XGBoost 3.4.1: 104 training weeks, 51 held-out weeks. MAE: median 184.2941; linear 163.9575; XGBoost 80.8117. Signed mean error: XGBoost −1.9552 detections.
- Reconciled 2025-08-04–2025-08-10 with raw CSVs: Terra 41, Aqua 22, VIIRS 318. Offline integrity and XGBoost execution checks pass.
- Fixed colored empty calendar cells belonging to adjacent months. Regression coverage retains colors on included Mondays. Suite: 53 passed, dependency deprecation warnings only.
- User supplied a screenshot of the running 2025 dashboard. This confirms a user-side launch; the in-process tests cover year filtering and unavailable analyst states. A complete browser walkthrough is still pending.
- NASA retrieval corpus and stub-tested analyst exist. The challenge brief is missing, so the full corpus task stays open. Live vLLM endpoint/model, three live question checks, and walkthrough recording stay open; `full_plan_complete` remains false.

### Live analyst follow-up

The Qwen/Qwen2.5-1.5B-Instruct endpoint is reachable on the user's machine, but repeated smoke runs did not pass all tasks. The live vLLM client now uses the structured-routing option above: one typed classification, one Python evidence operation, numeric rendering directly from results, and method-only synthesis for documentation questions. This replaces the unsuccessful free-form tool loop for live requests. Live validation remains unchecked until the revised flow passes the three questions.

The 08:03:08Z user-side Qwen run passed automated checks and both numeric answers were reviewed against saved results. Manual review rejected a false claim in its generated methodological prose. Method output now displays curated source passages selected by document ID. This revised flow still needs a live rerun, so full live validation remains open. See `data/live-review.json`.

### Live validation accepted — 2026-11-15 08:06:39 UTC

Qwen/Qwen2.5-1.5B-Instruct passed all three demo questions in the user-executed smoke test. Inspected the saved report and independently reconciled numeric answers with SQLite; the methodological output matches the selected curated NASA passages and correct pixel/detection interpretation. The endpoint/model integration task is now checked. This validates the three examples, not general question-answering accuracy. Evidence: `data/live-smoke.json`, `data/live-review.json`. Browser-based question checks, challenge brief ingestion and walkthrough recording remain open.

### User-confirmed dashboard questions

The user submitted all three demo questions through the running Streamlit dashboard and shared their rendered answers. The largest linear-error answer is 2025-07-07 with absolute error 870.28; model comparison reports XGBoost MAE 80.81 versus linear 163.96; the methodological answer displays the curated VIIRS-resolution and MODIS-pixel notes. These match the verified reference results. This is user-reported browser-flow verification, not agent-operated browser inspection. Walkthrough recording and challenge brief remain pending.

### Scope robustness follow-up

Additional user-side Texas/2026 probes returned unsupported 2025 outputs despite the three-case demo pass. Added conservative application-side state/year checks before model/data access; four offline refusal checks pass and the full suite passes 84 tests. Expanded `smoke_test.py` reports `scope_checks` as part of pass/fail. See `data/scope-verification.json`. The three original demo results remain valid, but expanded live scope validation is still pending.

### Automated analyst robustness benchmark

The five user-checked robustness questions are now automated in `smoke_test.py`: two numeric paraphrases, hotspot-versus-fire definition and Texas/2026 scope refusals. The report records `benchmark_version`, `model`, `benchmark_cases` and `benchmark_summary`; any failed case fails the overall run. Refusals are application checks and do not measure model scope reasoning. Full suite: 91 passed. Expanded live baseline report remains pending; the command is in README. No new data year or calibration model was added.

### Qwen baseline accepted

The expanded automated run passed at 2026-11-15T08:58:13Z. Saved report `data/qwen-1.5b-benchmark.json` identifies Qwen/Qwen2.5-1.5B-Instruct, benchmark version 1, five-case score 5/5, all three demo checks passed and four scope checks passed without model dispatch. This supersedes the pending expanded-baseline note above. Challenge brief and walkthrough recording remain pending.

### Second-model comparison recorded

Qwen2.5-7B completed the same benchmark version 1 with 4/5, versus 1.5B at 5/5. The larger model falsely refused the supported hotspot-definition question; both models passed all three original demos. Keep 1.5B as the working baseline for the current workflow. Reports remain unchanged; see `docs/model-comparison.md`. This small example suite does not establish a general model ranking.
