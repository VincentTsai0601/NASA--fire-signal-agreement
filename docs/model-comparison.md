# Analyst model comparison

Both reports use benchmark version 1, the same five questions, the same application routing workflow, and the same 2023–2025 evidence. These are user-executed live runs, inspected from the saved JSON reports.

| Model | Original demo cases | Five-case benchmark | Application scope checks | Overall run |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 3/3 | 5/5 | 4/4 | Passed |
| Qwen2.5-7B-Instruct | 3/3 | 4/5 | 4/4 | Failed |

The 7B model incorrectly routed “Do 100 hotspot detections mean 100 separate fires?” to `unsupported`. This is a false refusal: the question is a supported methodological explanation, whose reference notes distinguish hotspot pixels from individual fires. Its numeric routes and the sensor-explanation demo passed. The first 7B attempt failed to connect before the server was ready; the completed second run is the comparison reported above.

Use 1.5B as the working demo baseline for now. This small benchmark does not establish that 1.5B is generally more capable than 7B. Two benchmark cases and all four scope checks are enforced by application guards and do not measure model scope reasoning. Numeric results are computed by Python, and methodological output uses curated source passages; the model is being evaluated for routing and source selection. No latency measurements were collected, so no speed claim is made.

The original 1.5B server used its default context settings; the recommended 7B launch capped context at 4096 and GPU memory utilization at 0.85. No configuration-normalized performance comparison is claimed. If routing prompts are revised, rerun both models and version the benchmark before drawing a new comparison.

Reports:

- `data/qwen-1.5b-benchmark.json` — 2026-11-15T08:58:13Z
- `data/qwen-7b-benchmark.json` — 2026-11-15T09:01:53Z
