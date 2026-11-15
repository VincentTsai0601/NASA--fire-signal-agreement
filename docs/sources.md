# Sources and limits

The two local standard FIRMS archives are the source of all counts. SHA-256 hashes, raw row counts, acquisition-date ranges, geographic bounds, and schema version are stored in `data/build-report.json` and SQLite metadata. NRT files are excluded. Raw data stays local; the optional analyst sends aggregate weekly evidence and retrieved passages to the configured endpoint.

The retrieval corpus in `corpus.json` contains short paraphrases, not a complete copy of NASA documentation. Source sections were checked on 2026-11-15:

- [NASA MODIS pixel definition](https://firms2.modaps.eosdis.nasa.gov/descriptions/FIRMS_MODIS_Firehotspots.html): a detection locates a pixel center; one pixel can contain more than one fire.
- [NASA VIIRS product notes](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html): the 375 m product responds better to relatively small fires than MODIS.
- [NASA archive delivery](https://firms.modaps.eosdis.nasa.gov/content/active_fire/): standard processing replaces NRT records with a delay.
- Local implementation plan: the supplied study scope and modeling rules. No separate challenge brief was supplied; it has not been fabricated or indexed.

The study box includes areas beyond California. Counts include all supplied detection types and confidence levels, with no deduplication or observing-coverage correction. Zero rows cannot prove absence of fire. Estimates calibrate sensor counts, not burned area, fire severity, distinct fires, or future fire risk. Cloud cover, overpass timing and sensor characteristics can affect observations; this project does not quantify those effects.

Complete UTC Monday–Sunday weeks wholly within 2023–2025 are used, excluding the week starting 2024-12-30 because it crosses the train/test boundary. The week starting 2023-12-25 is retained in training. No geographic generalization is evaluated. Model parameters are fixed before testing; no 2025 target enters model fitting. Signed mean error is estimate minus observation; signed weekly gap is observation minus estimate.
