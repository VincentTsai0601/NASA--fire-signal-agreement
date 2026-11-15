# Verified demo answers

These reference answers use generated SQLite tables and local NASA source retrieval, without a language model.

## Which 2025 week had the largest calibration error?

median: week starting 2025-07-07 UTC, observed VIIRS 1,631, estimated VIIRS 243.00; absolute error 1,388.00, signed gap (observed − estimated) 1,388.00. linear: week starting 2025-07-07 UTC, observed VIIRS 1,631, estimated VIIRS 760.72; absolute error 870.28, signed gap (observed − estimated) 870.28. xgboost: week starting 2025-07-14 UTC, observed VIIRS 1,271, estimated VIIRS 803.07; absolute error 467.93, signed gap (observed − estimated) 467.93. Window: complete 2025 weeks. Metric: satellite hotspot detections. Source: SQLite predictions; ties use the earliest week.

## Did XGBoost beat linear regression?

Yes. XGBoost MAE 80.81; linear regression MAE 163.96. Lower MAE is better. Window: complete 2025 held-out weeks. Metric: satellite hotspot detections. Source: SQLite metrics.

## Why can the sensors report different counts?

VIIRS 375 m data complements MODIS detections. Its improved resolution responds better to relatively small fires and maps large fire perimeters in more detail, so MODIS and VIIRS counts can differ. [VIIRS spatial resolution](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)

MODIS records locate the center of a 1 km hotspot pixel, not necessarily an individual fire. One pixel can contain multiple fires. Satellite hotspot detections are not a count of distinct fires. [MODIS pixel definition](https://firms2.modaps.eosdis.nasa.gov/descriptions/FIRMS_MODIS_Firehotspots.html)

FIRMS replaces near real time NRT archive records with standard MODIS and VIIRS products as they become available, usually after two to three months but possibly up to five months. [Archive latency](https://firms.modaps.eosdis.nasa.gov/content/active_fire/)
