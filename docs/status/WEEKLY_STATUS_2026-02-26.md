# Weekly Status - 2026-02-26

Scope: Epistemic hygiene + settlement reliability lane (runtime KPI automation).

## KPI Automation

1. Added weekly extraction script:
   - `scripts/extract_weekly_epistemic_kpis.py`
2. Added weekly scheduled workflow:
   - `.github/workflows/weekly-epistemic-kpis.yml`
3. Workflow output artifacts:
   - `weekly-epistemic-kpis.json`
   - `weekly-epistemic-kpis.md`

## Runtime Signals Tracked

1. Settlement review success rate.
2. Settlement unresolved due count (last run).
3. Calibration updates realized (`correct + incorrect`).
4. Oracle stream stall rate (`stalls_total / sessions_started`).

## Threshold Defaults

1. Settlement success rate: `>= 0.99`
2. Oracle stall rate: `<= 0.02`
3. Calibration updates realized: `>= 1`

## Notes

1. Workflow supports manual `strict=true` runs that fail on KPI threshold breaches.
2. Scheduled runs always publish artifacts and summary for weekly review.
