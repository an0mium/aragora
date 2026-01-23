# Aragora 2.0.2 Release Notes

> **Deprecated:** Historical release notes. See `docs/RELEASE_NOTES.md` for the
> current release history.

**Release Date:** January 19, 2026
**Version:** 2.0.2
**Codename:** UI Enhancement Release

---

## Overview

Aragora 2.0.2 focuses on surfacing existing backend capabilities through the UI. This release adds 3 new pages and enhances 4 existing pages to expose powerful features that were previously only accessible via API. The codebase now includes 74 fully integrated features with 34,400+ tests.

---

## What's New

### New Pages

#### Decision Receipt Browser (`/receipts`)
Enterprise compliance feature for audit trails:
- Browse gauntlet receipts with verdict badges (PASS/FAIL/WARN)
- Filter by date range, verdict type, and severity
- Full receipt viewer with artifact hash verification
- Export as HTML/JSON for compliance documentation
- Provenance chain visualization showing decision lineage

```bash
# Generate a receipt via CLI
aragora gauntlet policy.yaml --output receipt.html
```

#### Training Data Explorer (`/training/explorer`)
ML engineers can now browse and export debate data for model fine-tuning:
- Dataset statistics dashboard (total debates, topic breakdown, win rates)
- Format documentation for SFT, DPO, and Gauntlet exports
- Live preview of training examples with confidence scores
- Confidence threshold filtering slider
- Bulk export with format selection

#### Model Registry (`/training/models`)
Track fine-tuned specialist models:
- Job listing with status filters (pending, training, completed, failed)
- Performance metrics: ELO rating, win rate, accuracy, loss curves
- Training progress visualization
- Start/cancel job controls
- Artifact download links

---

### Enhanced Pages

#### Risk Heatmap (`/gauntlet`)
New visualization for security and compliance testing:
- Interactive risk heatmap grid
- Rows = categories (Security, Logic, Compliance)
- Columns = severity levels (critical, high, medium, low)
- Click cell to filter findings
- Export as SVG for reports

#### Belief Network Dashboard (`/crux`)
Enhanced crux analysis with network visualization:
- New tabs: Cruxes | Load-Bearing | Contested | Stats
- Contested claims panel showing confidence deltas
- Graph statistics (nodes, edges, depth, centrality)
- Network export options (JSON, GraphML, CSV)
- Enhanced sensitivity analysis

#### Episode Generator (`/broadcast`)
Generate podcast episodes from debates:
- Debate selector dropdown
- Custom title and description fields
- Optional video generation flag
- Generation progress indicator
- Direct play/download after generation

#### Knowledge Graph Export (`/knowledge`)
Export and monitor knowledge freshness:
- Export buttons (D3 JSON, GraphML formats)
- Staleness panel with aging/stale/expired indicators
- Color-coded freshness badges
- Batch refresh operations for stale facts

---

### Backend Enhancements

#### RLM Training Module
New reinforcement learning training infrastructure:
- `aragora/rlm/training/buffer.py` - Experience replay storage with prioritization
- `aragora/rlm/training/reward.py` - Reward signal computation from debate outcomes
- Entropy bonuses for exploration
- Temporal discounting for long-horizon learning
- Margin-based rewards for preference modeling

---

### Navigation Improvements

- Added [EXPLORER] and [MODELS] links to `/training`
- Added [RECEIPTS] link to `/gauntlet`
- Added [RECEIPTS], [KNOWLEDGE], [BROADCAST], [VERIFY] links to main dashboard Quick Links Bar
- Improved cross-page navigation consistency

---

## Quick Access

| Feature | Path | Description |
|---------|------|-------------|
| Receipt Browser | `/receipts` | Compliance audit trails |
| Training Explorer | `/training/explorer` | ML data preview/export |
| Model Registry | `/training/models` | Fine-tuned model tracking |
| Risk Heatmap | `/gauntlet` | Security visualization |
| Belief Network | `/crux` | Claim network analysis |
| Episode Generator | `/broadcast` | Podcast from debates |
| Knowledge Export | `/knowledge` | Graph export & staleness |
| Audit Logs | `/admin/audit` | System event export |
| Calibration | `/calibration` | Agent confidence tracking |
| Verification | `/verification` | Formal proof explorer |

---

## Statistics

| Metric | v2.0.1 | v2.0.2 | Change |
|--------|--------|--------|--------|
| Fully Integrated Features | 63 | 74 | +11 |
| UI Pages | 45+ | 48+ | +3 |
| Tests | 34,300+ | 34,400+ | +100 |
| Lines of Code | 440,000+ | 443,000+ | +3,000 |

---

## Upgrading

No breaking changes from v2.0.1. Upgrade by pulling latest and rebuilding:

```bash
git pull origin main
cd aragora/live && npm install && npm run build
pip install -e ".[dev]"
```

---

## What's Next (2.1 Roadmap)

- **Control Plane** integration with verticals catalog
- **Connectors Page** for external data source management
- **Enhanced tournaments** with bracket visualizations
- **Real-time collaboration** improvements
- **Mobile-responsive** optimizations

---

## Support

- Documentation: https://aragora.ai/docs
- Issues: https://github.com/an0mium/aragora/issues
- Status: See `docs/STATUS.md` for feature status
