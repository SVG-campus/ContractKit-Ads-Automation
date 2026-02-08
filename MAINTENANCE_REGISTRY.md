# ContractKit Infinity Engine - Maintenance Registry

> **Purpose**: This file tracks all optimized components that have been LOCKED
> (no further improvement possible with current data/methods). Each locked item
> includes instructions for annual review to ensure it hasn't gone stale.
>
> **Cadence**: Review ALL locked items once per year. Review ACTIVE items every 2 months.
>
> **Last Updated**: 2026-02-07
> **Total Optimization Cycles Run**: 1

---

## LOCKED Components (Annual Review Required)

*No components locked yet. All are still being optimized.*

---

## ACTIVE Components (Optimized Every ~2 Months)

- **breadth.entity_coverage**: Which Google Ads entities we fetch (campaigns, ad_groups, keywords, search_terms, ads)
  - Current Best: campaigns, ad_groups, keywords, search_terms, ads (5/8 possible)
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

- **breadth.external_signals**: External data sources beyond Google Ads (Google Trends, seasonality, competitor data)
  - Current Best: Not yet integrated. Candidates: Google Trends, Stripe revenue, holiday calendar
  - Cycles without improvement: 1/5 (locks at 5)
  - Last optimized: None

- **breadth.segmentation_depth**: How granularly we segment data (campaign, ad_group, keyword, search_term, hour, device)
  - Current Best: keyword x daily (50 keywords x 30 days)
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **breadth.time_range**: How far back we pull data (currently LAST_30_DAYS)
  - Current Best: 29 days
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

- **foundation.attribution_architecture**: Neural proxy architecture (48->24->1), training epochs, noise injection std
  - Current Best: Entropy=0.903, Top driver: impressions=0.5737
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **foundation.causal_graph_structure**: The DAG edges defining causal relationships (QS->CTR, CPC->Cost, etc.)
  - Current Best: None
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **foundation.discovery_anchors**: Which features are 'trusted anchors' vs 'suspects' in the Double ML proxy kill
  - Current Best: 17 survivors, 0 proxies killed
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **foundation.intervention_vault_potencies**: Potency values for each intervention (bid +20% -> +0.20 impressions, etc.)
  - Current Best: 13 interventions with estimated potencies (needs A/B validation)
  - Cycles without improvement: 1/5 (locks at 5)
  - Last optimized: None

- **foundation.optimization_hyperparameters**: BO iterations, GP kernel, DE popsize, LightGBM params for Infinity Loop
  - Current Best: BO: n_init=5, n_iter=15, GP(Matern2.5) | DE: popsize=10, maxiter=30
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **foundation.regime_detection_sensitivity**: PELT penalty, zombie/decay/surge ratio thresholds
  - Current Best: PELT pen=10, 2 changepoints, 17 features analyzed
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **foundation.validation_thresholds**: TVF scanner thresholds (drift_alpha, max_null_spike, iforest_contamination, etc.)
  - Current Best: drift_alpha=0.01, max_null_spike=0.15, iforest_contamination=0.03
  - Cycles without improvement: 0/3 (locks at 3)
  - Last optimized: 2026-02-07T23:48:57.123682

- **operational.control_surface_format**: Structure of the output JSON/CSV control surface
  - Current Best: None
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

- **operational.dashboard_metrics**: Which metrics and KPIs are shown in the terminal dashboard
  - Current Best: None
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

- **operational.feature_engineering**: Engineered features (kw_length_x_exact, cpc_x_qs, ctr_x_impr, etc.)
  - Current Best: None
  - Cycles without improvement: 1/3 (locks at 3)
  - Last optimized: None

- **operational.gaql_queries**: Google Ads Query Language queries for data ingestion
  - Current Best: None
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

- **operational.synthetic_data_model**: Parameters of the synthetic data generator for testing
  - Current Best: {"clicks": {"mean": 36.940666666666665, "std": 52.74547566402486}, "impressions": {"mean": 157.848, "std": 68.39864383879252}, "ctr": {"mean": 0.21419891238760025, "std": 0.11282348123308532}, "avg_cp
  - Cycles without improvement: 1/2 (locks at 2)
  - Last optimized: None

---

## How to Run a Meta-Optimization Cycle

```bash
# In Cursor terminal:
python meta_optimizer.py

# Or ask the AI agent:
# 'Run a meta-optimization cycle on the ContractKit Infinity Engine'
```

The meta-optimizer will:
1. Check which components are still ACTIVE
2. For each active component, try to expand/improve it
3. If improvement found -> record it, reset stagnation counter
4. If no improvement -> increment stagnation counter
5. If stagnation counter hits threshold -> LOCK the component
6. Update this registry file

---

## Optimization Schedule

| Cycle | Target Date | Status |
|-------|-------------|--------|
| 1 | 2026-02-07 | **NOW** (initial cycle) |
| 2 | 2026-04-08 | Scheduled |
| 3 | 2026-06-07 | Scheduled |
| 4 | 2026-08-06 | Scheduled |
| 5 | 2026-10-05 | Scheduled |
| 6 | 2026-12-04 | Scheduled |
| 7 | 2027-02-02 | Scheduled |
| 8 | 2027-04-03 | Scheduled |
| 9 | 2027-06-02 | Scheduled |
| 10 | 2027-08-01 | Scheduled |
| 11 | 2027-09-30 | Scheduled |
| 12 | 2027-11-29 | Scheduled |

---

## Annual Deep Review Checklist

Run this checklist once per year (or when Google Ads makes major changes):

- [ ] **Google Ads API Version**: Check if API version needs updating
- [ ] **OAuth Credentials**: Rotate credentials if needed
- [ ] **Developer Token**: Verify still active, check usage quotas
- [ ] **Causal Graph**: Re-derive DAG from fresh data
- [ ] **Validation Thresholds**: Recalibrate against latest 90 days
- [ ] **Intervention Potencies**: Validate against actual A/B test results
- [ ] **Feature Engineering**: Check for new Google Ads fields/metrics
- [ ] **LightGBM Hyperparameters**: Re-tune with Optuna on fresh data
- [ ] **GAQL Queries**: Verify against latest Google Ads API docs
- [ ] **External Signals**: Add any newly available data sources
- [ ] **Competitor Landscape**: Update regime detection for new competitors
- [ ] **Product Changes**: Update $19/mo if pricing changed
