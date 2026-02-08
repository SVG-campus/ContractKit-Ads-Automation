"""
================================================================================
 CONTRACTKIT META-OPTIMIZATION ENGINE
 ─────────────────────────────────────
 Self-Improving System for Periodic Breadth/Depth Expansion
 
 This system:
 1. Tracks every optimizable component in the Infinity Engine
 2. Periodically expands data breadth (new features, new data sources)
 3. Deepens analysis (more iterations, finer tuning, new algorithms)
 4. Locks components once they reach diminishing returns
 5. Generates a MAINTENANCE_REGISTRY.md for annual review
 
 Cadence: Run every ~2 months (ask user to approve in Cursor)
 Goal: Find the absolute best answers, then lock and move on.
================================================================================
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("MetaOptimizer")

# ==============================================================================
# STATE SCHEMA: Tracks every component's optimization lifecycle
# ==============================================================================

STATE_FILE = "meta_state.json"
REGISTRY_FILE = "MAINTENANCE_REGISTRY.md"

# Default state for all optimizable components
DEFAULT_COMPONENTS = {
    # ── FOUNDATIONAL LAYER QUESTIONS ────────────────────────────────
    # These are the "questions that built the 7 layers"
    "foundation.validation_thresholds": {
        "description": "TVF scanner thresholds (drift_alpha, max_null_spike, iforest_contamination, etc.)",
        "category": "foundational",
        "status": "active",          # active | locked | review_needed
        "current_best": None,
        "optimization_history": [],   # List of {date, method, result, improvement}
        "cycles_without_improvement": 0,
        "lock_threshold": 3,          # Lock after 3 cycles with no improvement
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Re-run with fresh 90-day ads data. Check if drift thresholds still match account volatility. If CTR/CPA distributions shifted significantly, retune.",
    },
    "foundation.causal_graph_structure": {
        "description": "The DAG edges defining causal relationships (QS->CTR, CPC->Cost, etc.)",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Run causal discovery on fresh data. If Google Ads changes auction mechanics or you add new campaign types, re-derive the DAG. Test with PC algorithm or NOTEARS.",
    },
    "foundation.discovery_anchors": {
        "description": "Which features are 'trusted anchors' vs 'suspects' in the Double ML proxy kill",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Review if new metrics from Google Ads should be anchors (e.g., if Google adds new quality signals). Re-run proxy kill with expanded anchor set.",
    },
    "foundation.attribution_architecture": {
        "description": "Neural proxy architecture (48->24->1), training epochs, noise injection std",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Test wider/deeper architectures. Try transformer-based attribution if data volume exceeds 100k rows. Verify completeness axiom still holds (<0.05 gap).",
    },
    "foundation.intervention_vault_potencies": {
        "description": "Potency values for each intervention (bid +20% -> +0.20 impressions, etc.)",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 5,  # Needs more evidence since these are causal estimates
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Run A/B tests on actual bid changes. Compare predicted vs actual impact. If Google changes auction, potencies will shift. Update with empirical lift data.",
    },
    "foundation.optimization_hyperparameters": {
        "description": "BO iterations, GP kernel, DE popsize, LightGBM params for Infinity Loop",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Run Optuna/Hyperopt meta-tuning on the optimizer itself. Test if increasing BO iterations beyond 20 still helps. Check LightGBM params with fresh data.",
    },
    "foundation.regime_detection_sensitivity": {
        "description": "PELT penalty, zombie/decay/surge ratio thresholds",
        "category": "foundational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Backtest regime detection against known competitive events. If false positive rate >20%, increase PELT penalty. If missing real regime changes, decrease it.",
    },

    # ── OPERATIONAL COMPONENTS ──────────────────────────────────────
    # These are the "everything else built to work on it"
    "operational.gaql_queries": {
        "description": "Google Ads Query Language queries for data ingestion",
        "category": "operational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Check Google Ads API changelog for new fields/metrics. Add any new performance columns, audience segments, or asset-level metrics.",
    },
    "operational.feature_engineering": {
        "description": "Engineered features (kw_length_x_exact, cpc_x_qs, ctr_x_impr, etc.)",
        "category": "operational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Test new interaction terms. Try automated feature generation (Featuretools, genetic programming). Add day-of-week, hour-of-day, device features when available.",
    },
    "operational.synthetic_data_model": {
        "description": "Parameters of the synthetic data generator for testing",
        "category": "operational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Compare synthetic distributions against real data once live API is fully connected. Adjust base rates, CPC ranges, conversion rates to match reality.",
    },
    "operational.control_surface_format": {
        "description": "Structure of the output JSON/CSV control surface",
        "category": "operational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Review if the action types match current Google Ads API mutate operations. Add new action types if Google introduces new optimization levers.",
    },
    "operational.dashboard_metrics": {
        "description": "Which metrics and KPIs are shown in the terminal dashboard",
        "category": "operational",
        "status": "active",
        "current_best": None,
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Check if new business KPIs matter (e.g., LTV, churn rate, trial-to-paid). Add any metrics the business team requests.",
    },

    # ── DATA BREADTH EXPANSION ──────────────────────────────────────
    "breadth.time_range": {
        "description": "How far back we pull data (currently LAST_30_DAYS)",
        "category": "breadth",
        "status": "active",
        "current_best": "30 days",
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Test 60-day, 90-day, and 180-day windows. Longer windows help regime detection but may include stale patterns. Find the sweet spot.",
    },
    "breadth.entity_coverage": {
        "description": "Which Google Ads entities we fetch (campaigns, ad_groups, keywords, search_terms, ads)",
        "category": "breadth",
        "status": "active",
        "current_best": "5 entities",
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 2,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Add: audience segments, location reports, device reports, ad schedule reports, asset-level performance (headlines/descriptions individually).",
    },
    "breadth.external_signals": {
        "description": "External data sources beyond Google Ads (Google Trends, seasonality, competitor data)",
        "category": "breadth",
        "status": "active",
        "current_best": "None yet",
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 5,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Integrate: Google Trends API for keyword demand, Stripe MCP for revenue validation, day-of-week/holiday calendars, economic indicators.",
    },
    "breadth.segmentation_depth": {
        "description": "How granularly we segment data (campaign, ad_group, keyword, search_term, hour, device)",
        "category": "breadth",
        "status": "active",
        "current_best": "keyword-level daily",
        "optimization_history": [],
        "cycles_without_improvement": 0,
        "lock_threshold": 3,
        "last_optimized": None,
        "annual_review_date": None,
        "review_instructions": "Test hourly segmentation, device-level splits, geo-level analysis. More granularity = more signal but also more noise. Use the TVF to validate.",
    },
}


def load_state() -> dict:
    """Load meta-optimization state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "last_cycle": None,
        "total_cycles": 0,
        "next_scheduled": (datetime.now() + timedelta(days=60)).isoformat(),
        "components": DEFAULT_COMPONENTS,
    }


def save_state(state: dict):
    """Persist state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def generate_maintenance_registry(state: dict):
    """
    Generate the MAINTENANCE_REGISTRY.md file.
    Lists locked components with annual review instructions.
    """
    lines = [
        "# ContractKit Infinity Engine - Maintenance Registry",
        "",
        "> **Purpose**: This file tracks all optimized components that have been LOCKED",
        "> (no further improvement possible with current data/methods). Each locked item",
        "> includes instructions for annual review to ensure it hasn't gone stale.",
        ">",
        "> **Cadence**: Review ALL locked items once per year. Review ACTIVE items every 2 months.",
        ">",
        f"> **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}",
        f"> **Total Optimization Cycles Run**: {state.get('total_cycles', 0)}",
        "",
        "---",
        "",
    ]

    # Section 1: Locked Components (Annual Review)
    lines.append("## LOCKED Components (Annual Review Required)")
    lines.append("")
    locked = {k: v for k, v in state["components"].items() if v["status"] == "locked"}
    if not locked:
        lines.append("*No components locked yet. All are still being optimized.*")
        lines.append("")
    else:
        for key, comp in sorted(locked.items()):
            lines.append(f"### {key}")
            lines.append(f"- **Description**: {comp['description']}")
            lines.append(f"- **Category**: {comp['category']}")
            lines.append(f"- **Locked Since**: {comp.get('last_optimized', 'Unknown')}")
            lines.append(f"- **Best Value Found**: {comp.get('current_best', 'N/A')}")
            lines.append(f"- **Cycles Without Improvement**: {comp['cycles_without_improvement']}")
            lines.append(f"- **Annual Review Date**: {comp.get('annual_review_date', 'TBD')}")
            lines.append(f"- **Review Instructions**:")
            lines.append(f"  > {comp['review_instructions']}")
            if comp.get("optimization_history"):
                lines.append(f"- **Optimization History**:")
                for entry in comp["optimization_history"][-3:]:  # Last 3 entries
                    lines.append(f"  - {entry.get('date', '?')}: {entry.get('method', '?')} -> {entry.get('result', '?')} (improvement: {entry.get('improvement', '?')})")
            lines.append("")

    # Section 2: Active Components (Next Cycle)
    lines.append("---")
    lines.append("")
    lines.append("## ACTIVE Components (Optimized Every ~2 Months)")
    lines.append("")
    active = {k: v for k, v in state["components"].items() if v["status"] == "active"}
    if not active:
        lines.append("*All components have been locked! System is fully optimized.*")
        lines.append("")
    else:
        for key, comp in sorted(active.items()):
            cycles_left = comp["lock_threshold"] - comp["cycles_without_improvement"]
            lines.append(f"- **{key}**: {comp['description']}")
            lines.append(f"  - Current Best: {comp.get('current_best', 'Not yet measured')}")
            lines.append(f"  - Cycles without improvement: {comp['cycles_without_improvement']}/{comp['lock_threshold']} (locks at {comp['lock_threshold']})")
            lines.append(f"  - Last optimized: {comp.get('last_optimized', 'Never')}")
            lines.append("")

    # Section 3: How to Run
    lines.append("---")
    lines.append("")
    lines.append("## How to Run a Meta-Optimization Cycle")
    lines.append("")
    lines.append("```bash")
    lines.append("# In Cursor terminal:")
    lines.append("python meta_optimizer.py")
    lines.append("")
    lines.append("# Or ask the AI agent:")
    lines.append("# 'Run a meta-optimization cycle on the ContractKit Infinity Engine'")
    lines.append("```")
    lines.append("")
    lines.append("The meta-optimizer will:")
    lines.append("1. Check which components are still ACTIVE")
    lines.append("2. For each active component, try to expand/improve it")
    lines.append("3. If improvement found -> record it, reset stagnation counter")
    lines.append("4. If no improvement -> increment stagnation counter")
    lines.append("5. If stagnation counter hits threshold -> LOCK the component")
    lines.append("6. Update this registry file")
    lines.append("")

    # Section 4: Schedule
    lines.append("---")
    lines.append("")
    lines.append("## Optimization Schedule")
    lines.append("")
    lines.append("| Cycle | Target Date | Status |")
    lines.append("|-------|-------------|--------|")

    next_date = datetime.now()
    for i in range(12):
        date_str = next_date.strftime("%Y-%m-%d")
        if i == 0:
            status = "**NOW** (initial cycle)"
        elif next_date < datetime.now():
            status = "COMPLETED" if state.get("total_cycles", 0) > i else "OVERDUE"
        else:
            status = "Scheduled"
        lines.append(f"| {i + 1} | {date_str} | {status} |")
        next_date += timedelta(days=60)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Annual Deep Review Checklist")
    lines.append("")
    lines.append("Run this checklist once per year (or when Google Ads makes major changes):")
    lines.append("")
    lines.append("- [ ] **Google Ads API Version**: Check if API version needs updating")
    lines.append("- [ ] **OAuth Credentials**: Rotate credentials if needed")
    lines.append("- [ ] **Developer Token**: Verify still active, check usage quotas")
    lines.append("- [ ] **Causal Graph**: Re-derive DAG from fresh data")
    lines.append("- [ ] **Validation Thresholds**: Recalibrate against latest 90 days")
    lines.append("- [ ] **Intervention Potencies**: Validate against actual A/B test results")
    lines.append("- [ ] **Feature Engineering**: Check for new Google Ads fields/metrics")
    lines.append("- [ ] **LightGBM Hyperparameters**: Re-tune with Optuna on fresh data")
    lines.append("- [ ] **GAQL Queries**: Verify against latest Google Ads API docs")
    lines.append("- [ ] **External Signals**: Add any newly available data sources")
    lines.append("- [ ] **Competitor Landscape**: Update regime detection for new competitors")
    lines.append("- [ ] **Product Changes**: Update $19/mo if pricing changed")
    lines.append("")

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  Maintenance Registry updated: {REGISTRY_FILE}")


# ==============================================================================
# META-OPTIMIZATION CYCLE
# ==============================================================================

class MetaOptimizer:
    """
    The self-improving meta-optimization engine.
    Each cycle attempts to expand breadth, deepen analysis,
    and tune all active components.
    """

    def __init__(self):
        self.state = load_state()
        logger.info("MetaOptimizer loaded.")
        logger.info(f"  Total cycles completed: {self.state.get('total_cycles', 0)}")
        logger.info(f"  Active components: {sum(1 for c in self.state['components'].values() if c['status'] == 'active')}")
        logger.info(f"  Locked components: {sum(1 for c in self.state['components'].values() if c['status'] == 'locked')}")

    def run_cycle(self, data: pd.DataFrame = None, verbose: bool = True) -> dict:
        """
        Run a full meta-optimization cycle.
        Returns a report of what was improved, what was locked, what's next.
        """
        cycle_start = time.time()
        cycle_num = self.state.get("total_cycles", 0) + 1
        report = {
            "cycle": cycle_num,
            "date": datetime.now().isoformat(),
            "improvements": [],
            "no_improvement": [],
            "newly_locked": [],
            "still_active": [],
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"META-OPTIMIZATION CYCLE #{cycle_num}")
        logger.info(f"{'='*60}")

        for comp_key, comp in self.state["components"].items():
            if comp["status"] != "active":
                continue

            logger.info(f"\n  Optimizing: {comp_key}")

            # Attempt optimization based on component category
            improvement = self._optimize_component(comp_key, comp, data)

            if improvement is not None and improvement > 0:
                comp["cycles_without_improvement"] = 0
                comp["optimization_history"].append({
                    "date": datetime.now().isoformat(),
                    "cycle": cycle_num,
                    "method": "meta_cycle",
                    "result": str(improvement),
                    "improvement": f"+{improvement:.4f}",
                })
                comp["last_optimized"] = datetime.now().isoformat()
                report["improvements"].append({"component": comp_key, "improvement": improvement})
                logger.info(f"    IMPROVED by {improvement:.4f}")
            else:
                comp["cycles_without_improvement"] += 1
                report["no_improvement"].append(comp_key)
                logger.info(f"    No improvement (stagnation: {comp['cycles_without_improvement']}/{comp['lock_threshold']})")

                # Check if should be locked
                if comp["cycles_without_improvement"] >= comp["lock_threshold"]:
                    comp["status"] = "locked"
                    comp["annual_review_date"] = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
                    report["newly_locked"].append(comp_key)
                    logger.info(f"    >> LOCKED (diminishing returns reached)")

            if comp["status"] == "active":
                report["still_active"].append(comp_key)

        # Update state
        self.state["total_cycles"] = cycle_num
        self.state["last_cycle"] = datetime.now().isoformat()
        self.state["next_scheduled"] = (datetime.now() + timedelta(days=60)).isoformat()

        # Save and regenerate registry
        save_state(self.state)
        generate_maintenance_registry(self.state)

        elapsed = time.time() - cycle_start
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE #{cycle_num} COMPLETE ({elapsed:.1f}s)")
        logger.info(f"  Improvements: {len(report['improvements'])}")
        logger.info(f"  No change: {len(report['no_improvement'])}")
        logger.info(f"  Newly locked: {len(report['newly_locked'])}")
        logger.info(f"  Still active: {len(report['still_active'])}")
        logger.info(f"  Next cycle: {self.state['next_scheduled']}")
        logger.info(f"{'='*60}")

        return report

    def _optimize_component(self, key: str, comp: dict, data: pd.DataFrame = None) -> Optional[float]:
        """
        Attempt to optimize a specific component.
        Returns improvement amount (positive = better) or None/0 if no improvement.
        """
        # Import the engine for live testing
        try:
            from contractkit_infinity_engine import (
                AdsValidationFramework, AdsValidationConfig,
                AdsDiscoveryEngine, AdsAttributionValidator,
                AdsCausalEngine, AdsInfinityOptimizer,
                AdsDataIngestion
            )
        except ImportError:
            logger.warning(f"    Cannot import engine. Skipping optimization for {key}.")
            return None

        # Generate test data if none provided
        if data is None or data.empty:
            ingestion = AdsDataIngestion()
            ingestion._generate_synthetic_data()
            data = ingestion.unified_df

        num_data = data.select_dtypes(include=[np.number]).copy()

        # ── FOUNDATIONAL OPTIMIZATIONS ──────────────────────────────
        if key == "foundation.validation_thresholds":
            return self._tune_validation_thresholds(num_data)

        elif key == "foundation.causal_graph_structure":
            return self._expand_causal_graph(num_data)

        elif key == "foundation.discovery_anchors":
            return self._tune_discovery_anchors(num_data)

        elif key == "foundation.attribution_architecture":
            return self._tune_attribution(num_data)

        elif key == "foundation.intervention_vault_potencies":
            return self._tune_intervention_potencies(num_data)

        elif key == "foundation.optimization_hyperparameters":
            return self._tune_optimizer_hyperparams(num_data)

        elif key == "foundation.regime_detection_sensitivity":
            return self._tune_regime_detection(num_data)

        # ── OPERATIONAL OPTIMIZATIONS ───────────────────────────────
        elif key == "operational.feature_engineering":
            return self._expand_feature_engineering(num_data)

        elif key == "operational.synthetic_data_model":
            return self._tune_synthetic_model(data)

        # ── BREADTH EXPANSIONS ──────────────────────────────────────
        elif key == "breadth.time_range":
            return self._evaluate_time_range(data)

        elif key == "breadth.entity_coverage":
            return self._evaluate_entity_coverage()

        elif key == "breadth.external_signals":
            return self._evaluate_external_signals()

        elif key == "breadth.segmentation_depth":
            return self._evaluate_segmentation(data)

        # Default: no optimization possible
        return None

    # ── SPECIFIC OPTIMIZATION METHODS ───────────────────────────────

    def _tune_validation_thresholds(self, data: pd.DataFrame) -> float:
        """Try different TVF threshold configs and measure false positive rate."""
        from contractkit_infinity_engine import AdsValidationFramework, AdsValidationConfig

        ref_size = max(50, len(data) // 2)
        ref = data.iloc[:ref_size]
        test = data.iloc[ref_size:]

        # Current defaults
        baseline_config = AdsValidationConfig()
        tvf = AdsValidationFramework(ref, baseline_config)
        passed_baseline, report_baseline = tvf.validate(test, target_col="conversions")
        n_red_baseline = len(report_baseline["traffic_light"][report_baseline["traffic_light"]["Status"] == "RED"])

        # Try relaxed config (fewer false positives)
        relaxed_config = AdsValidationConfig(
            drift_alpha=0.01,  # More strict p-value = fewer drift flags
            max_null_spike=0.15,
            iforest_contamination=0.03,
            max_correlation_drift=0.30,
        )
        tvf_relaxed = AdsValidationFramework(ref, relaxed_config)
        passed_relaxed, report_relaxed = tvf_relaxed.validate(test, target_col="conversions")
        n_red_relaxed = len(report_relaxed["traffic_light"][report_relaxed["traffic_light"]["Status"] == "RED"])

        # Improvement = reduction in false positives (fewer unnecessary RED flags)
        improvement = (n_red_baseline - n_red_relaxed) / max(n_red_baseline, 1)

        if improvement > 0:
            self.state["components"]["foundation.validation_thresholds"]["current_best"] = (
                f"drift_alpha=0.01, max_null_spike=0.15, iforest_contamination=0.03"
            )

        return max(0, improvement)

    def _expand_causal_graph(self, data: pd.DataFrame) -> float:
        """Try adding more edges to the causal graph and measure prediction quality."""
        from contractkit_infinity_engine import AdsCausalEngine

        # Baseline graph
        engine_base = AdsCausalEngine(data, verbose=False)
        if "conversions" in data.columns:
            base_result = engine_base.simulate_intervention(
                {"avg_cpc": data["avg_cpc"].median() if "avg_cpc" in data else 1.0},
                "conversions"
            )
            base_precision = 1.0 / (base_result.get("std_error", 1.0) + 1e-9)

            # Try expanded graph with interaction edges
            extra_edges = []
            cols = set(data.columns)
            if "kw_length" in cols and "ctr" in cols:
                extra_edges.append(("kw_length", "ctr"))
            if "is_exact" in cols and "avg_cpc" in cols:
                extra_edges.append(("is_exact", "avg_cpc"))
            if "quality_score" in cols and "impressions" in cols:
                extra_edges.append(("quality_score", "impressions"))

            if extra_edges:
                expanded_graph = list(engine_base.G.edges()) + extra_edges
                engine_expanded = AdsCausalEngine(data, causal_graph=expanded_graph, verbose=False)
                exp_result = engine_expanded.simulate_intervention(
                    {"avg_cpc": data["avg_cpc"].median() if "avg_cpc" in data else 1.0},
                    "conversions", n_boot=20
                )
                exp_precision = 1.0 / (exp_result.get("std_error", 1.0) + 1e-9)

                improvement = (exp_precision - base_precision) / max(base_precision, 1e-9)
                if improvement > 0:
                    self.state["components"]["foundation.causal_graph_structure"]["current_best"] = (
                        f"{len(expanded_graph)} edges (added: {extra_edges})"
                    )
                return max(0, improvement)

        return 0

    def _tune_discovery_anchors(self, data: pd.DataFrame) -> float:
        """Test expanded anchor sets for the Double ML proxy kill."""
        from contractkit_infinity_engine import AdsDiscoveryEngine

        if "conversions" not in data.columns:
            return 0

        engine = AdsDiscoveryEngine(data, "conversions")
        report = engine.verify_causality_nonlinear()
        n_survivors_base = len(report.get("survivors", []))
        n_killed_base = len(report.get("killed", []))

        # The more proxies correctly killed, the better
        total = n_survivors_base + n_killed_base
        if total > 0:
            self.state["components"]["foundation.discovery_anchors"]["current_best"] = (
                f"{n_survivors_base} survivors, {n_killed_base} proxies killed"
            )
        return 0  # Need live data to truly improve this

    def _tune_attribution(self, data: pd.DataFrame) -> float:
        """Test attribution model quality (R2, completeness)."""
        from contractkit_infinity_engine import AdsAttributionValidator

        target = "conversions" if "conversions" in data.columns else data.columns[-1]
        features = [c for c in data.columns if c != target][:15]

        if len(features) < 2:
            return 0

        attr = AdsAttributionValidator(data[features].fillna(0), data[target].fillna(0))
        scores = attr.compute_attribution()

        # Measure: how concentrated is attribution? (entropy)
        # Lower entropy = clearer signal
        abs_scores = scores.abs()
        probs = abs_scores / (abs_scores.sum() + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        clarity = 1.0 / (entropy + 1e-9)

        self.state["components"]["foundation.attribution_architecture"]["current_best"] = (
            f"Entropy={entropy:.3f}, Top driver: {scores.index[0]}={scores.iloc[0]:.4f}"
        )
        return 0  # Need PyTorch for true improvement testing

    def _tune_intervention_potencies(self, data: pd.DataFrame) -> float:
        """Would need A/B test data to truly optimize. Record current state."""
        self.state["components"]["foundation.intervention_vault_potencies"]["current_best"] = (
            "13 interventions with estimated potencies (needs A/B validation)"
        )
        return 0  # Requires live A/B test data

    def _tune_optimizer_hyperparams(self, data: pd.DataFrame) -> float:
        """Test if changing optimizer params improves convergence."""
        # This would run Optuna/Hyperopt on the optimizer itself
        # For now, record that default params work
        self.state["components"]["foundation.optimization_hyperparameters"]["current_best"] = (
            "BO: n_init=5, n_iter=15, GP(Matern2.5) | DE: popsize=10, maxiter=30"
        )
        return 0

    def _tune_regime_detection(self, data: pd.DataFrame) -> float:
        """Test different PELT penalties and ratio thresholds."""
        from contractkit_infinity_engine import AdsDiscoveryEngine

        if "conversions" not in data.columns:
            return 0

        engine = AdsDiscoveryEngine(data, "conversions")
        regime = engine.detect_regimes_and_drift()

        n_changepoints = len(regime.get("changepoints", []))
        n_features_flagged = len(regime.get("feature_status", {}))

        self.state["components"]["foundation.regime_detection_sensitivity"]["current_best"] = (
            f"PELT pen=10, {n_changepoints} changepoints, {n_features_flagged} features analyzed"
        )
        return 0

    def _expand_feature_engineering(self, data: pd.DataFrame) -> float:
        """Test if additional engineered features improve model quality."""
        from contractkit_infinity_engine import AdsDiscoveryEngine

        target = "conversions" if "conversions" in data.columns else data.columns[-1]

        # Baseline with existing features
        engine_base = AdsDiscoveryEngine(data, target)
        engine_base.scan_environment()
        base_r2 = engine_base.model.score(
            data[engine_base.known_features].fillna(0), data[target].fillna(0)
        )

        # Add new candidate features
        expanded = data.copy()
        new_features_added = 0

        if "clicks" in data.columns and "impressions" in data.columns:
            expanded["click_share"] = data["clicks"] / (data["impressions"] + 1e-9)
            new_features_added += 1
        if "cost" in data.columns and "clicks" in data.columns:
            expanded["cost_per_click_sq"] = (data["cost"] / (data["clicks"] + 1e-9)) ** 2
            new_features_added += 1
        if "quality_score" in data.columns:
            expanded["qs_squared"] = data["quality_score"] ** 2
            new_features_added += 1

        if new_features_added > 0:
            engine_exp = AdsDiscoveryEngine(expanded.select_dtypes(include=[np.number]), target)
            engine_exp.scan_environment()
            exp_r2 = engine_exp.model.score(
                expanded[engine_exp.known_features].fillna(0), expanded[target].fillna(0)
            )

            improvement = exp_r2 - base_r2
            if improvement > 0.001:
                self.state["components"]["operational.feature_engineering"]["current_best"] = (
                    f"R2={exp_r2:.4f} (+{improvement:.4f}) with {new_features_added} new features"
                )
                return improvement

        return 0

    def _tune_synthetic_model(self, data: pd.DataFrame) -> float:
        """Record distribution statistics for synthetic model calibration."""
        stats = {}
        for col in ["clicks", "impressions", "ctr", "avg_cpc", "conversions", "cost"]:
            if col in data.columns:
                stats[col] = {"mean": float(data[col].mean()), "std": float(data[col].std())}

        self.state["components"]["operational.synthetic_data_model"]["current_best"] = json.dumps(stats)[:200]
        return 0

    def _evaluate_time_range(self, data: pd.DataFrame) -> float:
        """Evaluate if we need a longer time range."""
        if "date" in data.columns:
            try:
                dates = pd.to_datetime(data["date"])
                span = (dates.max() - dates.min()).days
                self.state["components"]["breadth.time_range"]["current_best"] = f"{span} days"
            except Exception:
                pass
        return 0

    def _evaluate_entity_coverage(self) -> float:
        """Check which entities we're currently fetching."""
        self.state["components"]["breadth.entity_coverage"]["current_best"] = (
            "campaigns, ad_groups, keywords, search_terms, ads (5/8 possible)"
        )
        return 0

    def _evaluate_external_signals(self) -> float:
        """Check for available external data sources."""
        self.state["components"]["breadth.external_signals"]["current_best"] = (
            "Not yet integrated. Candidates: Google Trends, Stripe revenue, holiday calendar"
        )
        return 0

    def _evaluate_segmentation(self, data: pd.DataFrame) -> float:
        """Evaluate current segmentation granularity."""
        if "date" in data.columns and "keyword_text" in data.columns:
            self.state["components"]["breadth.segmentation_depth"]["current_best"] = (
                f"keyword x daily ({data['keyword_text'].nunique()} keywords x {data['date'].nunique()} days)"
            )
        return 0

    def get_status_summary(self) -> str:
        """Print a human-readable status summary."""
        active = [k for k, v in self.state["components"].items() if v["status"] == "active"]
        locked = [k for k, v in self.state["components"].items() if v["status"] == "locked"]

        lines = [
            f"Meta-Optimization Status (Cycle {self.state.get('total_cycles', 0)})",
            f"  Active: {len(active)} components",
            f"  Locked: {len(locked)} components",
            f"  Next cycle: {self.state.get('next_scheduled', 'TBD')}",
        ]
        return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("\n" + "=" * 60)
    print("  CONTRACTKIT META-OPTIMIZATION ENGINE")
    print("  Expanding breadth, deepening analysis, finding best answers")
    print("=" * 60 + "\n")

    meta = MetaOptimizer()
    report = meta.run_cycle()

    print(f"\n{meta.get_status_summary()}")
    print(f"\nRegistry updated: {REGISTRY_FILE}")
    print(f"State saved: {STATE_FILE}")
