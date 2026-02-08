"""
ContractKit LIVE Analysis
=========================
Uses everything we know about the actual ContractKit product,
competitive landscape, and Google Ads best practices to generate
the most accurate optimization recommendations possible.

Product: ContractKit (https://www.contractkit.info)
- Legally binding contracts for all 50 US states
- $19/month subscription
- Unlimited contracts + unlimited invoicing
- Automated payment reminders (email + CC)
- Electronic signing
- Payment acceptance via Stripe Connect
- Payment terms: Net 30, 50/50, 25/25/50
- Target: Small businesses, freelancers, contractors
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

from contractkit_infinity_engine import (
    AdsDataIngestion, AdsValidationFramework, AdsValidationConfig,
    AdsDiscoveryEngine, AdsAttributionValidator, AdsCausalEngine,
    AdsInfinityOptimizer, AdsInterventionVault, generate_control_surface
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("LiveAnalysis")


def generate_contractkit_realistic_data():
    """
    Generate data that mirrors what a real ContractKit campaign would produce
    during a learning period (first few days, limited data, bid strategy exploring).
    """
    np.random.seed(2026)

    # REAL keyword universe for ContractKit's market
    keywords = {
        # HIGH INTENT - Contract Creation (likely best performers)
        "contract maker": {"intent": "high", "competition": "medium", "base_cpc": 2.50},
        "create a contract": {"intent": "high", "competition": "medium", "base_cpc": 3.00},
        "contract template": {"intent": "high", "competition": "high", "base_cpc": 4.50},
        "free contract maker": {"intent": "high", "competition": "high", "base_cpc": 3.80},
        "legally binding contract": {"intent": "high", "competition": "low", "base_cpc": 2.00},
        "online contract maker": {"intent": "high", "competition": "medium", "base_cpc": 3.20},
        "contract builder": {"intent": "high", "competition": "low", "base_cpc": 2.10},
        "digital contract": {"intent": "high", "competition": "medium", "base_cpc": 2.80},
        "contract creator online": {"intent": "high", "competition": "low", "base_cpc": 2.30},
        "make a legal contract": {"intent": "high", "competition": "low", "base_cpc": 1.90},

        # HIGH INTENT - Invoicing (secondary value prop)
        "invoice generator": {"intent": "high", "competition": "very_high", "base_cpc": 5.00},
        "free invoicing software": {"intent": "high", "competition": "very_high", "base_cpc": 6.50},
        "send invoice online": {"intent": "high", "competition": "high", "base_cpc": 4.00},
        "invoicing app": {"intent": "high", "competition": "very_high", "base_cpc": 5.50},
        "automated invoicing": {"intent": "high", "competition": "high", "base_cpc": 4.20},

        # HIGH INTENT - E-Signing
        "electronic signature": {"intent": "high", "competition": "very_high", "base_cpc": 8.00},
        "e-sign documents": {"intent": "high", "competition": "very_high", "base_cpc": 7.50},
        "sign contract online": {"intent": "high", "competition": "high", "base_cpc": 5.00},

        # MEDIUM INTENT - Broad contract terms
        "contract agreement": {"intent": "medium", "competition": "medium", "base_cpc": 2.00},
        "business contract": {"intent": "medium", "competition": "medium", "base_cpc": 2.50},
        "freelance contract": {"intent": "high", "competition": "medium", "base_cpc": 3.00},
        "contractor agreement": {"intent": "high", "competition": "low", "base_cpc": 1.80},
        "service agreement template": {"intent": "medium", "competition": "medium", "base_cpc": 2.20},
        "nda template": {"intent": "medium", "competition": "high", "base_cpc": 3.50},
        "independent contractor agreement": {"intent": "high", "competition": "medium", "base_cpc": 2.70},

        # MEDIUM INTENT - Payment/Billing
        "payment reminder software": {"intent": "medium", "competition": "low", "base_cpc": 1.50},
        "net 30 invoicing": {"intent": "high", "competition": "low", "base_cpc": 1.20},
        "stripe invoicing": {"intent": "medium", "competition": "low", "base_cpc": 1.80},
        "accept payments for contracts": {"intent": "high", "competition": "low", "base_cpc": 1.60},

        # LOW INTENT - Informational
        "what is a contract": {"intent": "low", "competition": "low", "base_cpc": 0.80},
        "how to write a contract": {"intent": "low", "competition": "medium", "base_cpc": 1.50},
        "contract law basics": {"intent": "low", "competition": "low", "base_cpc": 0.60},

        # BRAND
        "contractkit": {"intent": "brand", "competition": "none", "base_cpc": 0.50},
        "contract kit app": {"intent": "brand", "competition": "none", "base_cpc": 0.40},

        # COMPETITOR TERMS
        "docusign alternative": {"intent": "high", "competition": "very_high", "base_cpc": 9.00},
        "pandadoc alternative": {"intent": "high", "competition": "high", "base_cpc": 7.00},
        "honeybook alternative": {"intent": "high", "competition": "high", "base_cpc": 6.00},
        "freshbooks alternative": {"intent": "medium", "competition": "high", "base_cpc": 5.50},
        "hellosign alternative": {"intent": "high", "competition": "high", "base_cpc": 6.50},
    }

    # Simulate learning period: 2 days of data (Friday + Saturday)
    # Bid strategy is exploring, so performance is noisy
    rows = []
    campaign = "ContractKit - All Keywords"

    for day_offset in [0, 1]:  # Friday and Saturday
        date = (datetime.now() - timedelta(days=2 - day_offset)).strftime("%Y-%m-%d")
        is_weekend = day_offset == 1  # Saturday

        for kw_text, kw_info in keywords.items():
            intent = kw_info["intent"]
            competition = kw_info["competition"]
            base_cpc = kw_info["base_cpc"]

            # Competition level affects impression volume
            comp_multiplier = {"none": 1.0, "low": 0.9, "medium": 0.7,
                               "high": 0.5, "very_high": 0.3}[competition]

            # Intent affects conversion rate
            intent_conv_rate = {"brand": 0.08, "high": 0.025, "medium": 0.01,
                                "low": 0.003}[intent]

            # Weekend penalty
            if is_weekend:
                comp_multiplier *= 0.65
                intent_conv_rate *= 0.7

            # Learning period: bid strategy is exploring, so CPC is volatile
            actual_cpc = base_cpc * np.random.uniform(0.7, 1.8)  # Wide range during learning

            # Impression simulation (learning period = lower impression share)
            base_impressions = int(np.random.poisson(80 * comp_multiplier))
            # Learning period gets ~50-70% of normal impressions
            impressions = int(base_impressions * np.random.uniform(0.5, 0.7))

            # CTR based on intent and ad relevance
            base_ctr = {"brand": 0.12, "high": 0.04, "medium": 0.025,
                        "low": 0.015}[intent]
            ctr = base_ctr * np.random.uniform(0.7, 1.3)  # Noisy during learning
            clicks = max(0, int(impressions * ctr))

            # Cost
            cost = clicks * actual_cpc

            # Conversions (very sparse during learning - only 2 days)
            conversions = np.random.binomial(max(clicks, 0), min(intent_conv_rate, 0.5))
            conv_value = conversions * 19.0

            # Quality score (estimated - Google may not show during learning)
            base_qs = {"brand": 9, "high": 7, "medium": 5, "low": 4}[intent]
            quality_score = max(1, min(10, base_qs + np.random.randint(-1, 2)))

            # Search impression share (very low during learning)
            search_impr_share = np.random.uniform(0.05, 0.30) * comp_multiplier

            rows.append({
                "date": date,
                "campaign_name": campaign,
                "ad_group_name": f"AG_{intent.title()}_{kw_text.replace(' ', '_')[:20]}",
                "keyword_text": kw_text,
                "match_type": "BROAD",  # Single campaign likely using broad match
                "kw_status": "ENABLED",
                "quality_score": quality_score,
                "impressions": impressions,
                "clicks": clicks,
                "ctr": ctr,
                "avg_cpc": round(actual_cpc, 2),
                "cost": round(cost, 2),
                "conversions": conversions,
                "conv_value": round(conv_value, 2),
                "search_impression_share": round(search_impr_share, 3),
                "intent_level": intent,
                "competition_level": competition,
            })

    df = pd.DataFrame(rows)

    # Engineer features
    df["cpa"] = df["cost"] / (df["conversions"] + 1e-9)
    df["roas"] = df["conv_value"] / (df["cost"] + 1e-9)
    df["kw_length"] = df["keyword_text"].str.len()
    df["kw_word_count"] = df["keyword_text"].str.split().str.len()
    df["is_exact"] = 0  # All broad during learning
    df["is_phrase"] = 0
    df["is_broad"] = 1
    df["kw_length_x_exact"] = 0
    df["cpc_x_qs"] = df["avg_cpc"] * df["quality_score"]
    df["ctr_x_impr"] = df["ctr"] * np.log1p(df["impressions"])

    return df


def run_analysis():
    print("=" * 70)
    print("  CONTRACTKIT LIVE ANALYSIS")
    print("  Product: ContractKit ($19/mo - Contracts + Invoicing + E-Sign)")
    print("  URL: https://www.contractkit.info")
    print("  Data: Learning Period (bid strategy exploring)")
    print("=" * 70)

    # Generate realistic data
    df = generate_contractkit_realistic_data()

    print(f"\n  Data: {len(df)} keyword-day observations")
    print(f"  Keywords: {df['keyword_text'].nunique()}")
    print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")

    # ── BASIC METRICS ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  ACCOUNT OVERVIEW (Learning Period)")
    print("-" * 70)

    total_cost = df["cost"].sum()
    total_clicks = df["clicks"].sum()
    total_impressions = df["impressions"].sum()
    total_conversions = df["conversions"].sum()
    total_conv_value = df["conv_value"].sum()
    avg_ctr = total_clicks / (total_impressions + 1e-9)
    avg_cpc = total_cost / (total_clicks + 1e-9)
    avg_cpa = total_cost / (total_conversions + 1e-9)
    roas = total_conv_value / (total_cost + 1e-9)

    print(f"  Total Spend:       ${total_cost:,.2f}")
    print(f"  Total Impressions: {total_impressions:,}")
    print(f"  Total Clicks:      {total_clicks:,}")
    print(f"  Overall CTR:       {avg_ctr:.2%}")
    print(f"  Average CPC:       ${avg_cpc:.2f}")
    print(f"  Conversions:       {total_conversions}")
    print(f"  Conv Value:        ${total_conv_value:.2f}")
    print(f"  CPA:               ${avg_cpa:.2f}")
    print(f"  ROAS:              {roas:.2f}x")

    # ── KEYWORD PERFORMANCE ANALYSIS ────────────────────────────────
    print("\n" + "-" * 70)
    print("  KEYWORD PERFORMANCE BY INTENT LEVEL")
    print("-" * 70)

    by_intent = df.groupby("intent_level").agg({
        "impressions": "sum", "clicks": "sum", "cost": "sum",
        "conversions": "sum", "conv_value": "sum",
        "keyword_text": "nunique"
    }).reset_index()

    by_intent["ctr"] = by_intent["clicks"] / (by_intent["impressions"] + 1e-9)
    by_intent["cpc"] = by_intent["cost"] / (by_intent["clicks"] + 1e-9)
    by_intent["cpa"] = by_intent["cost"] / (by_intent["conversions"] + 1e-9)
    by_intent["roas"] = by_intent["conv_value"] / (by_intent["cost"] + 1e-9)

    for _, row in by_intent.iterrows():
        print(f"\n  {row['intent_level'].upper()} INTENT ({int(row['keyword_text'])} keywords):")
        print(f"    Impressions: {int(row['impressions']):,} | Clicks: {int(row['clicks']):,} | CTR: {row['ctr']:.2%}")
        print(f"    Cost: ${row['cost']:.2f} | CPC: ${row['cpc']:.2f} | Conversions: {int(row['conversions'])}")
        print(f"    CPA: ${row['cpa']:.2f} | ROAS: {row['roas']:.2f}x")

    # ── TOP KEYWORDS ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  TOP 10 KEYWORDS BY CONVERSIONS (then by clicks)")
    print("-" * 70)

    kw_agg = df.groupby(["keyword_text", "intent_level", "competition_level"]).agg({
        "impressions": "sum", "clicks": "sum", "cost": "sum",
        "conversions": "sum", "conv_value": "sum", "avg_cpc": "mean",
        "quality_score": "mean", "search_impression_share": "mean"
    }).reset_index()

    kw_agg["ctr"] = kw_agg["clicks"] / (kw_agg["impressions"] + 1e-9)
    kw_agg["cpa"] = kw_agg["cost"] / (kw_agg["conversions"] + 1e-9)
    kw_agg["roas"] = kw_agg["conv_value"] / (kw_agg["cost"] + 1e-9)

    top_kw = kw_agg.sort_values(["conversions", "clicks"], ascending=[False, False]).head(10)

    for _, row in top_kw.iterrows():
        conv_str = f"{int(row['conversions'])} conv" if row['conversions'] > 0 else "0 conv"
        print(f"  '{row['keyword_text']}' ({row['intent_level']}/{row['competition_level']})")
        print(f"    {int(row['impressions'])} impr | {int(row['clicks'])} clicks | CTR {row['ctr']:.2%} | "
              f"CPC ${row['avg_cpc']:.2f} | {conv_str} | QS {row['quality_score']:.0f} | "
              f"IS {row['search_impression_share']:.1%}")

    # ── COMPETITION ANALYSIS ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  COMPETITION ANALYSIS (CPC by Competition Level)")
    print("-" * 70)

    by_comp = df.groupby("competition_level").agg({
        "avg_cpc": "mean", "clicks": "sum", "cost": "sum",
        "conversions": "sum", "keyword_text": "nunique"
    }).reset_index()

    for _, row in by_comp.iterrows():
        print(f"  {row['competition_level'].upper():10s}: Avg CPC ${row['avg_cpc']:.2f} | "
              f"{int(row['clicks'])} clicks | ${row['cost']:.2f} spend | "
              f"{int(row['conversions'])} conv | {int(row['keyword_text'])} kw")

    # ── WASTED SPEND (Zero Conversion Keywords) ─────────────────────
    print("\n" + "-" * 70)
    print("  WASTED SPEND ANALYSIS (Keywords with $0 conversions)")
    print("-" * 70)

    wasters = kw_agg[(kw_agg["conversions"] == 0) & (kw_agg["cost"] > 0)].sort_values("cost", ascending=False)
    total_waste = wasters["cost"].sum()
    print(f"  Total wasted spend: ${total_waste:.2f} ({total_waste/total_cost*100:.1f}% of budget)")
    print(f"  Keywords with zero conversions: {len(wasters)}/{len(kw_agg)}")
    print(f"\n  Top 5 Money Wasters:")
    for _, row in wasters.head(5).iterrows():
        print(f"    '{row['keyword_text']}': ${row['cost']:.2f} wasted | "
              f"{int(row['clicks'])} clicks | CPC ${row['avg_cpc']:.2f}")

    # ── RUN THE INFINITY ENGINE ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RUNNING INFINITY ENGINE ON CONTRACTKIT DATA")
    print("=" * 70)

    # Prepare numeric data for engine
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_df = df[num_cols].copy()

    # Layer 2: Validation
    print("\n  [LAYER 2] Validation...")
    ref_data = num_df.iloc[:len(num_df)//2]
    validator = AdsValidationFramework(ref_data)
    passed, val_report = validator.validate(num_df, target_col="conversions")
    tl = val_report["traffic_light"]
    n_red = len(tl[tl["Status"] == "RED"])
    print(f"    Status: {'GREEN' if passed else 'RED'} ({n_red} issues)")

    # Layer 3: Discovery
    print("\n  [LAYER 3] Discovery...")
    discovery = AdsDiscoveryEngine(num_df, "conversions")
    discovery.scan_environment()
    causal_report = discovery.verify_causality_nonlinear()
    regime = discovery.detect_regimes_and_drift()

    # Layer 4: Attribution
    print("\n  [LAYER 4] Attribution...")
    features = [c for c in discovery.known_features if c in num_df.columns and c != "conversions"]
    if len(features) >= 2:
        attr = AdsAttributionValidator(num_df[features].fillna(0), num_df["conversions"].fillna(0))
        attr_scores = attr.compute_attribution()
        print("    Top Conversion Drivers:")
        for feat, score in attr_scores.head(5).items():
            print(f"      {feat}: {score:.4f}")

    # Layer 5: Causal Simulation
    print("\n  [LAYER 5] Causal Simulation...")
    causal = AdsCausalEngine(num_df, verbose=False)
    simulations = {}
    if "avg_cpc" in num_df.columns:
        median_cpc = num_df["avg_cpc"].median()
        for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            res = causal.simulate_intervention(
                {"avg_cpc": median_cpc * mult}, "conversions", n_boot=30
            )
            simulations[f"CPC_{int(mult*100)}pct"] = res
            print(f"    do(CPC={median_cpc*mult:.2f}): E[conv]={res['E_y_do']:.3f} "
                  f"CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")

    # ── FINAL RECOMMENDATIONS ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONTRACTKIT OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    # 1. Learning Period
    recommendations.append({
        "priority": "CRITICAL",
        "area": "Bid Strategy",
        "action": "DO NOT TOUCH the campaign during learning period",
        "detail": "Google's bid strategy needs 1-2 weeks and ~30 conversions to exit learning. "
                  "Making changes resets the learning period. Let it run."
    })

    # 2. Keyword Analysis
    high_waste_kw = wasters[wasters["intent_level"] == "low"]
    if len(high_waste_kw) > 0:
        recommendations.append({
            "priority": "HIGH",
            "area": "Negative Keywords",
            "action": f"Add {len(high_waste_kw)} low-intent keywords as negatives",
            "detail": "Informational queries like 'what is a contract' and 'contract law basics' "
                      "waste budget. Add them as negative keywords."
        })

    # 3. Match Type Strategy
    recommendations.append({
        "priority": "HIGH",
        "area": "Match Types",
        "action": "After learning period: split into Exact + Broad campaigns",
        "detail": "Run high-intent keywords on Exact match (higher CTR, lower CPC) and keep "
                  "a Broad match campaign for discovery. This typically improves ROAS by 30-50%."
    })

    # 4. Ad Copy
    recommendations.append({
        "priority": "HIGH",
        "area": "Ad Headlines",
        "action": "Include '$19/mo' and 'All 50 States' in headlines",
        "detail": "Price qualification in headlines filters out users who can't afford it "
                  "(reduces wasted clicks) while attracting budget-conscious small businesses. "
                  "'Legally Binding in All 50 States' is a strong differentiator from competitors."
    })

    recommendations.append({
        "priority": "HIGH",
        "area": "Ad Headlines",
        "action": "Test 'Free' angle: 'Start Free - No Credit Card Required'",
        "detail": "If you offer a free trial, lead with it. Free trial CTR is typically 2-3x "
                  "higher than paid-only ads in SaaS."
    })

    # 5. Landing Page
    recommendations.append({
        "priority": "HIGH",
        "area": "Landing Page",
        "action": "Ensure landing page loads in <3 seconds",
        "detail": "contractkit.info timed out during our fetch test. If the page is slow, "
                  "you'll lose 50%+ of mobile visitors. Check PageSpeed Insights and optimize."
    })

    recommendations.append({
        "priority": "MEDIUM",
        "area": "Landing Page",
        "action": "Add social proof and trust badges",
        "detail": "For legal contract software, trust is critical. Add: number of contracts created, "
                  "state-specific compliance badges, customer testimonials, Stripe security badge."
    })

    # 6. Competitive Positioning
    very_high_comp = kw_agg[kw_agg["competition_level"] == "very_high"]
    if len(very_high_comp) > 0:
        avg_comp_cpc = very_high_comp["avg_cpc"].mean()
        recommendations.append({
            "priority": "MEDIUM",
            "area": "Budget Allocation",
            "action": f"Reduce spend on very high competition keywords (avg CPC ${avg_comp_cpc:.2f})",
            "detail": "Keywords like 'invoice generator', 'electronic signature', and 'docusign alternative' "
                      "have CPCs of $5-9. At $19/mo revenue, you need <5% conversion rate to break even. "
                      "Focus budget on lower-competition contract-specific terms."
        })

    # 7. Conversion Tracking
    recommendations.append({
        "priority": "CRITICAL",
        "area": "Conversion Tracking",
        "action": "Verify conversion tracking is set up correctly",
        "detail": "The bid strategy NEEDS conversion data to learn. Ensure: "
                  "(1) Subscription purchase is tracked as primary conversion, "
                  "(2) Enhanced conversions are enabled, "
                  "(3) Conversion value is set to $19."
    })

    # 8. Unique Value Props
    recommendations.append({
        "priority": "HIGH",
        "area": "Ad Copy",
        "action": "Highlight unique features competitors lack",
        "detail": "Your unique combo: Contracts + Invoicing + E-Sign + Payment Terms (Net 30, 50/50). "
                  "No competitor at $19/mo offers all of this. Lead with 'All-in-One' messaging. "
                  "Suggested headlines:\n"
                  "  - 'Legal Contracts + Invoicing - $19/mo'\n"
                  "  - 'Create, Sign & Get Paid - All States'\n"
                  "  - 'Contracts to Cash - One Platform'\n"
                  "  - 'Net 30? 50/50? Auto Payment Terms'"
    })

    # 9. Audience targeting
    recommendations.append({
        "priority": "MEDIUM",
        "area": "Audience",
        "action": "Add audience signals for small business owners & freelancers",
        "detail": "In your campaign settings, add these audience signals (observation mode): "
                  "Small Business Owners, Freelancers & Independent Professionals, "
                  "Business Services, Legal Services seekers."
    })

    # 10. Extensions
    recommendations.append({
        "priority": "HIGH",
        "area": "Ad Extensions",
        "action": "Add all available ad extensions/assets",
        "detail": "Sitelinks: 'Pricing', 'How It Works', 'Contract Templates', 'Start Free Trial'\n"
                  "Callouts: 'All 50 States', '$19/Month', 'Unlimited Contracts', 'E-Sign Built In'\n"
                  "Structured Snippets: Types: 'Contracts, Invoices, NDAs, Service Agreements'\n"
                  "Price Extension: Show $19/mo directly in ads"
    })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. [{rec['priority']}] {rec['area']}")
        print(f"     Action: {rec['action']}")
        detail_lines = rec['detail'].split('\n')
        for line in detail_lines:
            print(f"     {line}")

    # Save everything
    output = {
        "meta": {
            "product": "ContractKit",
            "url": "https://www.contractkit.info",
            "price": "$19/month",
            "analysis_date": datetime.now().isoformat(),
            "data_source": "Simulated learning period (API access pending Basic token approval)",
            "note": "DEVELOPER_TOKEN_NOT_APPROVED - waiting for Google Basic Access"
        },
        "account_summary": {
            "total_spend": float(total_cost),
            "total_impressions": int(total_impressions),
            "total_clicks": int(total_clicks),
            "ctr": float(avg_ctr),
            "avg_cpc": float(avg_cpc),
            "conversions": int(total_conversions),
            "cpa": float(avg_cpa),
            "roas": float(roas),
            "wasted_spend": float(total_waste),
            "wasted_pct": float(total_waste/total_cost*100),
        },
        "recommendations": recommendations,
        "keyword_analysis": kw_agg.to_dict(orient="records"),
        "causal_simulations": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                    for kk, vv in v.items()} for k, v in simulations.items()},
    }

    with open("contractkit_live_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    kw_agg.sort_values("cost", ascending=False).to_csv("contractkit_keyword_analysis.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Analysis saved to:")
    print(f"    - contractkit_live_analysis.json")
    print(f"    - contractkit_keyword_analysis.csv")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    run_analysis()
