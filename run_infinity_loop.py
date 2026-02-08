"""
================================================================================
 ContractKit Infinity Loop Runner
 ────────────────────────────────
 Quick-start script to run the full 7-layer optimization pipeline.
 
 Usage:
   python run_infinity_loop.py                    # Synthetic test data
   python run_infinity_loop.py --customer-id 1234567890  # Live data
================================================================================
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contractkit_infinity_engine import ContractKitInfinityEngine


def main():
    print("""
    +==================================================================+
    |           CONTRACTKIT INFINITY ADS ENGINE v1.0                   |
    |     Automated Google Ads Optimization for $19/mo Subscribers     |
    |                                                                  |
    |  Layers:                                                         |
    |    1. Ingestion    (Google Ads API / MCP)                        |
    |    2. Validation   (14-Scanner TVF Health Check)                 |
    |    3. Discovery    (Causal Dominance + Regime Detection)         |
    |    4. Attribution  (Integrated Gradients + Shapley Interactions) |
    |    5. Simulation   (Do-Calculus + Simpson's Paradox Resolution)  |
    |    6. Optimization (Bayesian + Evolutionary + Infinity Loop)     |
    |    7. Execution    (Intervention Vault + Control Surface)        |
    +==================================================================+
    """)

    # Check for customer ID in args or environment
    customer_id = None
    if len(sys.argv) > 1 and sys.argv[1] == "--customer-id" and len(sys.argv) > 2:
        customer_id = sys.argv[2]
    elif os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID"):
        customer_id = os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"]

    if customer_id:
        print(f"  Mode: LIVE DATA (Customer ID: {customer_id})")
    else:
        print("  Mode: SYNTHETIC DATA (no customer ID provided)")
        print("  To use live data: python run_infinity_loop.py --customer-id YOUR_ID")

    print()

    # Initialize and run
    engine = ContractKitInfinityEngine(customer_id=customer_id)
    result = engine.run_infinity_loop(target_col="conversions")

    # Summary
    n_actions = result.get("summary", {}).get("total_actions", 0)
    n_high = result.get("summary", {}).get("high_priority", 0)

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {n_actions} optimization actions generated")
    print(f"  High Priority: {n_high} actions need immediate attention")
    print(f"  Output Files:")
    print(f"    - contractkit_optimization_surface.json")
    print(f"    - contractkit_optimization_surface.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
