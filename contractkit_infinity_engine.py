"""
================================================================================
 CONTRACTKIT INFINITY ADS ENGINE (Production Release)
 ─────────────────────────────────────────────────────
 7-Layer Automated Google Ads Optimization System
 
 All logic derived from the 8 ContractKit Framework Notebooks:
   - full-spectrum-validation-framework.ipynb  (Layer 2: TVF)
   - unified-discovery-framework.ipynb         (Layer 3: UDE)
   - unified-attribution-framework.ipynb       (Layer 4: UAV)
   - unified-intervention-framework.ipynb      (Layer 5: PCE)
   - unified-optimization-framework.ipynb      (Layer 6: UOF)
   - gradient-boost-ai-opt.ipynb               (Layer 6: Infinity Loop)
   - health.ipynb                              (Layer 7: Therapeutic Vault)
   - op-system-opt.ipynb                       (Layer 7: Control Surface)
================================================================================
"""

import os
import json
import logging
import warnings
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare, norm
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.stats import qmc

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C_Kernel, Matern

import xgboost as xgb
import lightgbm as lgb
import networkx as nx

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = 'cpu'

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("InfinityEngine")


# ==============================================================================
# LAYER 1: INGESTION (Google Ads API Data Fetcher)
# ==============================================================================

# GAQL Queries for fetching all required Google Ads data
GAQL_QUERIES = {
    "campaigns": """
        SELECT
            campaign.id, campaign.name, campaign.status,
            campaign.bidding_strategy_type,
            campaign.campaign_budget,
            metrics.clicks, metrics.impressions, metrics.ctr,
            metrics.average_cpc, metrics.conversions,
            metrics.conversions_value, metrics.cost_micros,
            metrics.all_conversions, metrics.average_cost,
            segments.date
        FROM campaign
        WHERE segments.date DURING LAST_30_DAYS
        ORDER BY metrics.cost_micros DESC
    """,
    "ad_groups": """
        SELECT
            campaign.id, campaign.name,
            ad_group.id, ad_group.name, ad_group.status,
            ad_group.cpc_bid_micros,
            metrics.clicks, metrics.impressions, metrics.ctr,
            metrics.average_cpc, metrics.conversions,
            metrics.conversions_value, metrics.cost_micros,
            segments.date
        FROM ad_group
        WHERE segments.date DURING LAST_30_DAYS
        ORDER BY metrics.cost_micros DESC
    """,
    "keywords": """
        SELECT
            campaign.name, ad_group.name,
            ad_group_criterion.keyword.text,
            ad_group_criterion.keyword.match_type,
            ad_group_criterion.status,
            ad_group_criterion.quality_info.quality_score,
            metrics.clicks, metrics.impressions, metrics.ctr,
            metrics.average_cpc, metrics.conversions,
            metrics.conversions_value, metrics.cost_micros,
            metrics.search_impression_share,
            segments.date
        FROM keyword_view
        WHERE segments.date DURING LAST_30_DAYS
        ORDER BY metrics.cost_micros DESC
    """,
    "search_terms": """
        SELECT
            campaign.name, ad_group.name,
            search_term_view.search_term,
            metrics.clicks, metrics.impressions, metrics.ctr,
            metrics.conversions, metrics.conversions_value,
            metrics.cost_micros,
            segments.date
        FROM search_term_view
        WHERE segments.date DURING LAST_30_DAYS
        ORDER BY metrics.impressions DESC
    """,
    "ads": """
        SELECT
            campaign.name, ad_group.name,
            ad_group_ad.ad.id, ad_group_ad.ad.type,
            ad_group_ad.ad.final_urls,
            ad_group_ad.status,
            metrics.clicks, metrics.impressions, metrics.ctr,
            metrics.average_cpc, metrics.conversions,
            metrics.conversions_value, metrics.cost_micros,
            segments.date
        FROM ad_group_ad
        WHERE segments.date DURING LAST_30_DAYS
        ORDER BY metrics.impressions DESC
    """,
}


class AdsDataIngestion:
    """
    LAYER 1: Fetch live Google Ads data via google-ads Python client library.
    Falls back to synthetic data for testing when API is unavailable.
    """

    def __init__(self, customer_id: str = None, use_mcp: bool = False):
        self.customer_id = customer_id
        self.use_mcp = use_mcp
        self.raw_data = {}
        self.unified_df = None
        self.client = None
        logger.info("Layer 1: AdsDataIngestion initialized")

    def connect(self):
        """Attempt to connect to Google Ads API using ADC credentials."""
        try:
            from google.ads.googleads.client import GoogleAdsClient

            # Try loading from environment / ADC
            developer_token = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "")
            login_customer_id = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")

            if developer_token:
                config = {
                    "developer_token": developer_token,
                    "use_proto_plus": True,
                }
                if login_customer_id:
                    config["login_customer_id"] = login_customer_id

                self.client = GoogleAdsClient.load_from_dict(config)
                logger.info("Connected to Google Ads API via ADC")
                return True
            else:
                logger.warning("No developer token found. Will use synthetic data.")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to Google Ads API: {e}")
            return False

    def fetch_all(self, customer_id: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data entities from Google Ads.
        Returns dict of DataFrames keyed by entity name.
        """
        cid = customer_id or self.customer_id
        if not cid:
            logger.warning("No customer ID provided. Using synthetic data.")
            return self._generate_synthetic_data()

        if self.client is None:
            if not self.connect():
                return self._generate_synthetic_data()

        try:
            ga_service = self.client.get_service("GoogleAdsService")
            for name, query in GAQL_QUERIES.items():
                logger.info(f"Fetching {name}...")
                try:
                    stream = ga_service.search_stream(
                        customer_id=cid.replace("-", ""),
                        query=query.strip()
                    )
                    rows = []
                    for batch in stream:
                        for row in batch.results:
                            rows.append(self._row_to_dict(row, name))
                    self.raw_data[name] = pd.DataFrame(rows) if rows else pd.DataFrame()
                    logger.info(f"  {name}: {len(rows)} rows fetched")
                except Exception as e:
                    logger.warning(f"  {name}: fetch failed ({e})")
                    self.raw_data[name] = pd.DataFrame()

            self.unified_df = self._build_unified_dataframe()
            return self.raw_data

        except Exception as e:
            logger.error(f"API fetch failed: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_data()

    def _row_to_dict(self, row, entity_type: str) -> dict:
        """Convert a Google Ads API row to a flat dictionary."""
        d = {}
        try:
            # Metrics (common across all entities)
            m = row.metrics
            d["clicks"] = m.clicks
            d["impressions"] = m.impressions
            d["ctr"] = m.ctr
            d["avg_cpc"] = m.average_cpc / 1e6 if hasattr(m, 'average_cpc') else 0
            d["conversions"] = m.conversions
            d["conv_value"] = m.conversions_value
            d["cost"] = m.cost_micros / 1e6
            d["date"] = str(row.segments.date) if hasattr(row.segments, 'date') else ""

            # Entity-specific fields
            if entity_type == "campaigns":
                d["campaign_id"] = row.campaign.id
                d["campaign_name"] = row.campaign.name
                d["campaign_status"] = row.campaign.status.name
                d["bidding_strategy"] = row.campaign.bidding_strategy_type.name

            elif entity_type == "ad_groups":
                d["campaign_name"] = row.campaign.name
                d["ad_group_id"] = row.ad_group.id
                d["ad_group_name"] = row.ad_group.name
                d["ad_group_status"] = row.ad_group.status.name
                d["cpc_bid"] = row.ad_group.cpc_bid_micros / 1e6 if row.ad_group.cpc_bid_micros else 0

            elif entity_type == "keywords":
                d["campaign_name"] = row.campaign.name
                d["ad_group_name"] = row.ad_group.name
                d["keyword_text"] = row.ad_group_criterion.keyword.text
                d["match_type"] = row.ad_group_criterion.keyword.match_type.name
                d["kw_status"] = row.ad_group_criterion.status.name
                qs = row.ad_group_criterion.quality_info
                d["quality_score"] = qs.quality_score if hasattr(qs, 'quality_score') else 0

            elif entity_type == "search_terms":
                d["campaign_name"] = row.campaign.name
                d["ad_group_name"] = row.ad_group.name
                d["search_term"] = row.search_term_view.search_term

            elif entity_type == "ads":
                d["campaign_name"] = row.campaign.name
                d["ad_group_name"] = row.ad_group.name
                d["ad_id"] = row.ad_group_ad.ad.id
                d["ad_type"] = row.ad_group_ad.ad.type_.name
                d["ad_status"] = row.ad_group_ad.status.name
                urls = row.ad_group_ad.ad.final_urls
                d["final_url"] = urls[0] if urls else ""

        except Exception as e:
            logger.debug(f"Row parse error: {e}")

        return d

    def _build_unified_dataframe(self) -> pd.DataFrame:
        """
        Build a unified analysis DataFrame from all fetched entities.
        Aggregates to keyword-level with all metrics.
        """
        kw_df = self.raw_data.get("keywords", pd.DataFrame())
        if kw_df.empty:
            # Fall back to campaign-level if no keywords
            return self.raw_data.get("campaigns", pd.DataFrame())

        # Engineer features for the analytics layers
        df = kw_df.copy()
        if "cost" in df.columns and "conversions" in df.columns:
            df["cpa"] = df["cost"] / (df["conversions"] + 1e-9)
            df["roas"] = df["conv_value"] / (df["cost"] + 1e-9)
        if "keyword_text" in df.columns:
            df["kw_length"] = df["keyword_text"].str.len()
            df["kw_word_count"] = df["keyword_text"].str.split().str.len()
        if "match_type" in df.columns:
            df["is_exact"] = (df["match_type"] == "EXACT").astype(int)
            df["is_phrase"] = (df["match_type"] == "PHRASE").astype(int)
            df["is_broad"] = (df["match_type"] == "BROAD").astype(int)
        # Interaction features (from gradient-boost spatial/semantic engineering)
        if "kw_length" in df.columns and "is_exact" in df.columns:
            df["kw_length_x_exact"] = df["kw_length"] * df["is_exact"]
        if "avg_cpc" in df.columns and "quality_score" in df.columns:
            df["cpc_x_qs"] = df["avg_cpc"] * (df["quality_score"] + 1e-9)
        if "ctr" in df.columns and "impressions" in df.columns:
            df["ctr_x_impr"] = df["ctr"] * np.log1p(df["impressions"])

        self.unified_df = df
        return df

    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic synthetic Google Ads data for testing.
        Models ContractKit's $19/mo subscription product.
        """
        logger.info("Generating synthetic ContractKit Ads data...")
        np.random.seed(2025)
        n_days = 30
        n_keywords = 50

        # ContractKit-relevant keywords
        kw_pool = [
            "invoice template", "free invoice generator", "contract maker",
            "billing software", "invoice app", "contract template",
            "freelance invoice", "small business invoicing", "online invoicing",
            "receipt maker", "estimate template", "proposal template",
            "invoice software free", "create invoice online", "send invoice",
            "professional invoice", "contractor invoice", "invoice pdf",
            "business invoice", "auto invoicing", "recurring invoice",
            "subscription billing", "invoicing tool", "invoice tracker",
            "contractkit", "contractkit invoice", "contract management",
            "digital invoice", "mobile invoicing", "cloud invoicing",
            "invoice generator free", "free billing software", "best invoicing app",
            "simple invoice maker", "invoice template word", "invoice template pdf",
            "contract agreement", "freelancer tools", "self employed invoicing",
            "gig economy invoice", "contractor billing", "service invoice",
            "payment request", "accounts receivable", "invoice reminder",
            "overdue invoice", "invoice automation", "smart invoicing",
            "ai invoice", "automated billing"
        ][:n_keywords]

        match_types = ["EXACT", "PHRASE", "BROAD"]
        campaigns = ["ContractKit - Brand", "ContractKit - NonBrand", "ContractKit - Competitor"]

        rows = []
        for day in range(n_days):
            date = (datetime.now() - timedelta(days=n_days - day)).strftime("%Y-%m-%d")
            for i, kw in enumerate(kw_pool):
                # Simulate keyword-level performance
                campaign = campaigns[i % 3]
                match = match_types[i % 3]
                quality_score = np.random.choice([3, 4, 5, 6, 7, 8, 9, 10],
                                                  p=[0.02, 0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.08])

                # Brand keywords perform better
                is_brand = "contractkit" in kw.lower()
                base_impr = np.random.poisson(500 if is_brand else 150)
                base_ctr = np.random.beta(5, 20) if not is_brand else np.random.beta(15, 10)
                clicks = int(base_impr * base_ctr)
                avg_cpc = np.random.gamma(2, 0.5) if not is_brand else np.random.gamma(1.5, 0.3)
                cost = clicks * avg_cpc

                # Conversion modeling: $19/mo subscription
                # Higher quality score + exact match -> higher conv rate
                base_conv_rate = 0.02 + 0.005 * quality_score
                if match == "EXACT":
                    base_conv_rate *= 1.3
                if is_brand:
                    base_conv_rate *= 2.0

                conversions = np.random.binomial(max(clicks, 0), min(base_conv_rate, 0.5))
                conv_value = conversions * 19.0  # $19/mo subscription

                # Day-of-week effect (weekdays better)
                dow = (datetime.now() - timedelta(days=n_days - day)).weekday()
                if dow >= 5:  # weekend
                    clicks = int(clicks * 0.7)
                    conversions = int(conversions * 0.6)

                # Competitor activity simulation (regime change at day 15)
                if day >= 15 and "invoice" in kw:
                    avg_cpc *= 1.15  # competitor bid pressure
                    base_impr = int(base_impr * 0.85)

                rows.append({
                    "date": date,
                    "campaign_name": campaign,
                    "ad_group_name": f"AG_{kw.replace(' ', '_')[:20]}",
                    "keyword_text": kw,
                    "match_type": match,
                    "kw_status": "ENABLED",
                    "quality_score": quality_score,
                    "impressions": max(base_impr, 0),
                    "clicks": max(clicks, 0),
                    "ctr": base_ctr,
                    "avg_cpc": round(avg_cpc, 2),
                    "cost": round(cost, 2),
                    "conversions": max(conversions, 0),
                    "conv_value": round(conv_value, 2),
                })

        kw_df = pd.DataFrame(rows)

        # Engineer features
        kw_df["cpa"] = kw_df["cost"] / (kw_df["conversions"] + 1e-9)
        kw_df["roas"] = kw_df["conv_value"] / (kw_df["cost"] + 1e-9)
        kw_df["kw_length"] = kw_df["keyword_text"].str.len()
        kw_df["kw_word_count"] = kw_df["keyword_text"].str.split().str.len()
        kw_df["is_exact"] = (kw_df["match_type"] == "EXACT").astype(int)
        kw_df["is_phrase"] = (kw_df["match_type"] == "PHRASE").astype(int)
        kw_df["is_broad"] = (kw_df["match_type"] == "BROAD").astype(int)
        kw_df["kw_length_x_exact"] = kw_df["kw_length"] * kw_df["is_exact"]
        kw_df["cpc_x_qs"] = kw_df["avg_cpc"] * (kw_df["quality_score"] + 1e-9)
        kw_df["ctr_x_impr"] = kw_df["ctr"] * np.log1p(kw_df["impressions"])

        self.raw_data = {"keywords": kw_df}
        self.unified_df = kw_df
        return self.raw_data


# ==============================================================================
# LAYER 2: FULL SPECTRUM VALIDATION (from full-spectrum-validation-framework.ipynb)
# ==============================================================================

@dataclass
class AdsValidationConfig:
    """Config adapted from TVFConfig for Google Ads metrics."""
    # Statistical
    drift_alpha: float = 0.05
    reconstruction_error_threshold: float = 0.05
    min_explained_variance_ratio: float = 0.90
    # Integrity & Ops
    max_null_spike: float = 0.10
    max_volumetric_drift: float = 0.50
    max_duplicate_rate: float = 0.0
    # Forensics
    enable_benfords_law: bool = True
    # Causal & Ethics
    max_correlation_drift: float = 0.25
    simpsons_threshold: float = 0.3
    max_fairness_disparity: float = 0.25
    # Compute
    max_cardinality: int = 100
    allow_zero_variance: bool = False
    max_leakage_r2: float = 0.98
    # Temporal
    max_recency_days: int = 30
    # Unsupervised
    enable_iforest: bool = True
    iforest_contamination: float = 0.02
    # Ads-specific thresholds
    min_ctr: float = 0.005          # Below 0.5% CTR is suspicious
    max_cpa: float = 100.0          # Above $100 CPA is a red flag for $19 product
    min_quality_score: int = 3      # Below 3 QS needs attention


class AdsValidationFramework:
    """
    LAYER 2: Full Spectrum Validation for Google Ads data.
    Implements all 14 scanners from TitanValidationFramework,
    adapted for ads-specific drift, anomaly, and health detection.
    """

    def __init__(self, reference_data: pd.DataFrame, config: AdsValidationConfig = None):
        self.config = config or AdsValidationConfig()
        self.reference = reference_data.copy()
        self.num_cols = self.reference.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.reference.select_dtypes(include=['object', 'category']).columns.tolist()
        self.ref_corr = self.reference[self.num_cols].corr() if self.num_cols else pd.DataFrame()
        self.ref_nulls = self.reference.isna().mean()
        self.ref_count = len(self.reference)
        self.ref_cat_freqs = {c: self.reference[c].value_counts(normalize=True) for c in self.cat_cols}

        # Train models from TVF
        self._train_unsupervised()
        self._train_structure()
        logger.info(f"Layer 2: AdsValidationFramework online. Baseline: {self.ref_count} rows, "
                     f"{len(self.num_cols)} numeric, {len(self.cat_cols)} categorical cols.")

    def _train_unsupervised(self):
        self.iforest = None
        if self.config.enable_iforest and len(self.reference) > 50 and self.num_cols:
            self.iforest = IsolationForest(
                contamination=self.config.iforest_contamination, random_state=42, n_jobs=-1
            )
            self.iforest.fit(self.reference[self.num_cols].fillna(0))

    def _train_structure(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.ref_recon_error = 0.0
        self.ref_explained_var = 1.0

        if len(self.num_cols) < 2:
            return

        X = self.reference[self.num_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        try:
            n_comp = min(0.95, X_scaled.shape[1] - 1) if X_scaled.shape[1] > 1 else 1
            self.pca = PCA(n_components=n_comp)
            self.pca.fit(X_scaled)
            X_recon = self.pca.inverse_transform(self.pca.transform(X_scaled))
            self.ref_recon_error = np.mean(np.square(X_scaled - X_recon))
            var_proj = np.var(self.pca.transform(X_scaled), axis=0).sum()
            var_orig = np.var(X_scaled, axis=0).sum()
            self.ref_explained_var = var_proj / (var_orig + 1e-9)
        except Exception:
            self.pca = None

    def validate(self, new_data: pd.DataFrame, target_col: str = None,
                 date_col: str = "date", subgroups: List[str] = None) -> Tuple[bool, Dict]:
        """Run all 14 validation scanners + ads-specific checks."""
        report = {"modules": {}, "traffic_light": None}

        report["modules"]["integrity"] = self._scan_integrity(new_data)
        report["modules"]["ops"] = self._scan_ops(new_data)
        report["modules"]["drift_num"] = self._scan_drift_numeric(new_data)
        report["modules"]["drift_cat"] = self._scan_drift_categorical(new_data)
        report["modules"]["forensics"] = self._scan_forensics(new_data)
        report["modules"]["structure"] = self._scan_structure(new_data)
        report["modules"]["stability"] = self._scan_stability(new_data)
        report["modules"]["paradox"] = self._scan_simpsons(new_data, target_col, subgroups)
        report["modules"]["fairness"] = self._scan_fairness(new_data, target_col, subgroups)
        report["modules"]["temporal"] = self._scan_temporal(new_data, date_col)
        report["modules"]["compute"] = self._scan_compute(new_data)
        report["modules"]["leakage"] = self._scan_leakage(new_data, target_col)
        report["modules"]["anomalies"] = self._scan_anomalies(new_data)
        # Ads-specific scanner
        report["modules"]["ads_health"] = self._scan_ads_health(new_data)

        report["traffic_light"] = self._generate_report(report)
        failed = len(report["traffic_light"][report["traffic_light"]["Status"] == "RED"]) > 0
        return (not failed), report

    def _scan_integrity(self, df):
        issues = []
        missing = set(self.reference.columns) - set(df.columns)
        if missing:
            issues.append(f"Missing Columns: {list(missing)[:3]}...")
        for c in df.columns:
            new_null = df[c].isna().mean()
            ref_null = self.ref_nulls.get(c, 0)
            if (new_null - ref_null) > self.config.max_null_spike:
                issues.append(f"{c}: Null Spike ({new_null:.1%})")
        return {"issues": issues}

    def _scan_ops(self, df):
        issues = []
        ratio = len(df) / (self.ref_count + 1e-9)
        if abs(1 - ratio) > self.config.max_volumetric_drift:
            issues.append(f"Volumetric Drift (Size Ratio: {ratio:.2f}x)")
        dupe_rate = df.duplicated().mean()
        if dupe_rate > self.config.max_duplicate_rate:
            issues.append(f"Duplicate Rows ({dupe_rate:.1%})")
        return {"issues": issues}

    def _scan_drift_numeric(self, df):
        drifted = []
        for c in self.num_cols:
            if c not in df.columns:
                continue
            try:
                stat, pval = ks_2samp(self.reference[c].dropna(), df[c].dropna())
                if pval < self.config.drift_alpha:
                    drifted.append(c)
            except Exception:
                pass
        return {"drifted": drifted}

    def _scan_drift_categorical(self, df):
        drifted = []
        for c in self.cat_cols:
            if c not in df.columns:
                continue
            ref_freq = self.ref_cat_freqs.get(c)
            if ref_freq is None:
                continue
            new_counts = df[c].value_counts()
            common_cats = ref_freq.index.intersection(new_counts.index)
            if len(common_cats) < 2:
                continue
            obs = new_counts[common_cats].sort_index().values
            exp = ref_freq[common_cats].sort_index().values * len(df)
            exp = exp * (obs.sum() / (exp.sum() + 1e-9))
            try:
                if chisquare(f_obs=obs, f_exp=exp)[1] < self.config.drift_alpha:
                    drifted.append(c)
            except Exception:
                pass
        return {"drifted": drifted}

    def _scan_forensics(self, df):
        if not self.config.enable_benfords_law:
            return {"suspicious": []}
        suspicious = []
        benford_probs = np.log10(1 + 1 / np.arange(1, 10))
        for c in self.num_cols:
            if c not in df.columns:
                continue
            try:
                digits = df[c].astype(str).str.lstrip('-').str[0]
                digits = digits[digits.isin([str(i) for i in range(1, 10)])].astype(int)
                if len(digits) < 100:
                    continue
                counts = digits.value_counts().sort_index()
                obs = np.array([counts.get(i, 0) for i in range(1, 10)])
                if chisquare(obs, f_exp=benford_probs * len(digits))[1] < 0.001:
                    suspicious.append(f"{c} (Benford Violation)")
            except Exception:
                pass
        return {"suspicious": suspicious}

    def _scan_structure(self, df):
        if self.pca is None or not self.num_cols:
            return {"drifted": False}
        try:
            X = self.scaler.transform(df[self.num_cols].fillna(0))
            X_recon = self.pca.inverse_transform(self.pca.transform(X))
            ratio = np.mean(np.square(X - X_recon)) / max(self.ref_recon_error, 1e-6)
            if ratio > (1 + self.config.reconstruction_error_threshold):
                return {"drifted": True, "msg": f"Structure Shift (Error Ratio: {ratio:.2f}x)"}
            return {"drifted": False}
        except Exception as e:
            return {"drifted": False, "msg": str(e)}

    def _scan_stability(self, df):
        common = [c for c in self.num_cols if c in df.columns]
        if len(common) < 2:
            return {"broken": []}
        diff = (self.reference[common].corr() - df[common].corr()).abs()
        broken = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if diff.iloc[i, j] > self.config.max_correlation_drift:
                    broken.append(f"{common[i]}-{common[j]}")
        return {"broken": broken}

    def _scan_simpsons(self, df, target, groups):
        reversals = []
        if not target or not groups:
            return {"reversals": []}
        feats = [f for f in self.num_cols if f != target and f in df.columns]
        for g in (groups or []):
            if g not in df.columns:
                continue
            for f in feats:
                try:
                    g_corr = df[f].corr(df[target])
                    if abs(g_corr) < self.config.simpsons_threshold:
                        continue
                    for sub in df[g].unique():
                        sub_df = df[df[g] == sub]
                        if len(sub_df) < 30:
                            continue
                        s_corr = sub_df[f].corr(sub_df[target])
                        if (np.sign(g_corr) != np.sign(s_corr)) and abs(s_corr) > self.config.simpsons_threshold:
                            reversals.append(f"{f} (Global {g_corr:.2f}, {sub} {s_corr:.2f})")
                            break
                except Exception:
                    pass
        return {"reversals": reversals}

    def _scan_fairness(self, df, target, groups):
        if not target or not groups or target not in df.columns:
            return {"issues": []}
        issues = []
        for g in (groups or []):
            if g not in df.columns:
                continue
            means = df.groupby(g)[target].mean()
            if len(means) < 2:
                continue
            disparity = (means.max() - means.min()) / (abs(means.min()) + 1e-9)
            if disparity > self.config.max_fairness_disparity:
                issues.append(f"Fairness Warning on '{g}' (Disparity: {disparity:.1%})")
        return {"issues": issues}

    def _scan_temporal(self, df, date_col):
        issues = []
        if date_col and date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col])
                if (pd.Timestamp.now() - dates.max()).days > self.config.max_recency_days:
                    issues.append("Stale Data")
            except Exception:
                issues.append("Date Parse Failed")
        return {"issues": issues}

    def _scan_compute(self, df):
        issues = []
        for c in df.select_dtypes(include=['object', 'category']):
            if df[c].nunique() > self.config.max_cardinality:
                issues.append(f"{c}: High Cardinality")
        if not self.config.allow_zero_variance:
            for c in self.num_cols:
                if c in df.columns and df[c].var() == 0:
                    issues.append(f"{c}: Zero Variance")
        return {"issues": issues}

    def _scan_leakage(self, df, target):
        leaks = []
        if target and target in df.columns:
            y = df[target].fillna(0)
            for c in self.num_cols:
                if c == target or c not in df.columns:
                    continue
                try:
                    r2 = LinearRegression().fit(
                        df[[c]].fillna(0), y
                    ).score(df[[c]].fillna(0), y)
                    if r2 > self.config.max_leakage_r2:
                        leaks.append(c)
                except Exception:
                    pass
        return {"leaks": leaks}

    def _scan_anomalies(self, df):
        if not self.iforest or not self.num_cols:
            return {"rate": 0.0, "status": "SKIPPED"}
        try:
            rate = (self.iforest.predict(df[self.num_cols].fillna(0)) == -1).mean()
            status = "RED" if rate > self.config.iforest_contamination * 4 else "GREEN"
            return {"rate": rate, "status": status}
        except Exception:
            return {"rate": 0.0, "status": "SKIPPED"}

    def _scan_ads_health(self, df):
        """Ads-specific health checks beyond the base TVF."""
        issues = []
        if "ctr" in df.columns:
            low_ctr = (df["ctr"] < self.config.min_ctr).mean()
            if low_ctr > 0.3:
                issues.append(f"30%+ keywords below {self.config.min_ctr:.1%} CTR (Click Decay)")
        if "cpa" in df.columns:
            high_cpa = (df["cpa"] > self.config.max_cpa).mean()
            if high_cpa > 0.2:
                issues.append(f"20%+ keywords above ${self.config.max_cpa} CPA (Cost Explosion)")
        if "quality_score" in df.columns:
            low_qs = (df["quality_score"] < self.config.min_quality_score).mean()
            if low_qs > 0.2:
                issues.append(f"20%+ keywords below QS {self.config.min_quality_score} (Quality Collapse)")
        if "conversions" in df.columns:
            zero_conv = (df["conversions"] == 0).mean()
            if zero_conv > 0.7:
                issues.append(f"70%+ keywords with 0 conversions (Zombie Alert)")
        return {"issues": issues}

    def _generate_report(self, report):
        rows = []
        map_ = {
            "integrity": "Integrity", "ops": "Ops Health", "drift_num": "Numeric Drift",
            "drift_cat": "Categorical Drift", "structure": "Multivariate Drift",
            "stability": "Causal Stability", "paradox": "Simpson's Paradox",
            "fairness": "Algorithmic Fairness", "temporal": "Temporal",
            "compute": "Compute", "leakage": "Data Leakage",
            "anomalies": "Multivariate Anomaly", "forensics": "Forensics",
            "ads_health": "Ads Health"
        }
        for mod, label in map_.items():
            data = report["modules"].get(mod, {})
            errs = []
            for k in ["issues", "drifted", "broken", "violations", "reversals", "leaks", "suspicious"]:
                v = data.get(k)
                if isinstance(v, list):
                    errs.extend(v)
            if mod == "anomalies" and data.get("status") == "RED":
                errs.append(f"Anomaly Rate {data['rate']:.1%}")
            if mod == "structure" and data.get("drifted"):
                errs.append(data.get("msg", "Structure Drift"))
            for e in errs:
                feat = str(e).split(":")[0] if ":" in str(e) else "Global"
                rows.append({"Module": label, "Feature": feat, "Status": "RED", "Reason": str(e)})

        if rows:
            return pd.DataFrame(rows).sort_values("Module").reset_index(drop=True)
        return pd.DataFrame([{"Module": "Global", "Feature": "All", "Status": "GREEN",
                              "Reason": "All 14 Checks Passed"}])

    def manifold_penalty(self, candidate_df: pd.DataFrame) -> float:
        """Penalty for unrealistic configs (from op-system-opt.ipynb)."""
        if self.pca is None or not self.num_cols:
            return 0.0
        try:
            df = candidate_df.copy()
            for c in self.num_cols:
                if c not in df.columns:
                    df[c] = 0.0
            X = self.scaler.transform(df[self.num_cols].fillna(0))
            X_recon = self.pca.inverse_transform(self.pca.transform(X))
            recon_ratio = np.mean((X - X_recon) ** 2) / max(self.ref_recon_error, 1e-9)
            return float(max(0.0, recon_ratio - 1.2))
        except Exception:
            return 0.0


# ==============================================================================
# LAYER 3: UNIFIED DISCOVERY ENGINE (from unified-discovery-framework.ipynb)
# ==============================================================================

class AdsDiscoveryEngine:
    """
    LAYER 3: Causal Discovery for Google Ads features.
    Implements: scan_environment, verify_causality_nonlinear,
    detect_regimes_and_drift, scan_for_instruments.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str = "date"):
        self.raw_df = df.copy()
        self.target = target_col
        self.date_col = date_col
        self.known_features = [c for c in df.select_dtypes(include=[np.number]).columns
                                if c not in [target_col]]
        self.regime_report = {}
        self.causal_report = {}
        logger.info(f"Layer 3: AdsDiscoveryEngine online. Target: '{self.target}', "
                     f"Features: {len(self.known_features)}")

    def scan_environment(self) -> Optional[str]:
        """Phase 1: Baseline model + temporal lag detection."""
        logger.info("[Discovery Phase 1] Scanning Environment...")
        X = self.raw_df[self.known_features].fillna(0)
        y = self.raw_df[self.target].fillna(0)

        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        baseline_r2 = self.model.score(X, y)
        logger.info(f"  Baseline Model R2: {baseline_r2:.4f}")

        # Residual temporal scan
        preds = self.model.predict(X)
        residuals = y.values - preds

        max_corr = 0
        best_lag = 0
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(residuals) <= lag:
                continue
            res_shift = np.roll(residuals, lag)
            from scipy.stats import pearsonr
            corr, _ = pearsonr(residuals[lag:], res_shift[lag:])
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                best_lag = lag

        if max_corr > 0.15:
            logger.info(f"  TEMPORAL GAP: Lag-{best_lag} detected (Corr: {max_corr:.2f})")
            return f"{self.target}_Lag_{best_lag}"

        logger.info("  No significant temporal gaps detected.")
        return None

    def verify_causality_nonlinear(self) -> Dict:
        """
        Phase 2: Non-Linear Double ML Proxy Kill.
        Separates true causal drivers from spurious proxies.
        """
        logger.info("[Discovery Phase 2] Causal Dominance Check...")

        # Anchors = features we trust (lag features, CPC, QS)
        anchors = [f for f in self.known_features
                    if any(kw in f.lower() for kw in ["lag", "cpc", "cost", "quality", "impression"])]
        suspects = [f for f in self.known_features if f not in anchors]

        if not anchors:
            logger.info("  No anchors found. Skipping proxy kill.")
            return {"survivors": self.known_features, "killed": []}

        survivors = list(anchors)
        killed = []

        for suspect in suspects:
            X_anchor = self.raw_df[anchors].fillna(0)
            y_suspect = self.raw_df[suspect].fillna(0)

            m_check = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            m_check.fit(X_anchor, y_suspect)
            pred_suspect = m_check.predict(X_anchor)
            explained_var = r2_score(y_suspect, pred_suspect)

            if explained_var > 0.90:
                # Residual-on-residual test (Double ML core)
                resid_suspect = y_suspect.values - pred_suspect

                m_sales = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                m_sales.fit(X_anchor, self.raw_df[self.target].fillna(0))
                resid_target = self.raw_df[self.target].fillna(0).values - m_sales.predict(X_anchor)

                from scipy.stats import pearsonr
                corr_resid, _ = pearsonr(resid_target, resid_suspect)

                if abs(corr_resid) < 0.15:
                    logger.info(f"  REJECTED '{suspect}': Pure proxy (R2={explained_var:.2f}, ResidCorr={corr_resid:.3f})")
                    killed.append(suspect)
                else:
                    logger.info(f"  KEPT '{suspect}': Unique signal exists (ResidCorr={corr_resid:.3f})")
                    survivors.append(suspect)
            else:
                logger.info(f"  VERIFIED '{suspect}': Unique driver (R2={explained_var:.2f})")
                survivors.append(suspect)

        self.known_features = list(set(survivors))
        self.causal_report = {"survivors": survivors, "killed": killed}
        return self.causal_report

    def detect_regimes_and_drift(self) -> Dict:
        """
        Phase 3: Regime Change Detection via PELT algorithm.
        Identifies zombie/decaying/surging keywords.
        """
        logger.info("[Discovery Phase 3] Regime Change Detection...")

        X = self.raw_df[self.known_features].fillna(0)
        y = self.raw_df[self.target].fillna(0)

        result = []
        if HAS_RUPTURES:
            try:
                global_model = Ridge().fit(X, y)
                residuals = (y.values - global_model.predict(X)).reshape(-1, 1)
                algo = rpt.Pelt(model="rbf").fit(residuals)
                result = algo.predict(pen=10)
            except Exception as e:
                logger.warning(f"  Ruptures scan failed: {e}")

        regime_report = {"changepoints": result, "feature_status": {}}

        if len(result) > 1:
            last_cp = result[-2]
            logger.info(f"  REGIME CHANGE detected at index {last_cp}")

            df_past = self.raw_df.iloc[:last_cp]
            df_recent = self.raw_df.iloc[last_cp:]

            if len(df_recent) >= 20:
                m_past = LinearRegression().fit(df_past[self.known_features].fillna(0), df_past[self.target].fillna(0))
                m_recent = LinearRegression().fit(df_recent[self.known_features].fillna(0), df_recent[self.target].fillna(0))

                for i, feat in enumerate(self.known_features):
                    past_coef = m_past.coef_[i]
                    recent_coef = m_recent.coef_[i]
                    mag_past = abs(past_coef)
                    mag_recent = abs(recent_coef)

                    if mag_past < 0.001:
                        ratio = 1.0
                    else:
                        ratio = mag_recent / mag_past

                    if ratio < 0.2:
                        status = "ZOMBIE"
                    elif ratio < 0.5:
                        status = "DECAYING"
                    elif ratio > 1.5:
                        status = "SURGING"
                    else:
                        status = "STABLE"

                    regime_report["feature_status"][feat] = {
                        "past_coef": float(past_coef),
                        "recent_coef": float(recent_coef),
                        "ratio": float(ratio),
                        "status": status
                    }
                    if status in ("ZOMBIE", "SURGING"):
                        logger.info(f"  {feat}: {status} (ratio={ratio:.2f})")
        else:
            logger.info("  No regime change detected (Stable).")

        self.regime_report = regime_report
        return regime_report


# ==============================================================================
# LAYER 4: UNIVERSAL ATTRIBUTION VALIDATOR (from unified-attribution-framework.ipynb)
# ==============================================================================

class AdsAttributionValidator:
    """
    LAYER 4: Attribution via Integrated Gradients + Shapley Interaction Index.
    Detects synergy between ad features (e.g., keyword + match_type combos).
    Falls back to XGBoost feature importance if PyTorch unavailable.
    """

    def __init__(self, X_df: pd.DataFrame, y_series: pd.Series):
        self.features = X_df.columns.tolist()
        self.X_raw = X_df.copy()
        self.y_raw = y_series.copy()
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X_df.fillna(0))
        self.ig_scores = None
        self.inter_scores = None
        self.regime_status = "UNKNOWN"

        if HAS_TORCH:
            self.X_tensor = torch.FloatTensor(self.X_scaled).to(DEVICE)
            self.y_tensor = torch.FloatTensor(y_series.values).reshape(-1, 1).to(DEVICE)
            self._train_neural_proxy()
        else:
            self.model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
            self.model.fit(self.X_scaled, y_series.values)
            logger.info("  (Using XGBoost fallback — install PyTorch for neural attribution)")

        logger.info(f"Layer 4: AdsAttributionValidator online. "
                     f"{len(self.features)} features, {len(X_df)} samples.")

    def _train_neural_proxy(self):
        """Train noise-injected neural proxy (from UAV notebook)."""
        n_feat = self.X_tensor.shape[1]
        self.nn_model = nn.Sequential(
            nn.Linear(n_feat, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1)
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.005, weight_decay=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(500):
            optimizer.zero_grad()
            noise = torch.randn_like(self.X_tensor) * 0.05
            y_pred = self.nn_model(self.X_tensor + noise)
            loss = loss_fn(y_pred, self.y_tensor)
            loss.backward()
            optimizer.step()

        r2 = 1 - (loss.item() / torch.var(self.y_tensor).item())
        logger.info(f"  Neural Proxy trained (R2: {r2:.3f})")

    def compute_attribution(self, steps: int = 100) -> pd.Series:
        """Integrated Gradients attribution (exact from UAV notebook)."""
        if not HAS_TORCH:
            # Fallback: XGBoost feature importance
            imp = pd.Series(self.model.feature_importances_, index=self.features)
            self.ig_scores = pd.DataFrame([imp.values] * len(self.X_raw), columns=self.features)
            return imp.sort_values(ascending=False)

        logger.info("[Attribution Phase 1] Computing Integrated Gradients...")
        baseline = torch.zeros_like(self.X_tensor)
        attributions = []
        batch_size = 2000

        for i in range(0, len(self.X_tensor), batch_size):
            bs = min(batch_size, len(self.X_tensor) - i)
            batch_X = self.X_tensor[i:i + bs]
            batch_base = baseline[i:i + bs]

            alphas = torch.linspace(0, 1, steps).to(DEVICE)
            path = batch_base.unsqueeze(0) + alphas.view(-1, 1, 1) * (batch_X - batch_base).unsqueeze(0)
            path.requires_grad = True

            preds = self.nn_model(path.reshape(-1, self.X_tensor.shape[1]))
            grads = torch.autograd.grad(torch.sum(preds), path)[0]
            attr = (batch_X - batch_base) * torch.mean(grads, dim=0)
            attributions.append(attr.detach().cpu().numpy())

        self.ig_scores = pd.DataFrame(np.vstack(attributions), columns=self.features)
        return self.ig_scores.mean().sort_values(ascending=False)

    def compute_interactions(self, top_n: int = 5) -> pd.Series:
        """Shapley Interaction Index for feature synergy detection."""
        logger.info("[Attribution Phase 2] Scanning for Synergies...")

        if not HAS_TORCH:
            self.inter_scores = pd.Series(dtype=float)
            return self.inter_scores

        idx = np.random.choice(len(self.X_tensor), min(500, len(self.X_tensor)), replace=False)
        X_s = self.X_tensor[idx]
        base = torch.mean(self.X_tensor, dim=0)

        interactions = {}
        corr = self.X_raw.corr().abs()
        pairs = [(c, r) for c in corr.columns for r in corr.columns
                  if c < r and corr.loc[c, r] > 0.3]

        for name_a, name_b in pairs[:20]:  # Cap at 20 pairs for speed
            if name_a not in self.features or name_b not in self.features:
                continue
            idx_a = self.features.index(name_a)
            idx_b = self.features.index(name_b)

            X_00, X_11 = X_s.clone(), X_s.clone()
            X_10, X_01 = X_s.clone(), X_s.clone()

            X_00[:, [idx_a, idx_b]] = base[[idx_a, idx_b]]
            X_10[:, idx_b] = base[idx_b]
            X_01[:, idx_a] = base[idx_a]

            with torch.no_grad():
                val = (self.nn_model(X_11) - self.nn_model(X_10) -
                       self.nn_model(X_01) + self.nn_model(X_00))
                interactions[f"{name_a} + {name_b}"] = val.mean().item()

        self.inter_scores = pd.Series(interactions).sort_values(key=abs, ascending=False).head(top_n)
        return self.inter_scores

    def detect_regimes(self) -> str:
        """K-Means regime detection on attribution vectors."""
        logger.info("[Attribution Phase 3] Scanning for Regime Changes...")

        if self.ig_scores is None or len(self.ig_scores) < 10:
            self.regime_status = "INSUFFICIENT_DATA"
            return self.regime_status

        attrs = self.ig_scores.values
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(attrs)
            score = silhouette_score(attrs, kmeans.labels_)
            self.regime_status = "STABLE" if score < 0.5 else "MULTI-REGIME"
            logger.info(f"  Clustering Score: {score:.3f} ({self.regime_status})")
        except Exception:
            self.regime_status = "UNKNOWN"

        return self.regime_status


# ==============================================================================
# LAYER 5: INTERVENTION SIMULATION (from unified-intervention-framework.ipynb)
# ==============================================================================

class AdsCausalEngine:
    """
    LAYER 5: Do-Calculus intervention simulation for Google Ads.
    Implements PlatinumCausalEngine adapted for bid/budget simulations.
    Resolves Simpson's Paradox (expensive clicks may actually be cheaper per acquisition).
    """

    def __init__(self, data: pd.DataFrame, causal_graph: list = None, verbose: bool = True):
        self.df = data.select_dtypes(include=[np.number]).copy()
        self.verbose = verbose
        self.models = {}
        self.model_types = {}
        self.residuals = pd.DataFrame(index=self.df.index)

        # Build default ads causal graph if none provided
        if causal_graph is None:
            causal_graph = self._build_ads_causal_graph()

        self.G = nx.DiGraph()
        self.G.add_edges_from(causal_graph)

        # Only keep nodes that exist in data
        valid_nodes = [n for n in self.G.nodes() if n in self.df.columns]
        self.G = self.G.subgraph(valid_nodes).copy()
        self.nodes = list(nx.topological_sort(self.G))

        self._fit_adaptive_models()
        self._compute_residuals()
        logger.info(f"Layer 5: AdsCausalEngine online. {len(self.nodes)} causal nodes, "
                     f"{self.G.number_of_edges()} edges.")

    def _build_ads_causal_graph(self) -> list:
        """Build a default causal DAG for Google Ads."""
        edges = []
        cols = set(self.df.columns)

        # Quality Score -> CTR -> Conversions
        if "quality_score" in cols and "ctr" in cols:
            edges.append(("quality_score", "ctr"))
        if "ctr" in cols and "conversions" in cols:
            edges.append(("ctr", "conversions"))

        # Avg CPC -> Cost -> CPA
        if "avg_cpc" in cols and "cost" in cols:
            edges.append(("avg_cpc", "cost"))
        if "cost" in cols and "cpa" in cols:
            edges.append(("cost", "cpa"))

        # Impressions -> Clicks -> Conversions
        if "impressions" in cols and "clicks" in cols:
            edges.append(("impressions", "clicks"))
        if "clicks" in cols and "conversions" in cols:
            edges.append(("clicks", "conversions"))

        # Quality Score -> Avg CPC (higher QS = lower CPC)
        if "quality_score" in cols and "avg_cpc" in cols:
            edges.append(("quality_score", "avg_cpc"))

        # Conversions -> Conv Value -> ROAS
        if "conversions" in cols and "conv_value" in cols:
            edges.append(("conversions", "conv_value"))
        if "conv_value" in cols and "roas" in cols:
            edges.append(("conv_value", "roas"))
        if "cost" in cols and "roas" in cols:
            edges.append(("cost", "roas"))

        # Feature interactions -> target metrics
        for feat in ["kw_length", "kw_word_count", "is_exact", "is_phrase",
                      "kw_length_x_exact", "cpc_x_qs", "ctr_x_impr"]:
            if feat in cols and "conversions" in cols:
                edges.append((feat, "conversions"))

        return edges

    def _fit_adaptive_models(self):
        """3-fold CV model selection: Linear vs XGBoost per node."""
        for node in self.nodes:
            parents = list(self.G.predecessors(node))
            if not parents:
                self.residuals[node] = self.df[node]
                continue

            X = self.df[parents].fillna(0)
            y = self.df[node].fillna(0)

            if len(X) < 10:
                self.models[node] = LinearRegression().fit(X, y)
                self.model_types[node] = "Linear (Small N)"
                continue

            lin_scores, xgb_scores = [], []
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            try:
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    lin = LinearRegression().fit(X_tr, y_tr)
                    lin_scores.append(lin.score(X_val, y_val))

                    xg = xgb.XGBRegressor(n_estimators=50, max_depth=4, n_jobs=-1, verbosity=0)
                    xg.fit(X_tr, y_tr)
                    xgb_scores.append(xg.score(X_val, y_val))

                avg_lin, avg_xgb = np.mean(lin_scores), np.mean(xgb_scores)

                if avg_xgb > avg_lin + 0.05:
                    self.models[node] = xgb.XGBRegressor(
                        n_estimators=100, max_depth=5, learning_rate=0.05, n_jobs=-1, verbosity=0
                    ).fit(X, y)
                    self.model_types[node] = "XGBoost"
                else:
                    self.models[node] = LinearRegression().fit(X, y)
                    self.model_types[node] = "Linear"
            except Exception:
                self.models[node] = LinearRegression().fit(X, y)
                self.model_types[node] = "Linear (Fallback)"

    def _compute_residuals(self):
        for node in self.nodes:
            parents = list(self.G.predecessors(node))
            if not parents:
                continue
            try:
                X = self.df[parents].fillna(0)
                pred = self.models[node].predict(X)
                self.residuals[node] = self.df[node].values - pred
            except Exception:
                self.residuals[node] = 0.0

    def simulate_intervention(self, treatment: dict, target: str, n_boot: int = 0) -> dict:
        """
        Do-calculus: E[target | do(treatment=value)].
        With optional bootstrap confidence intervals.
        """
        mu = self._run_simulation_pass(self.df, self.residuals, treatment, target)

        result = {
            "E_y_do": float(mu),
            "std_error": 0.0,
            "ci_lower": float(mu),
            "ci_upper": float(mu),
            "model_used": self.model_types.get(target, "Direct"),
            "treatment": treatment,
        }

        if n_boot > 0:
            estimates = []
            for _ in range(n_boot):
                res_boot = self.residuals.sample(frac=1.0, replace=True)
                df_boot = self.df.loc[res_boot.index].copy()
                val = self._run_simulation_pass(df_boot, res_boot, treatment, target)
                estimates.append(val)

            result["ci_lower"] = float(np.percentile(estimates, 2.5))
            result["ci_upper"] = float(np.percentile(estimates, 97.5))
            result["std_error"] = float(np.std(estimates))

        return result

    def _run_simulation_pass(self, df_base, res_base, treatment, target):
        df_sim = df_base.copy()
        for t_var, t_val in treatment.items():
            if t_var in df_sim.columns:
                df_sim[t_var] = t_val

        for node in self.nodes:
            if node in treatment:
                continue
            parents = list(self.G.predecessors(node))
            if not parents or node not in self.models:
                continue
            try:
                X = df_sim[parents].fillna(0)
                base_val = self.models[node].predict(X)
                resid = res_base[node].values if node in res_base.columns else 0
                df_sim[node] = base_val + resid
            except Exception:
                pass

        return df_sim[target].mean() if target in df_sim.columns else 0.0


# ==============================================================================
# LAYER 6: INFINITY OPTIMIZATION (from gradient-boost + unified-optimization)
# ==============================================================================

class AdsInfinityOptimizer:
    """
    LAYER 6: Multi-strategy optimization combining:
    - Bayesian Optimization (GP + Expected Improvement) from unified-optimization
    - Evolutionary Search (Differential Evolution)
    - Infinity Loop (10-Seed LightGBM Averaging) from gradient-boost-ai-opt
    - Manifold-regularized search from op-system-opt
    """

    def __init__(self, objective_fn, bounds: dict, validation_framework=None):
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.history = {"params": [], "values": [], "method": []}
        self.tvf = validation_framework
        logger.info(f"Layer 6: AdsInfinityOptimizer online. {len(self.param_names)} params.")

    def _dict_to_array(self, d):
        return np.array([d[k] for k in self.param_names])

    def _array_to_dict(self, a):
        return dict(zip(self.param_names, a))

    def _evaluate(self, params_dict, method_name):
        value = self.objective_fn(**params_dict)
        self.history["params"].append(params_dict)
        self.history["values"].append(value)
        self.history["method"].append(method_name)
        return value

    def bayesian_optimize(self, n_init: int = 5, n_iter: int = 20) -> Tuple[dict, float]:
        """Bayesian Optimization with GP + Expected Improvement."""
        logger.info("[Optimizer] Running Bayesian Optimization...")
        X, y = [], []

        sampler = qmc.LatinHypercube(d=len(self.param_names))
        sample = sampler.random(n=n_init)

        for s in sample:
            params_dict = {}
            for j, (name, (low, high)) in enumerate(self.bounds.items()):
                params_dict[name] = low + s[j] * (high - low)
            value = self._evaluate(params_dict, "bayesian")
            X.append(self._dict_to_array(params_dict))
            y.append(value)

        best_value = min(y)

        for i in range(n_iter):
            kernel = C_Kernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True
            )
            gp.fit(np.array(X), np.array(y))

            best_candidate = None
            best_acq = -np.inf

            candidates = qmc.LatinHypercube(d=len(self.param_names)).random(n=500)
            for candidate in candidates:
                x_dict = {}
                for j, (name, (low, high)) in enumerate(self.bounds.items()):
                    x_dict[name] = low + candidate[j] * (high - low)

                x_arr = self._dict_to_array(x_dict).reshape(1, -1)
                mu, sigma = gp.predict(x_arr, return_std=True)

                improvement = best_value - mu
                Z = improvement / (sigma + 1e-9)
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

                if ei > best_acq:
                    best_acq = ei
                    best_candidate = x_dict

            value = self._evaluate(best_candidate, "bayesian")
            X.append(self._dict_to_array(best_candidate))
            y.append(value)

            if value < best_value:
                best_value = value

        best_idx = np.argmin(y)
        return self._array_to_dict(X[best_idx]), y[best_idx]

    def evolutionary_optimize(self, popsize: int = 15, maxiter: int = 50) -> Tuple[dict, float]:
        """Differential Evolution global search."""
        logger.info("[Optimizer] Running Evolutionary Search...")

        bounds_array = [self.bounds[k] for k in self.param_names]

        def wrapper(x):
            params = self._array_to_dict(x)
            return self._evaluate(params, "evolutionary")

        result = differential_evolution(
            wrapper, bounds_array, maxiter=maxiter, popsize=popsize,
            seed=42, polish=True, workers=1
        )
        return self._array_to_dict(result.x), result.fun

    def optimize_ensemble(self) -> Tuple[dict, float, dict]:
        """Run Bayesian + Evolutionary and return best result."""
        logger.info("[Optimizer] Running Ensemble Optimization...")
        results = {}

        try:
            params, value = self.bayesian_optimize(n_init=5, n_iter=15)
            results["bayesian"] = (params, value)
        except Exception as e:
            logger.warning(f"  Bayesian failed: {e}")

        try:
            params, value = self.evolutionary_optimize(popsize=10, maxiter=30)
            results["evolutionary"] = (params, value)
        except Exception as e:
            logger.warning(f"  Evolutionary failed: {e}")

        if not results:
            return {k: (v[0] + v[1]) / 2 for k, v in self.bounds.items()}, float("inf"), {}

        best_method = min(results.items(), key=lambda x: x[1][1])
        logger.info(f"  Best: {best_method[0].upper()} (value={best_method[1][1]:.4f})")
        return best_method[1][0], best_method[1][1], results

    @staticmethod
    def infinity_loop_predict(X_train, y_train, X_test, n_seeds: int = 10) -> np.ndarray:
        """
        10-Seed LightGBM Averaging from gradient-boost-ai-opt.ipynb.
        Variance reduction through seed averaging.
        """
        logger.info(f"[Infinity Loop] Running {n_seeds}-seed LightGBM ensemble...")

        params = {
            "num_leaves": 110,
            "learning_rate": 0.02,
            "n_estimators": 3500,
            "min_child_samples": 20,
            "lambda_l1": 0.74,
            "lambda_l2": 0.1,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_jobs": -1,
            "verbose": -1,
        }

        seeds = list(range(2025, 2025 + n_seeds))
        ensemble_preds = np.zeros(len(X_test))

        for seed in seeds:
            params["random_state"] = seed
            model = lgb.LGBMRegressor(**params)
            cat_features = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
            model.fit(X_train, y_train, categorical_feature=cat_features if cat_features else "auto")
            ensemble_preds += model.predict(X_test)

        ensemble_preds /= len(seeds)
        return ensemble_preds


# ==============================================================================
# LAYER 7: EXECUTION & HEALTH (from health.ipynb + op-system-opt.ipynb)
# ==============================================================================

class AdsInterventionVault:
    """
    LAYER 7: The "Ads Intervention Vault" — modeled after the Therapeutic Vault.
    Maps interventions (Pause Keyword, Increase Bid, Change Headline) to
    expected impact, then optimizes the intervention combination.
    Generates a Control Surface CSV/JSON.
    """

    def __init__(self):
        # Pharmacopeia -> Ads Interventions
        self.interventions = {
            # Bid Adjustments
            "Increase Bid +20%": {"target": "impressions", "potency": 0.20, "cost_factor": 1.20},
            "Decrease Bid -20%": {"target": "cost", "potency": -0.20, "cost_factor": 0.80},
            "Increase Bid +50%": {"target": "impressions", "potency": 0.45, "cost_factor": 1.50},

            # Status Changes
            "Pause Keyword": {"target": "cost", "potency": -1.0, "cost_factor": 0.0},
            "Enable Keyword": {"target": "impressions", "potency": 0.5, "cost_factor": 1.0},

            # Match Type Changes
            "Change to Exact Match": {"target": "ctr", "potency": 0.15, "cost_factor": 0.85},
            "Change to Broad Match": {"target": "impressions", "potency": 0.40, "cost_factor": 1.15},

            # Quality Improvements
            "Improve Ad Relevance": {"target": "quality_score", "potency": 1.5, "cost_factor": 0.95},
            "Optimize Landing Page": {"target": "quality_score", "potency": 2.0, "cost_factor": 0.90},

            # Budget Reallocation
            "Shift Budget to Top Performer": {"target": "conversions", "potency": 0.25, "cost_factor": 1.0},
            "Add Negative Keywords": {"target": "ctr", "potency": 0.10, "cost_factor": 0.92},

            # Ad Copy Changes
            "Test New Headline with CTA": {"target": "ctr", "potency": 0.08, "cost_factor": 1.0},
            "Add Price in Headline ($19/mo)": {"target": "conversions", "potency": 0.12, "cost_factor": 1.0},
        }
        self.all_interventions = list(self.interventions.keys())
        logger.info(f"Layer 7: AdsInterventionVault online. {len(self.all_interventions)} interventions available.")

    def get_effects(self, active_indices: list) -> Tuple[dict, list]:
        """Get cumulative effects of active interventions."""
        effect_map = {}
        active = [self.all_interventions[i] for i in range(len(active_indices))
                   if i < len(self.all_interventions) and active_indices[i] > 0.5]

        for intervention in active:
            data = self.interventions[intervention]
            tgt = data["target"]
            pot = data["potency"]
            if tgt not in effect_map:
                effect_map[tgt] = 0.0
            effect_map[tgt] += pot

        return effect_map, active

    def optimize_interventions(self, current_metrics: dict, target_metric: str = "roas") -> dict:
        """
        Basin-hopping optimizer to find best intervention combination.
        Modeled after the Therapeutic Vault's basinhopping approach.
        """
        logger.info("[Intervention Vault] Optimizing intervention combination...")

        def simulate(x):
            effects, active = self.get_effects(x)
            projected = current_metrics.copy()

            for tgt, pot in effects.items():
                if tgt in projected:
                    if tgt in ("cost",):
                        projected[tgt] *= max(0.01, (1.0 + pot))
                    elif tgt in ("quality_score",):
                        projected[tgt] = min(10, projected[tgt] + pot)
                    else:
                        projected[tgt] *= (1.0 + pot)

            # Causal ripples (like health.ipynb's inflammation -> BDNF)
            qs_delta = projected.get("quality_score", 5) - current_metrics.get("quality_score", 5)
            if "avg_cpc" in projected:
                projected["avg_cpc"] *= max(0.5, 1.0 - 0.03 * qs_delta)
            if "ctr" in projected:
                projected["ctr"] *= (1.0 + 0.02 * qs_delta)

            # Recalculate derived metrics
            projected["cost"] = projected.get("clicks", 0) * projected.get("avg_cpc", 0)
            projected["conv_value"] = projected.get("conversions", 0) * 19.0  # $19/mo
            projected["roas"] = projected["conv_value"] / (projected["cost"] + 1e-9)
            projected["cpa"] = projected["cost"] / (projected["conversions"] + 1e-9)

            # Objective: minimize negative ROAS (= maximize ROAS)
            if target_metric == "roas":
                return -projected["roas"]
            elif target_metric == "cpa":
                return projected["cpa"]
            elif target_metric == "conversions":
                return -projected["conversions"]
            return -projected.get(target_metric, 0)

        x0 = [0.0] * len(self.all_interventions)
        bounds = [(0, 1)] * len(self.all_interventions)

        result = basinhopping(
            simulate, x0, niter=50,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
            seed=2026
        )

        _, best_interventions = self.get_effects(result.x)
        return {
            "best_interventions": best_interventions,
            "objective_value": -result.fun if target_metric in ("roas", "conversions") else result.fun,
            "activation_vector": result.x.tolist(),
        }


def generate_control_surface(
    keyword_data: pd.DataFrame,
    discovery_report: dict,
    attribution_scores: pd.Series,
    causal_results: dict,
    intervention_results: dict,
    output_path: str = "contractkit_optimization_surface.json"
) -> dict:
    """
    Generate the final Control Surface (from op-system-opt.ipynb).
    This is the executable output: exact API calls needed.
    """
    logger.info("[Control Surface] Generating optimization surface...")

    actions = []

    # 1. Keyword-level actions from regime detection
    regime_status = discovery_report.get("feature_status", {})
    for feat, info in regime_status.items():
        if info["status"] == "ZOMBIE":
            actions.append({
                "type": "PAUSE_KEYWORD",
                "entity": feat,
                "reason": f"Zombie metric (coefficient ratio: {info['ratio']:.2f})",
                "priority": "HIGH",
                "expected_impact": "Reduce wasted spend"
            })
        elif info["status"] == "SURGING":
            actions.append({
                "type": "INCREASE_BID",
                "entity": feat,
                "reason": f"Surging metric (coefficient ratio: {info['ratio']:.2f})",
                "priority": "HIGH",
                "bid_change_pct": 20,
                "expected_impact": "Capture growing demand"
            })

    # 2. Actions from attribution (top positive and negative drivers)
    if attribution_scores is not None and len(attribution_scores) > 0:
        top_positive = attribution_scores.head(3)
        top_negative = attribution_scores.tail(3)

        for feat, score in top_positive.items():
            actions.append({
                "type": "INCREASE_ALLOCATION",
                "entity": feat,
                "attribution_score": float(score),
                "reason": f"Top positive driver (score: {score:.4f})",
                "priority": "MEDIUM"
            })

    # 3. Actions from intervention vault
    if intervention_results:
        for intervention in intervention_results.get("best_interventions", []):
            actions.append({
                "type": "APPLY_INTERVENTION",
                "intervention": intervention,
                "reason": "Optimized by Infinity Engine",
                "priority": "MEDIUM"
            })

    # 4. Keyword-specific recommendations
    if keyword_data is not None and "keyword_text" in keyword_data.columns:
        agg = keyword_data.groupby("keyword_text").agg({
            "clicks": "sum", "impressions": "sum", "conversions": "sum",
            "cost": "sum", "conv_value": "sum"
        }).reset_index()

        agg["cpa"] = agg["cost"] / (agg["conversions"] + 1e-9)
        agg["roas"] = agg["conv_value"] / (agg["cost"] + 1e-9)

        # Pause zero-conversion keywords with high spend
        zombies = agg[(agg["conversions"] == 0) & (agg["cost"] > agg["cost"].median())]
        for _, row in zombies.iterrows():
            actions.append({
                "type": "PAUSE_KEYWORD",
                "keyword": row["keyword_text"],
                "reason": f"Zero conversions, ${row['cost']:.2f} wasted spend",
                "priority": "HIGH",
                "cost_savings": float(row["cost"])
            })

        # Boost high-ROAS keywords
        stars = agg[(agg["roas"] > 2.0) & (agg["conversions"] > 0)]
        for _, row in stars.iterrows():
            actions.append({
                "type": "INCREASE_BID",
                "keyword": row["keyword_text"],
                "reason": f"ROAS {row['roas']:.1f}x — scale this keyword",
                "priority": "HIGH",
                "bid_change_pct": 25,
                "current_roas": float(row["roas"])
            })

    # Build the control surface
    surface = {
        "meta": {
            "version": "contractkit_infinity_v1",
            "generated_at": datetime.now().isoformat(),
            "description": "ContractKit Infinity Ads Engine - Optimization Surface",
            "target_product": "ContractKit ($19/mo subscription)",
            "optimization_target": "Maximize ROAS / Subscriber Acquisitions"
        },
        "summary": {
            "total_actions": len(actions),
            "high_priority": len([a for a in actions if a.get("priority") == "HIGH"]),
            "medium_priority": len([a for a in actions if a.get("priority") == "MEDIUM"]),
        },
        "actions": actions,
    }

    # Add causal simulation results if available
    if causal_results:
        surface["causal_simulations"] = causal_results

    # Save
    with open(output_path, "w") as f:
        json.dump(surface, f, indent=2, default=str)
    logger.info(f"  Saved to: {output_path}")

    # Also save CSV version
    csv_path = output_path.replace(".json", ".csv")
    pd.DataFrame(actions).to_csv(csv_path, index=False)
    logger.info(f"  Saved CSV to: {csv_path}")

    return surface


# ==============================================================================
# MASTER ORCHESTRATOR: THE CONTRACTKIT INFINITY LOOP
# ==============================================================================

class ContractKitInfinityEngine:
    """
    The Grand Orchestrator — runs all 7 layers in sequence:
    Ingest -> Validate -> Discover -> Attribute -> Simulate -> Optimize -> Execute
    """

    def __init__(self, customer_id: str = None):
        self.customer_id = customer_id
        self.ingestion = AdsDataIngestion(customer_id=customer_id)
        self.data = None
        self.validation_report = None
        self.discovery_report = None
        self.attribution_scores = None
        self.interaction_scores = None
        self.causal_results = {}
        self.optimization_results = None
        self.intervention_results = None
        self.control_surface = None
        logger.info("=" * 70)
        logger.info("CONTRACTKIT INFINITY ADS ENGINE v1.0")
        logger.info("7-Layer Automated Google Ads Optimization System")
        logger.info("=" * 70)

    def run_infinity_loop(self, target_col: str = "conversions") -> dict:
        """
        Execute the full Infinity Loop:
        Validate -> Discover -> Attribute -> Simulate -> Optimize
        """
        start_time = time.time()

        # ── LAYER 1: INGESTION ──────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 1: DATA INGESTION")
        logger.info("=" * 60)
        raw_data = self.ingestion.fetch_all(self.customer_id)
        self.data = self.ingestion.unified_df

        if self.data is None or self.data.empty:
            logger.error("No data available. Aborting.")
            return {"error": "No data"}

        logger.info(f"  Unified DataFrame: {self.data.shape}")

        # Prepare numeric-only view for analysis
        num_df = self.data.select_dtypes(include=[np.number]).copy()
        if target_col not in num_df.columns:
            # Try conv_value as alternative target
            if "conv_value" in num_df.columns:
                target_col = "conv_value"
            elif "clicks" in num_df.columns:
                target_col = "clicks"
            logger.info(f"  Using target: {target_col}")

        # ── LAYER 2: VALIDATION ─────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 2: FULL SPECTRUM VALIDATION")
        logger.info("=" * 60)

        # Use first 50% as reference, validate against full set
        ref_size = max(100, len(num_df) // 2)
        ref_data = num_df.iloc[:ref_size]
        validator = AdsValidationFramework(ref_data)
        passed, self.validation_report = validator.validate(
            num_df, target_col=target_col, subgroups=["match_type"] if "match_type" in self.data.columns else None
        )

        tl = self.validation_report["traffic_light"]
        n_red = len(tl[tl["Status"] == "RED"])
        logger.info(f"  Traffic Light: {'GREEN' if passed else 'RED'} ({n_red} issues)")
        if not passed:
            for _, row in tl[tl["Status"] == "RED"].iterrows():
                logger.warning(f"    [{row['Module']}] {row['Reason']}")

        # ── LAYER 3: DISCOVERY ──────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 3: UNIFIED DISCOVERY")
        logger.info("=" * 60)

        discovery = AdsDiscoveryEngine(num_df, target_col)
        lag_suggestion = discovery.scan_environment()
        if lag_suggestion and lag_suggestion.replace(f"{target_col}_Lag_", "") not in num_df.columns:
            lag_n = int(lag_suggestion.split("_")[-1])
            num_df[lag_suggestion] = num_df[target_col].shift(lag_n).fillna(0)
            discovery = AdsDiscoveryEngine(num_df, target_col)

        causal_report = discovery.verify_causality_nonlinear()
        regime_report = discovery.detect_regimes_and_drift()
        self.discovery_report = {
            "causal": causal_report,
            "regime": regime_report,
            "surviving_features": discovery.known_features
        }

        # ── LAYER 4: ATTRIBUTION ────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 4: UNIVERSAL ATTRIBUTION")
        logger.info("=" * 60)

        features_for_attr = [f for f in discovery.known_features if f in num_df.columns and f != target_col]
        if len(features_for_attr) >= 2:
            attributor = AdsAttributionValidator(
                num_df[features_for_attr].fillna(0),
                num_df[target_col].fillna(0)
            )
            self.attribution_scores = attributor.compute_attribution()
            self.interaction_scores = attributor.compute_interactions()
            regime_status = attributor.detect_regimes()

            logger.info("  Top Attribution Drivers:")
            for feat, score in self.attribution_scores.head(5).items():
                logger.info(f"    {feat}: {score:.4f}")
            if self.interaction_scores is not None and len(self.interaction_scores) > 0:
                logger.info("  Top Interactions (Synergy/Redundancy):")
                for pair, score in self.interaction_scores.items():
                    label = "SYNERGY" if score > 0 else "REDUNDANCY"
                    logger.info(f"    {pair}: {score:.4f} ({label})")
        else:
            logger.warning("  Insufficient features for attribution analysis.")

        # ── LAYER 5: INTERVENTION SIMULATION ────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 5: CAUSAL INTERVENTION SIMULATION")
        logger.info("=" * 60)

        causal_engine = AdsCausalEngine(num_df)

        # Simulate key interventions
        simulations = {}
        if "avg_cpc" in num_df.columns:
            current_cpc = num_df["avg_cpc"].median()

            for pct_change in [0.8, 0.9, 1.1, 1.2, 1.5]:
                label = f"CPC_{int(pct_change * 100)}pct"
                new_cpc = current_cpc * pct_change
                result = causal_engine.simulate_intervention(
                    {"avg_cpc": new_cpc}, target_col, n_boot=50
                )
                simulations[label] = result
                logger.info(f"  do(CPC={new_cpc:.2f}): E[{target_col}]={result['E_y_do']:.2f} "
                             f"CI=[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")

        self.causal_results = simulations

        # ── LAYER 6: OPTIMIZATION ───────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 6: INFINITY OPTIMIZATION")
        logger.info("=" * 60)

        # Define optimization objective using the causal engine
        def ads_objective(cpc_multiplier=1.0, budget_multiplier=1.0):
            """Maximize ROAS through CPC and budget adjustment."""
            new_cpc = current_cpc * cpc_multiplier if "avg_cpc" in num_df.columns else 1.0
            result = causal_engine.simulate_intervention(
                {"avg_cpc": new_cpc}, target_col
            )
            projected_conversions = max(result["E_y_do"], 0)
            projected_revenue = projected_conversions * 19.0  # $19/mo
            projected_cost = new_cpc * num_df["clicks"].mean() * budget_multiplier
            roas = projected_revenue / (projected_cost + 1e-9)
            return -roas  # Minimize negative ROAS = maximize ROAS

        opt_bounds = {
            "cpc_multiplier": (0.5, 2.0),
            "budget_multiplier": (0.5, 2.0),
        }

        optimizer = AdsInfinityOptimizer(ads_objective, opt_bounds, validation_framework=validator)
        try:
            best_params, best_value, all_results = optimizer.optimize_ensemble()
            self.optimization_results = {
                "best_params": best_params,
                "best_roas": -best_value,
                "all_results": {k: {"params": v[0], "value": v[1]} for k, v in all_results.items()},
            }
            logger.info(f"  Optimal CPC Multiplier: {best_params.get('cpc_multiplier', 1.0):.2f}x")
            logger.info(f"  Optimal Budget Multiplier: {best_params.get('budget_multiplier', 1.0):.2f}x")
            logger.info(f"  Projected ROAS: {-best_value:.2f}x")
        except Exception as e:
            logger.warning(f"  Optimization failed: {e}")
            self.optimization_results = {}

        # ── LAYER 7: EXECUTION ──────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("LAYER 7: EXECUTION & INTERVENTION VAULT")
        logger.info("=" * 60)

        vault = AdsInterventionVault()

        # Get current aggregate metrics
        current_metrics = {
            "impressions": num_df["impressions"].mean() if "impressions" in num_df else 0,
            "clicks": num_df["clicks"].mean() if "clicks" in num_df else 0,
            "ctr": num_df["ctr"].mean() if "ctr" in num_df else 0,
            "avg_cpc": num_df["avg_cpc"].mean() if "avg_cpc" in num_df else 0,
            "cost": num_df["cost"].mean() if "cost" in num_df else 0,
            "conversions": num_df["conversions"].mean() if "conversions" in num_df else 0,
            "conv_value": num_df.get("conv_value", pd.Series([0])).mean(),
            "quality_score": num_df["quality_score"].mean() if "quality_score" in num_df else 5,
        }

        self.intervention_results = vault.optimize_interventions(current_metrics, target_metric="roas")
        logger.info(f"  Best Interventions: {self.intervention_results['best_interventions']}")
        logger.info(f"  Projected ROAS: {self.intervention_results['objective_value']:.2f}x")

        # Generate Control Surface
        self.control_surface = generate_control_surface(
            keyword_data=self.data,
            discovery_report=regime_report,
            attribution_scores=self.attribution_scores,
            causal_results=self.causal_results,
            intervention_results=self.intervention_results,
        )

        # ── DASHBOARD ───────────────────────────────────────────────
        elapsed = time.time() - start_time
        self._print_dashboard(elapsed)

        return self.control_surface

    def _print_dashboard(self, elapsed: float):
        """Print the text-based Infinity Dashboard (like health.ipynb)."""
        print("\n")
        print("=" * 70)
        print("  CONTRACTKIT INFINITY ADS ENGINE - OPTIMIZATION DASHBOARD")
        print("=" * 70)

        # Validation Status
        if self.validation_report:
            tl = self.validation_report["traffic_light"]
            n_red = len(tl[tl["Status"] == "RED"])
            status = "GREEN (All Clear)" if n_red == 0 else f"RED ({n_red} issues)"
            print(f"\n  [HEALTH]  Account Status: {status}")
            if n_red > 0:
                for _, row in tl[tl["Status"] == "RED"].head(5).iterrows():
                    print(f"            - {row['Module']}: {row['Reason']}")

        # Discovery
        if self.discovery_report:
            survivors = self.discovery_report.get("surviving_features", [])
            killed = self.discovery_report.get("causal", {}).get("killed", [])
            print(f"\n  [DISCOVERY]  Causal Drivers: {len(survivors)} verified, {len(killed)} proxies killed")

            regime = self.discovery_report.get("regime", {}).get("feature_status", {})
            zombies = [k for k, v in regime.items() if v["status"] == "ZOMBIE"]
            surging = [k for k, v in regime.items() if v["status"] == "SURGING"]
            if zombies:
                print(f"               Zombies: {zombies}")
            if surging:
                print(f"               Surging: {surging}")

        # Attribution
        if self.attribution_scores is not None and len(self.attribution_scores) > 0:
            print(f"\n  [ATTRIBUTION]  Top Drivers:")
            for feat, score in self.attribution_scores.head(5).items():
                bar = "+" * int(abs(score) * 50) if score > 0 else "-" * int(abs(score) * 50)
                print(f"    {feat:25s} [{bar:20s}] {score:.4f}")

        # Interactions
        if self.interaction_scores is not None and len(self.interaction_scores) > 0:
            print(f"\n  [SYNERGIES]  Feature Interactions:")
            for pair, score in self.interaction_scores.items():
                label = "SYNERGY" if score > 0 else "REDUNDANCY"
                print(f"    {pair:35s} {score:+.4f} ({label})")

        # Causal Simulations
        if self.causal_results:
            print(f"\n  [SIMULATIONS]  CPC Intervention Results:")
            for label, res in self.causal_results.items():
                print(f"    {label:15s}: E[conv]={res['E_y_do']:.2f} "
                      f"CI=[{res['ci_lower']:.2f}, {res['ci_upper']:.2f}]")

        # Optimization
        if self.optimization_results:
            print(f"\n  [OPTIMIZATION]")
            print(f"    Best CPC Multiplier:    {self.optimization_results.get('best_params', {}).get('cpc_multiplier', 'N/A'):.2f}x")
            print(f"    Best Budget Multiplier: {self.optimization_results.get('best_params', {}).get('budget_multiplier', 'N/A'):.2f}x")
            print(f"    Projected ROAS:         {self.optimization_results.get('best_roas', 0):.2f}x")

        # Interventions
        if self.intervention_results:
            print(f"\n  [INTERVENTIONS]  Recommended Actions:")
            for action in self.intervention_results.get("best_interventions", []):
                print(f"    -> {action}")

        # Control Surface
        if self.control_surface:
            n_actions = self.control_surface.get("summary", {}).get("total_actions", 0)
            n_high = self.control_surface.get("summary", {}).get("high_priority", 0)
            print(f"\n  [CONTROL SURFACE]  {n_actions} actions generated ({n_high} HIGH priority)")
            print(f"    Output: contractkit_optimization_surface.json")
            print(f"    Output: contractkit_optimization_surface.csv")

        print(f"\n  [ENGINE]  Total execution time: {elapsed:.1f}s")
        print("=" * 70)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContractKit Infinity Ads Engine")
    parser.add_argument("--customer-id", type=str, default=None,
                        help="Google Ads Customer ID (e.g., 1234567890)")
    parser.add_argument("--target", type=str, default="conversions",
                        help="Target metric to optimize (default: conversions)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    args = parser.parse_args()

    engine = ContractKitInfinityEngine(customer_id=args.customer_id)

    if args.synthetic:
        engine.ingestion._generate_synthetic_data()

    result = engine.run_infinity_loop(target_col=args.target)

    print(f"\nOptimization complete. {result.get('summary', {}).get('total_actions', 0)} actions ready.")
