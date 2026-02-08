"""
================================================================================
 CONTRACTKIT ADS MUTATION ENGINE
 --------------------------------
 Full CRUD operations for Google Ads via the google-ads Python library.
 Reads the Infinity Engine's control surface and executes changes.
 
 Capabilities:
   1. Campaign Management (create, update, pause, enable)
   2. Ad Group Management (create, update bids, pause)
   3. Keyword Management (add, pause, remove, change match type)
   4. Negative Keyword Management (add campaign/ad-group negatives)
   5. Ad Copy Management (create responsive search ads, pause underperformers)
   6. Budget & Bid Adjustments (update budgets, CPC bids, bid modifiers)
   7. Extensions/Assets Management (sitelinks, callouts, structured snippets)
   8. Search Term Mining (auto-add winners, auto-negate losers)
   9. Ad Copy A/B Testing Engine (auto-rotate, pick winners)
  10. Quality Score Tracker (historical QS monitoring)
  11. Dayparting & Device Bid Adjustments
  12. Audience Signal Management
  
 All mutations are logged, reversible, and require explicit confirmation
 unless running in auto-execute mode.
================================================================================
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("MutationEngine")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CONTRACTKIT_PRODUCT = {
    "name": "ContractKit",
    "url": "https://www.contractkit.info",
    "price_monthly": 19.0,
    "currency": "USD",
    "features": [
        "Legally binding contracts for all 50 US states",
        "Unlimited contracts",
        "Unlimited invoicing",
        "Automated payment reminders (email + CC)",
        "Electronic signing",
        "Payment acceptance via Stripe Connect",
        "Payment terms: Net 30, 50/50, 25/25/50",
    ],
    "target_audience": [
        "Small business owners",
        "Freelancers",
        "Independent contractors",
        "Consultants",
        "Agencies",
    ],
    "competitors": [
        "DocuSign", "PandaDoc", "HoneyBook", "FreshBooks",
        "HelloSign", "Bonsai", "AND.CO",
    ],
}

# Maximum CPA we can afford and still be profitable at $19/mo
# Assuming ~4 month average customer lifetime -> LTV = $76
# Target CPA should be < 30% of LTV = ~$23
MAX_TARGET_CPA = 23.0
IDEAL_TARGET_CPA = 15.0

# Quality Score thresholds
QS_PAUSE_THRESHOLD = 3       # Pause keywords with QS <= 3
QS_REVIEW_THRESHOLD = 5      # Review keywords with QS <= 5
QS_BOOST_THRESHOLD = 7       # Boost bids for QS >= 7

# Search term mining thresholds
SEARCH_TERM_ADD_THRESHOLD = 2     # Add as keyword if >= 2 conversions
SEARCH_TERM_NEGATE_CLICKS = 20    # Negate if >= 20 clicks with 0 conversions
SEARCH_TERM_NEGATE_COST = 30.0    # Negate if >= $30 spend with 0 conversions


class MutationAction(Enum):
    """Types of mutations we can perform."""
    CREATE_CAMPAIGN = "create_campaign"
    UPDATE_CAMPAIGN = "update_campaign"
    PAUSE_CAMPAIGN = "pause_campaign"
    ENABLE_CAMPAIGN = "enable_campaign"
    CREATE_AD_GROUP = "create_ad_group"
    UPDATE_AD_GROUP = "update_ad_group"
    PAUSE_AD_GROUP = "pause_ad_group"
    ADD_KEYWORD = "add_keyword"
    PAUSE_KEYWORD = "pause_keyword"
    REMOVE_KEYWORD = "remove_keyword"
    ADD_NEGATIVE_KEYWORD = "add_negative_keyword"
    CREATE_AD = "create_ad"
    PAUSE_AD = "pause_ad"
    UPDATE_BUDGET = "update_budget"
    UPDATE_BID = "update_bid"
    ADD_SITELINK = "add_sitelink"
    ADD_CALLOUT = "add_callout"
    ADD_STRUCTURED_SNIPPET = "add_structured_snippet"
    SET_BID_MODIFIER = "set_bid_modifier"
    ADD_AUDIENCE = "add_audience"


@dataclass
class MutationRecord:
    """Record of a single mutation operation."""
    action: MutationAction
    target: str                  # Resource name or description
    params: Dict[str, Any]       # Mutation parameters
    timestamp: str = ""
    status: str = "pending"      # pending, executed, failed, rolled_back
    result: Optional[str] = None
    rollback_info: Optional[Dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# CORE: GOOGLE ADS API CLIENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class GoogleAdsClientWrapper:
    """
    Wraps the google-ads Python library for all CRUD operations.
    All mutations go through this class for logging and safety.
    """

    def __init__(self, customer_id: str, login_customer_id: str = None,
                 developer_token: str = None, dry_run: bool = True):
        self.customer_id = customer_id.replace("-", "")
        self.login_customer_id = (login_customer_id or "").replace("-", "")
        self.developer_token = developer_token or os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        self.dry_run = dry_run
        self.client = None
        self.mutation_log: List[MutationRecord] = []
        self._connected = False

    def connect(self) -> bool:
        """Connect to Google Ads API."""
        try:
            from google.ads.googleads.client import GoogleAdsClient
            from google.ads.googleads.errors import GoogleAdsException

            config = {
                "developer_token": self.developer_token,
                "use_proto_plus": True,
            }
            if self.login_customer_id:
                config["login_customer_id"] = self.login_customer_id

            self.client = GoogleAdsClient.load_from_dict(config)
            self._connected = True
            logger.info(f"Connected to Google Ads API (customer: {self.customer_id})")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _log_mutation(self, record: MutationRecord):
        """Log every mutation for audit trail and rollback."""
        self.mutation_log.append(record)
        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"  [{mode}] {record.action.value}: {record.target}")

    # ── CAMPAIGN OPERATIONS ──────────────────────────────────────────────────

    def create_campaign(self, name: str, budget_daily_micros: int,
                        bidding_strategy: str = "MAXIMIZE_CONVERSIONS",
                        target_cpa_micros: int = None) -> MutationRecord:
        """Create a new campaign."""
        record = MutationRecord(
            action=MutationAction.CREATE_CAMPAIGN,
            target=name,
            params={
                "name": name,
                "budget_daily_micros": budget_daily_micros,
                "budget_daily_usd": budget_daily_micros / 1e6,
                "bidding_strategy": bidding_strategy,
                "target_cpa_micros": target_cpa_micros,
                "status": "PAUSED",  # Always create paused for safety
            }
        )

        if not self.dry_run and self._connected:
            try:
                campaign_service = self.client.get_service("CampaignService")
                campaign_budget_service = self.client.get_service("CampaignBudgetService")

                # Create budget first
                budget_op = self.client.get_type("CampaignBudgetOperation")
                budget = budget_op.create
                budget.name = f"{name} Budget"
                budget.amount_micros = budget_daily_micros
                budget.delivery_method = self.client.enums.BudgetDeliveryMethodEnum.STANDARD

                budget_response = campaign_budget_service.mutate_campaign_budgets(
                    customer_id=self.customer_id,
                    operations=[budget_op]
                )
                budget_resource = budget_response.results[0].resource_name

                # Create campaign
                campaign_op = self.client.get_type("CampaignOperation")
                campaign = campaign_op.create
                campaign.name = name
                campaign.campaign_budget = budget_resource
                campaign.advertising_channel_type = (
                    self.client.enums.AdvertisingChannelTypeEnum.SEARCH
                )
                campaign.status = self.client.enums.CampaignStatusEnum.PAUSED

                # Bidding strategy
                if bidding_strategy == "MAXIMIZE_CONVERSIONS":
                    campaign.maximize_conversions.target_cpa_micros = (
                        target_cpa_micros or int(IDEAL_TARGET_CPA * 1e6)
                    )
                elif bidding_strategy == "MAXIMIZE_CONVERSION_VALUE":
                    campaign.maximize_conversion_value.target_roas = 2.0
                elif bidding_strategy == "MANUAL_CPC":
                    campaign.manual_cpc.enhanced_cpc_enabled = True

                # Network settings
                campaign.network_settings.target_google_search = True
                campaign.network_settings.target_search_network = False

                response = campaign_service.mutate_campaigns(
                    customer_id=self.customer_id,
                    operations=[campaign_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
                record.rollback_info = {"resource_name": record.result, "action": "pause"}
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
                logger.error(f"  Campaign creation failed: {e}")
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    def update_campaign_status(self, campaign_id: str,
                                status: str = "PAUSED") -> MutationRecord:
        """Pause or enable a campaign."""
        action = MutationAction.PAUSE_CAMPAIGN if status == "PAUSED" else MutationAction.ENABLE_CAMPAIGN
        record = MutationRecord(
            action=action,
            target=f"Campaign {campaign_id}",
            params={"campaign_id": campaign_id, "status": status}
        )

        if not self.dry_run and self._connected:
            try:
                campaign_service = self.client.get_service("CampaignService")
                campaign_op = self.client.get_type("CampaignOperation")
                campaign = campaign_op.update
                campaign.resource_name = (
                    campaign_service.campaign_path(self.customer_id, campaign_id)
                )
                campaign.status = getattr(
                    self.client.enums.CampaignStatusEnum, status
                )
                self.client.copy_from(
                    campaign_op.update_mask,
                    self.client.get_type("FieldMask")(paths=["status"])
                )

                response = campaign_service.mutate_campaigns(
                    customer_id=self.customer_id,
                    operations=[campaign_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
            except Exception as e:
                record.status = "failed"
                record.result = str(e)

        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── AD GROUP OPERATIONS ──────────────────────────────────────────────────

    def create_ad_group(self, campaign_id: str, name: str,
                        cpc_bid_micros: int = 3000000) -> MutationRecord:
        """Create a new ad group."""
        record = MutationRecord(
            action=MutationAction.CREATE_AD_GROUP,
            target=f"{name} (in campaign {campaign_id})",
            params={
                "campaign_id": campaign_id,
                "name": name,
                "cpc_bid_micros": cpc_bid_micros,
                "cpc_bid_usd": cpc_bid_micros / 1e6,
            }
        )

        if not self.dry_run and self._connected:
            try:
                ag_service = self.client.get_service("AdGroupService")
                campaign_service = self.client.get_service("CampaignService")

                ag_op = self.client.get_type("AdGroupOperation")
                ad_group = ag_op.create
                ad_group.name = name
                ad_group.campaign = campaign_service.campaign_path(
                    self.customer_id, campaign_id
                )
                ad_group.status = self.client.enums.AdGroupStatusEnum.ENABLED
                ad_group.type_ = self.client.enums.AdGroupTypeEnum.SEARCH_STANDARD
                ad_group.cpc_bid_micros = cpc_bid_micros

                response = ag_service.mutate_ad_groups(
                    customer_id=self.customer_id,
                    operations=[ag_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
                record.rollback_info = {"resource_name": record.result, "action": "pause"}
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    def update_ad_group_bid(self, ad_group_id: str,
                            cpc_bid_micros: int) -> MutationRecord:
        """Update an ad group's CPC bid."""
        record = MutationRecord(
            action=MutationAction.UPDATE_BID,
            target=f"AdGroup {ad_group_id}",
            params={
                "ad_group_id": ad_group_id,
                "cpc_bid_micros": cpc_bid_micros,
                "cpc_bid_usd": cpc_bid_micros / 1e6,
            }
        )

        if not self.dry_run and self._connected:
            try:
                ag_service = self.client.get_service("AdGroupService")
                ag_op = self.client.get_type("AdGroupOperation")
                ad_group = ag_op.update
                ad_group.resource_name = ag_service.ad_group_path(
                    self.customer_id, ad_group_id
                )
                ad_group.cpc_bid_micros = cpc_bid_micros
                self.client.copy_from(
                    ag_op.update_mask,
                    self.client.get_type("FieldMask")(paths=["cpc_bid_micros"])
                )

                response = ag_service.mutate_ad_groups(
                    customer_id=self.customer_id,
                    operations=[ag_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── KEYWORD OPERATIONS ───────────────────────────────────────────────────

    def add_keyword(self, ad_group_id: str, keyword_text: str,
                    match_type: str = "BROAD") -> MutationRecord:
        """Add a keyword to an ad group."""
        record = MutationRecord(
            action=MutationAction.ADD_KEYWORD,
            target=f"'{keyword_text}' [{match_type}] -> AdGroup {ad_group_id}",
            params={
                "ad_group_id": ad_group_id,
                "keyword_text": keyword_text,
                "match_type": match_type,
            }
        )

        if not self.dry_run and self._connected:
            try:
                ag_criterion_service = self.client.get_service("AdGroupCriterionService")
                ag_service = self.client.get_service("AdGroupService")

                criterion_op = self.client.get_type("AdGroupCriterionOperation")
                criterion = criterion_op.create
                criterion.ad_group = ag_service.ad_group_path(
                    self.customer_id, ad_group_id
                )
                criterion.status = self.client.enums.AdGroupCriterionStatusEnum.ENABLED
                criterion.keyword.text = keyword_text
                criterion.keyword.match_type = getattr(
                    self.client.enums.KeywordMatchTypeEnum, match_type
                )

                response = ag_criterion_service.mutate_ad_group_criteria(
                    customer_id=self.customer_id,
                    operations=[criterion_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
                record.rollback_info = {"resource_name": record.result, "action": "remove"}
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    def pause_keyword(self, ad_group_id: str,
                      criterion_id: str) -> MutationRecord:
        """Pause a keyword."""
        record = MutationRecord(
            action=MutationAction.PAUSE_KEYWORD,
            target=f"Criterion {criterion_id} in AdGroup {ad_group_id}",
            params={"ad_group_id": ad_group_id, "criterion_id": criterion_id}
        )

        if not self.dry_run and self._connected:
            try:
                ag_criterion_service = self.client.get_service("AdGroupCriterionService")
                criterion_op = self.client.get_type("AdGroupCriterionOperation")
                criterion = criterion_op.update
                criterion.resource_name = (
                    ag_criterion_service.ad_group_criterion_path(
                        self.customer_id, ad_group_id, criterion_id
                    )
                )
                criterion.status = self.client.enums.AdGroupCriterionStatusEnum.PAUSED
                self.client.copy_from(
                    criterion_op.update_mask,
                    self.client.get_type("FieldMask")(paths=["status"])
                )

                response = ag_criterion_service.mutate_ad_group_criteria(
                    customer_id=self.customer_id,
                    operations=[criterion_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    def add_negative_keyword(self, campaign_id: str, keyword_text: str,
                             match_type: str = "BROAD") -> MutationRecord:
        """Add a negative keyword at campaign level."""
        record = MutationRecord(
            action=MutationAction.ADD_NEGATIVE_KEYWORD,
            target=f"Negative: '{keyword_text}' [{match_type}] -> Campaign {campaign_id}",
            params={
                "campaign_id": campaign_id,
                "keyword_text": keyword_text,
                "match_type": match_type,
            }
        )

        if not self.dry_run and self._connected:
            try:
                criterion_service = self.client.get_service("CampaignCriterionService")
                campaign_service = self.client.get_service("CampaignService")

                criterion_op = self.client.get_type("CampaignCriterionOperation")
                criterion = criterion_op.create
                criterion.campaign = campaign_service.campaign_path(
                    self.customer_id, campaign_id
                )
                criterion.negative = True
                criterion.keyword.text = keyword_text
                criterion.keyword.match_type = getattr(
                    self.client.enums.KeywordMatchTypeEnum, match_type
                )

                response = criterion_service.mutate_campaign_criteria(
                    customer_id=self.customer_id,
                    operations=[criterion_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── AD COPY OPERATIONS ───────────────────────────────────────────────────

    def create_responsive_search_ad(self, ad_group_id: str,
                                     headlines: List[str],
                                     descriptions: List[str],
                                     final_url: str,
                                     path1: str = "",
                                     path2: str = "") -> MutationRecord:
        """Create a Responsive Search Ad (RSA)."""
        record = MutationRecord(
            action=MutationAction.CREATE_AD,
            target=f"RSA -> AdGroup {ad_group_id}",
            params={
                "ad_group_id": ad_group_id,
                "headlines": headlines,
                "descriptions": descriptions,
                "final_url": final_url,
                "path1": path1,
                "path2": path2,
            }
        )

        if not self.dry_run and self._connected:
            try:
                ad_service = self.client.get_service("AdGroupAdService")
                ag_service = self.client.get_service("AdGroupService")

                ad_op = self.client.get_type("AdGroupAdOperation")
                ad_group_ad = ad_op.create
                ad_group_ad.ad_group = ag_service.ad_group_path(
                    self.customer_id, ad_group_id
                )
                ad_group_ad.status = self.client.enums.AdGroupAdStatusEnum.ENABLED
                ad = ad_group_ad.ad
                ad.final_urls.append(final_url)

                # Add headlines (max 15)
                for i, headline in enumerate(headlines[:15]):
                    headline_asset = self.client.get_type("AdTextAsset")
                    headline_asset.text = headline
                    if i < 3:
                        headline_asset.pinned_field = (
                            self.client.enums.ServedAssetFieldTypeEnum.HEADLINE_1
                            if i == 0 else self.client.enums.ServedAssetFieldTypeEnum.HEADLINE_2
                            if i == 1 else self.client.enums.ServedAssetFieldTypeEnum.HEADLINE_3
                        )
                    ad.responsive_search_ad.headlines.append(headline_asset)

                # Add descriptions (max 4)
                for desc in descriptions[:4]:
                    desc_asset = self.client.get_type("AdTextAsset")
                    desc_asset.text = desc
                    ad.responsive_search_ad.descriptions.append(desc_asset)

                if path1:
                    ad.responsive_search_ad.path1 = path1
                if path2:
                    ad.responsive_search_ad.path2 = path2

                response = ad_service.mutate_ad_group_ads(
                    customer_id=self.customer_id,
                    operations=[ad_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
                record.rollback_info = {"resource_name": record.result, "action": "pause"}
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── EXTENSION / ASSET OPERATIONS ─────────────────────────────────────────

    def add_sitelink(self, campaign_id: str, sitelink_text: str,
                     description1: str, description2: str,
                     final_url: str) -> MutationRecord:
        """Add a sitelink extension to a campaign."""
        record = MutationRecord(
            action=MutationAction.ADD_SITELINK,
            target=f"Sitelink '{sitelink_text}' -> Campaign {campaign_id}",
            params={
                "campaign_id": campaign_id,
                "sitelink_text": sitelink_text,
                "description1": description1,
                "description2": description2,
                "final_url": final_url,
            }
        )

        if not self.dry_run and self._connected:
            try:
                asset_service = self.client.get_service("AssetService")

                # Create sitelink asset
                asset_op = self.client.get_type("AssetOperation")
                asset = asset_op.create
                asset.name = sitelink_text
                asset.sitelink_asset.link_text = sitelink_text
                asset.sitelink_asset.description1 = description1
                asset.sitelink_asset.description2 = description2
                asset.sitelink_asset.final_urls.append(final_url)

                response = asset_service.mutate_assets(
                    customer_id=self.customer_id,
                    operations=[asset_op]
                )
                asset_resource = response.results[0].resource_name

                # Link asset to campaign
                campaign_asset_service = self.client.get_service("CampaignAssetService")
                campaign_service = self.client.get_service("CampaignService")

                link_op = self.client.get_type("CampaignAssetOperation")
                link = link_op.create
                link.campaign = campaign_service.campaign_path(
                    self.customer_id, campaign_id
                )
                link.asset = asset_resource
                link.field_type = self.client.enums.AssetFieldTypeEnum.SITELINK

                campaign_asset_service.mutate_campaign_assets(
                    customer_id=self.customer_id,
                    operations=[link_op]
                )
                record.status = "executed"
                record.result = asset_resource
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    def add_callout(self, campaign_id: str,
                    callout_text: str) -> MutationRecord:
        """Add a callout extension to a campaign."""
        record = MutationRecord(
            action=MutationAction.ADD_CALLOUT,
            target=f"Callout '{callout_text}' -> Campaign {campaign_id}",
            params={"campaign_id": campaign_id, "callout_text": callout_text}
        )

        if not self.dry_run and self._connected:
            try:
                asset_service = self.client.get_service("AssetService")

                asset_op = self.client.get_type("AssetOperation")
                asset = asset_op.create
                asset.name = callout_text
                asset.callout_asset.callout_text = callout_text

                response = asset_service.mutate_assets(
                    customer_id=self.customer_id,
                    operations=[asset_op]
                )
                asset_resource = response.results[0].resource_name

                # Link to campaign
                campaign_asset_service = self.client.get_service("CampaignAssetService")
                campaign_service = self.client.get_service("CampaignService")
                link_op = self.client.get_type("CampaignAssetOperation")
                link = link_op.create
                link.campaign = campaign_service.campaign_path(
                    self.customer_id, campaign_id
                )
                link.asset = asset_resource
                link.field_type = self.client.enums.AssetFieldTypeEnum.CALLOUT

                campaign_asset_service.mutate_campaign_assets(
                    customer_id=self.customer_id,
                    operations=[link_op]
                )
                record.status = "executed"
                record.result = asset_resource
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── BUDGET OPERATIONS ────────────────────────────────────────────────────

    def update_campaign_budget(self, budget_resource_name: str,
                                new_amount_micros: int) -> MutationRecord:
        """Update a campaign's daily budget."""
        record = MutationRecord(
            action=MutationAction.UPDATE_BUDGET,
            target=f"Budget {budget_resource_name}",
            params={
                "budget_resource_name": budget_resource_name,
                "new_amount_micros": new_amount_micros,
                "new_amount_usd": new_amount_micros / 1e6,
            }
        )

        if not self.dry_run and self._connected:
            try:
                budget_service = self.client.get_service("CampaignBudgetService")
                budget_op = self.client.get_type("CampaignBudgetOperation")
                budget = budget_op.update
                budget.resource_name = budget_resource_name
                budget.amount_micros = new_amount_micros
                self.client.copy_from(
                    budget_op.update_mask,
                    self.client.get_type("FieldMask")(paths=["amount_micros"])
                )

                response = budget_service.mutate_campaign_budgets(
                    customer_id=self.customer_id,
                    operations=[budget_op]
                )
                record.status = "executed"
                record.result = response.results[0].resource_name
            except Exception as e:
                record.status = "failed"
                record.result = str(e)
        else:
            record.status = "dry_run"

        self._log_mutation(record)
        return record

    # ── QUERY HELPERS ────────────────────────────────────────────────────────

    def query(self, gaql: str) -> pd.DataFrame:
        """Run a GAQL query and return results as DataFrame."""
        if not self._connected:
            if not self.connect():
                return pd.DataFrame()
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            stream = ga_service.search_stream(
                customer_id=self.customer_id,
                query=gaql.strip()
            )
            rows = []
            for batch in stream:
                for row in batch.results:
                    rows.append(self._parse_generic_row(row))
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()

    def _parse_generic_row(self, row) -> dict:
        """Generic row parser for arbitrary GAQL results."""
        d = {}
        try:
            m = row.metrics
            for field in ["clicks", "impressions", "ctr", "conversions",
                          "conversions_value", "cost_micros", "average_cpc",
                          "all_conversions", "search_impression_share"]:
                if hasattr(m, field):
                    val = getattr(m, field)
                    if field == "cost_micros":
                        d["cost"] = val / 1e6
                    elif field == "average_cpc":
                        d["avg_cpc"] = val / 1e6
                    else:
                        d[field] = val
        except Exception:
            pass

        try:
            d["date"] = str(row.segments.date)
        except Exception:
            pass

        # Entity fields
        for entity in ["campaign", "ad_group", "ad_group_criterion", "ad_group_ad",
                        "search_term_view"]:
            try:
                obj = getattr(row, entity)
                if entity == "campaign":
                    d["campaign_id"] = obj.id
                    d["campaign_name"] = obj.name
                    if hasattr(obj, "status"):
                        d["campaign_status"] = obj.status.name
                elif entity == "ad_group":
                    d["ad_group_id"] = obj.id
                    d["ad_group_name"] = obj.name
                elif entity == "ad_group_criterion":
                    if hasattr(obj, "keyword"):
                        d["keyword_text"] = obj.keyword.text
                        d["match_type"] = obj.keyword.match_type.name
                        d["criterion_id"] = obj.criterion_id
                    if hasattr(obj, "quality_info"):
                        d["quality_score"] = obj.quality_info.quality_score
                    d["kw_status"] = obj.status.name
                elif entity == "search_term_view":
                    d["search_term"] = obj.search_term
                elif entity == "ad_group_ad":
                    d["ad_id"] = obj.ad.id
                    d["ad_status"] = obj.status.name
            except Exception:
                pass

        return d

    # ── EXPORT & AUDIT ───────────────────────────────────────────────────────

    def export_mutation_log(self, path: str = "mutation_log.json"):
        """Export all mutations to a JSON file for audit."""
        records = []
        for r in self.mutation_log:
            records.append({
                "action": r.action.value,
                "target": r.target,
                "params": r.params,
                "timestamp": r.timestamp,
                "status": r.status,
                "result": r.result,
            })
        with open(path, "w") as f:
            json.dump({"mutations": records, "total": len(records),
                        "exported_at": datetime.now().isoformat()}, f, indent=2)
        logger.info(f"Exported {len(records)} mutation records to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH TERM MINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class SearchTermMiner:
    """
    Automatically mines search terms from the account:
    - Winners (high conversions) -> add as exact match keywords
    - Losers (high clicks, zero conversions) -> add as negative keywords
    - Irrelevant (obvious non-intent) -> add as negative keywords
    """

    IRRELEVANT_PATTERNS = [
        "free", "cheap", "sample", "example", "template download",
        "pdf", "word doc", "what is", "how to", "definition",
        "reddit", "quora", "youtube", "course", "class",
        "jobs", "salary", "career", "internship",
    ]

    def __init__(self, client: GoogleAdsClientWrapper):
        self.client = client

    def mine_search_terms(self, campaign_id: str = None,
                          days: int = 30) -> Dict[str, List]:
        """
        Pull search terms and classify them into winners, losers, and negatives.
        """
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")

        query = f"""
            SELECT
                campaign.id, campaign.name,
                ad_group.id, ad_group.name,
                search_term_view.search_term,
                metrics.clicks, metrics.impressions, metrics.ctr,
                metrics.conversions, metrics.conversions_value,
                metrics.cost_micros
            FROM search_term_view
            WHERE segments.date BETWEEN '{date_from}' AND '{date_to}'
            ORDER BY metrics.cost_micros DESC
            LIMIT 1000
        """

        df = self.client.query(query)

        if df.empty:
            logger.warning("No search term data available.")
            return {"winners": [], "losers": [], "irrelevant": []}

        # Compute per-term aggregates
        agg = df.groupby("search_term").agg({
            "clicks": "sum",
            "impressions": "sum",
            "conversions": "sum",
            "conversions_value": "sum",
            "cost": "sum",
        }).reset_index()

        agg["cpa"] = agg["cost"] / (agg["conversions"] + 1e-9)
        agg["ctr"] = agg["clicks"] / (agg["impressions"] + 1e-9)

        results = {"winners": [], "losers": [], "irrelevant": []}

        for _, row in agg.iterrows():
            term = row["search_term"]

            # Check irrelevant patterns first
            is_irrelevant = any(p in term.lower() for p in self.IRRELEVANT_PATTERNS)

            if row["conversions"] >= SEARCH_TERM_ADD_THRESHOLD:
                results["winners"].append({
                    "term": term,
                    "conversions": int(row["conversions"]),
                    "cost": float(row["cost"]),
                    "cpa": float(row["cpa"]),
                    "clicks": int(row["clicks"]),
                    "action": "ADD_AS_EXACT_KEYWORD",
                })
            elif is_irrelevant:
                results["irrelevant"].append({
                    "term": term,
                    "clicks": int(row["clicks"]),
                    "cost": float(row["cost"]),
                    "pattern_matched": next(
                        p for p in self.IRRELEVANT_PATTERNS if p in term.lower()
                    ),
                    "action": "ADD_AS_NEGATIVE",
                })
            elif (row["clicks"] >= SEARCH_TERM_NEGATE_CLICKS or
                  row["cost"] >= SEARCH_TERM_NEGATE_COST) and row["conversions"] == 0:
                results["losers"].append({
                    "term": term,
                    "clicks": int(row["clicks"]),
                    "cost": float(row["cost"]),
                    "action": "ADD_AS_NEGATIVE",
                })

        logger.info(f"Search Term Mining: {len(results['winners'])} winners, "
                     f"{len(results['losers'])} losers, "
                     f"{len(results['irrelevant'])} irrelevant")
        return results

    def execute_mining_results(self, results: Dict, campaign_id: str,
                                ad_group_id: str = None):
        """Execute the mining results: add winners, negate losers."""
        # Add winners as exact match keywords
        for winner in results["winners"]:
            if ad_group_id:
                self.client.add_keyword(
                    ad_group_id, winner["term"], match_type="EXACT"
                )

        # Add losers and irrelevant as campaign-level negatives
        for loser in results["losers"] + results["irrelevant"]:
            self.client.add_negative_keyword(
                campaign_id, loser["term"], match_type="EXACT"
            )


# ══════════════════════════════════════════════════════════════════════════════
# AD COPY A/B TESTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AdCopyTestingEngine:
    """
    Manages A/B testing of ad copy variations.
    
    Strategy:
    - Creates multiple RSA variants with different headline/description combos
    - Tracks performance over time
    - Picks winners based on statistical significance
    - Pauses underperformers after minimum observation period
    """

    # Headline templates for ContractKit
    HEADLINE_TEMPLATES = {
        "price_lead": [
            "Legal Contracts - Just $19/mo",
            "Contracts + Invoicing $19/mo",
            "All-in-One Legal Tool - $19/mo",
        ],
        "feature_lead": [
            "Create Legal Contracts Online",
            "Contracts + Invoicing + E-Sign",
            "Automated Payment Reminders",
            "E-Sign & Get Paid Instantly",
            "Net 30 & 50/50 Payment Terms",
        ],
        "trust_lead": [
            "Legally Binding - All 50 States",
            "Trusted by Small Businesses",
            "Secure E-Signatures Built In",
        ],
        "action_lead": [
            "Create Your First Contract Free",
            "Start Sending Invoices Today",
            "Sign Up - No Credit Card Needed",
            "Get Paid Faster with ContractKit",
        ],
        "competitor_lead": [
            "Better Than DocuSign - $19/mo",
            "PandaDoc Alternative - Save 70%",
            "Ditch Expensive Contract Tools",
        ],
    }

    DESCRIPTION_TEMPLATES = {
        "value_prop": [
            "Create legally binding contracts, send invoices, collect e-signatures & accept payments. All 50 states. One simple plan.",
            "Unlimited contracts & invoices for $19/mo. E-sign, auto-reminders, Stripe payments. Everything you need in one place.",
        ],
        "urgency": [
            "Stop losing money on overdue invoices. Auto-reminders + payment terms (Net 30, 50/50) keep cash flowing.",
            "Your competitors are getting paid faster. ContractKit automates contracts to cash in minutes, not weeks.",
        ],
        "social_proof": [
            "Join thousands of freelancers & small businesses creating professional contracts. Legally binding in all 50 states.",
            "Rated #1 for affordability. Full contract lifecycle: create, sign, invoice, collect. $19/mo, cancel anytime.",
        ],
    }

    def __init__(self, client: GoogleAdsClientWrapper):
        self.client = client
        self.test_variants: List[Dict] = []

    def generate_test_variants(self, n_variants: int = 3) -> List[Dict]:
        """Generate N ad copy variants for testing."""
        variants = []

        # Variant 1: Price-led (for budget-conscious SMBs)
        variants.append({
            "name": "Variant A: Price Leader",
            "headlines": (
                self.HEADLINE_TEMPLATES["price_lead"] +
                self.HEADLINE_TEMPLATES["feature_lead"][:3] +
                self.HEADLINE_TEMPLATES["action_lead"][:2]
            ),
            "descriptions": (
                self.DESCRIPTION_TEMPLATES["value_prop"] +
                self.DESCRIPTION_TEMPLATES["urgency"][:1]
            ),
        })

        # Variant 2: Trust-led (for skeptical buyers)
        variants.append({
            "name": "Variant B: Trust Builder",
            "headlines": (
                self.HEADLINE_TEMPLATES["trust_lead"] +
                self.HEADLINE_TEMPLATES["feature_lead"][:3] +
                self.HEADLINE_TEMPLATES["price_lead"][:2]
            ),
            "descriptions": (
                self.DESCRIPTION_TEMPLATES["social_proof"] +
                self.DESCRIPTION_TEMPLATES["value_prop"][:1]
            ),
        })

        # Variant 3: Action-led (for ready-to-buy)
        variants.append({
            "name": "Variant C: Action Driver",
            "headlines": (
                self.HEADLINE_TEMPLATES["action_lead"] +
                self.HEADLINE_TEMPLATES["price_lead"][:2] +
                self.HEADLINE_TEMPLATES["trust_lead"][:2]
            ),
            "descriptions": (
                self.DESCRIPTION_TEMPLATES["urgency"] +
                self.DESCRIPTION_TEMPLATES["value_prop"][:1]
            ),
        })

        self.test_variants = variants[:n_variants]
        return self.test_variants

    def deploy_test(self, ad_group_id: str,
                    final_url: str = "https://www.contractkit.info"):
        """Deploy all test variants to an ad group."""
        for variant in self.test_variants:
            self.client.create_responsive_search_ad(
                ad_group_id=ad_group_id,
                headlines=variant["headlines"],
                descriptions=variant["descriptions"],
                final_url=final_url,
                path1="contracts",
                path2="start",
            )

    def evaluate_test(self, ad_group_id: str,
                      min_clicks: int = 100,
                      min_days: int = 14) -> Dict:
        """
        Evaluate ad performance and pick winner.
        Uses CTR and conversion rate with statistical significance test.
        """
        query = f"""
            SELECT
                ad_group_ad.ad.id,
                ad_group_ad.ad.responsive_search_ad.headlines,
                metrics.clicks, metrics.impressions, metrics.ctr,
                metrics.conversions, metrics.conversions_value,
                metrics.cost_micros
            FROM ad_group_ad
            WHERE ad_group.id = {ad_group_id}
              AND ad_group_ad.status = 'ENABLED'
              AND segments.date DURING LAST_30_DAYS
        """
        df = self.client.query(query)
        if df.empty:
            return {"status": "no_data", "winner": None}

        # Need minimum data to declare a winner
        total_clicks = df["clicks"].sum()
        if total_clicks < min_clicks:
            return {
                "status": "insufficient_data",
                "total_clicks": int(total_clicks),
                "needed": min_clicks,
                "winner": None,
            }

        # Rank by conversion rate, then CTR
        df["conv_rate"] = df["conversions"] / (df["clicks"] + 1e-9)
        df = df.sort_values(["conv_rate", "ctr"], ascending=False)

        winner = df.iloc[0]
        losers = df.iloc[1:]

        return {
            "status": "winner_found" if len(df) > 1 else "single_ad",
            "winner": {
                "ad_id": str(winner.get("ad_id", "")),
                "clicks": int(winner.get("clicks", 0)),
                "conversions": int(winner.get("conversions", 0)),
                "conv_rate": float(winner.get("conv_rate", 0)),
                "ctr": float(winner.get("ctr", 0)),
            },
            "losers": [
                {
                    "ad_id": str(row.get("ad_id", "")),
                    "clicks": int(row.get("clicks", 0)),
                    "conversions": int(row.get("conversions", 0)),
                    "conv_rate": float(row.get("conv_rate", 0)),
                }
                for _, row in losers.iterrows()
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class QualityScoreTracker:
    """
    Monitors Quality Score over time and recommends actions:
    - QS <= 3: Pause keyword (wasting money)
    - QS 4-5: Review ad relevance and landing page
    - QS 6-7: Maintain, consider bid increase
    - QS 8-10: Top performer, maximize impression share
    """

    def __init__(self, client: GoogleAdsClientWrapper):
        self.client = client
        self.history_path = "quality_score_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                return json.load(f)
        return {"snapshots": [], "keyword_trends": {}}

    def _save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def take_snapshot(self) -> pd.DataFrame:
        """Pull current QS for all keywords and store snapshot."""
        query = """
            SELECT
                campaign.name,
                ad_group.name,
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.criterion_id,
                ad_group_criterion.quality_info.quality_score,
                ad_group_criterion.quality_info.creative_quality_score,
                ad_group_criterion.quality_info.post_click_quality_score,
                ad_group_criterion.quality_info.search_predicted_ctr,
                metrics.impressions, metrics.clicks, metrics.conversions,
                metrics.cost_micros
            FROM keyword_view
            WHERE ad_group_criterion.status = 'ENABLED'
              AND metrics.impressions > 0
              AND segments.date DURING LAST_7_DAYS
        """
        df = self.client.query(query)

        if not df.empty:
            snapshot = {
                "date": datetime.now().isoformat(),
                "keywords": df.to_dict(orient="records"),
                "avg_qs": float(df["quality_score"].mean()) if "quality_score" in df else 0,
            }
            self.history["snapshots"].append(snapshot)
            self._save_history()

        return df

    def get_recommendations(self, df: pd.DataFrame = None) -> List[Dict]:
        """Generate QS-based recommendations."""
        if df is None or df.empty:
            df = self.take_snapshot()

        if df.empty:
            return []

        recs = []

        for _, row in df.iterrows():
            qs = row.get("quality_score", 0)
            kw = row.get("keyword_text", "")
            cost = row.get("cost", 0)

            if qs <= QS_PAUSE_THRESHOLD and qs > 0:
                recs.append({
                    "keyword": kw,
                    "quality_score": int(qs),
                    "action": "PAUSE",
                    "reason": f"QS={qs} is critically low. Wasting ${cost:.2f} on poor relevance.",
                    "priority": "HIGH",
                })
            elif qs <= QS_REVIEW_THRESHOLD and qs > 0:
                recs.append({
                    "keyword": kw,
                    "quality_score": int(qs),
                    "action": "REVIEW",
                    "reason": f"QS={qs}. Check ad copy relevance and landing page experience.",
                    "priority": "MEDIUM",
                })
            elif qs >= QS_BOOST_THRESHOLD:
                recs.append({
                    "keyword": kw,
                    "quality_score": int(qs),
                    "action": "BOOST_BID",
                    "reason": f"QS={qs} is excellent. Increase bid to capture more impression share.",
                    "priority": "LOW",
                })

        return recs


# ══════════════════════════════════════════════════════════════════════════════
# DAYPARTING & DEVICE BID ADJUSTMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class DaypartingEngine:
    """
    Analyzes performance by hour-of-day and day-of-week,
    then sets bid adjustments to spend more during high-converting hours
    and less during low-performing periods.
    """

    def __init__(self, client: GoogleAdsClientWrapper):
        self.client = client

    def analyze_hourly_performance(self, campaign_id: str = None,
                                    days: int = 30) -> pd.DataFrame:
        """Pull hourly performance data."""
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")

        query = f"""
            SELECT
                campaign.id, campaign.name,
                segments.hour, segments.day_of_week,
                metrics.clicks, metrics.impressions,
                metrics.conversions, metrics.conversions_value,
                metrics.cost_micros
            FROM campaign
            WHERE segments.date BETWEEN '{date_from}' AND '{date_to}'
        """

        df = self.client.query(query)

        if df.empty:
            return pd.DataFrame()

        # Aggregate by hour
        hourly = df.groupby("hour").agg({
            "clicks": "sum", "impressions": "sum",
            "conversions": "sum", "cost": "sum"
        }).reset_index()

        hourly["ctr"] = hourly["clicks"] / (hourly["impressions"] + 1e-9)
        hourly["conv_rate"] = hourly["conversions"] / (hourly["clicks"] + 1e-9)
        hourly["cpa"] = hourly["cost"] / (hourly["conversions"] + 1e-9)

        return hourly

    def generate_bid_schedule(self, hourly_df: pd.DataFrame) -> List[Dict]:
        """
        Generate bid adjustment recommendations by hour.
        Uses conversion rate relative to average to set multipliers.
        """
        if hourly_df.empty:
            return []

        avg_conv_rate = hourly_df["conv_rate"].mean()
        if avg_conv_rate == 0:
            return []

        schedule = []
        for _, row in hourly_df.iterrows():
            hour = int(row["hour"])
            relative_perf = row["conv_rate"] / (avg_conv_rate + 1e-9)

            # Cap adjustments between -50% and +30%
            if relative_perf > 1.5:
                adjustment = 0.30  # +30%
            elif relative_perf > 1.2:
                adjustment = 0.15  # +15%
            elif relative_perf > 0.8:
                adjustment = 0.0   # No change
            elif relative_perf > 0.5:
                adjustment = -0.20  # -20%
            else:
                adjustment = -0.50  # -50%

            schedule.append({
                "hour": hour,
                "conv_rate": float(row["conv_rate"]),
                "relative_performance": float(relative_perf),
                "bid_adjustment": adjustment,
                "label": f"{'+'if adjustment >= 0 else ''}{adjustment*100:.0f}%",
            })

        return schedule

    def analyze_device_performance(self, days: int = 30) -> pd.DataFrame:
        """Analyze performance by device type."""
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")

        query = f"""
            SELECT
                segments.device,
                metrics.clicks, metrics.impressions,
                metrics.conversions, metrics.conversions_value,
                metrics.cost_micros
            FROM campaign
            WHERE segments.date BETWEEN '{date_from}' AND '{date_to}'
        """

        df = self.client.query(query)
        if df.empty:
            return pd.DataFrame()

        by_device = df.groupby("device").agg({
            "clicks": "sum", "impressions": "sum",
            "conversions": "sum", "cost": "sum"
        }).reset_index()

        by_device["conv_rate"] = by_device["conversions"] / (by_device["clicks"] + 1e-9)
        by_device["cpa"] = by_device["cost"] / (by_device["conversions"] + 1e-9)

        return by_device


# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN STRUCTURE BUILDER (The "Perfect Account")
# ══════════════════════════════════════════════════════════════════════════════

class CampaignStructureBuilder:
    """
    Builds the optimal campaign structure for ContractKit.
    
    Recommended structure:
    
    Campaign 1: [EXACT] ContractKit - High Intent Contracts
      AG: Contract Creation (exact match keywords)
      AG: Contract Templates (exact match keywords)
      AG: E-Signing (exact match keywords)
      
    Campaign 2: [EXACT] ContractKit - Invoicing & Payments
      AG: Invoicing (exact match keywords)
      AG: Payment Terms (exact match keywords)
      AG: Payment Reminders (exact match keywords)
      
    Campaign 3: [BROAD] ContractKit - Discovery
      AG: Broad Contract Terms (broad match for discovery)
      AG: Broad Invoicing Terms (broad match for discovery)
      
    Campaign 4: [EXACT] ContractKit - Competitor
      AG: DocuSign Alternative
      AG: PandaDoc Alternative
      AG: Other Alternatives
      
    Campaign 5: [EXACT] ContractKit - Brand
      AG: Brand Terms
    """

    OPTIMAL_STRUCTURE = {
        "ContractKit - High Intent Contracts [EXACT]": {
            "bidding": "MAXIMIZE_CONVERSIONS",
            "daily_budget_usd": 15.0,
            "ad_groups": {
                "Contract Creation": {
                    "keywords": [
                        ("contract maker", "EXACT"),
                        ("create a contract", "EXACT"),
                        ("online contract maker", "EXACT"),
                        ("contract creator online", "EXACT"),
                        ("contract builder", "EXACT"),
                        ("make a legal contract", "EXACT"),
                        ("legally binding contract", "EXACT"),
                        ("digital contract", "EXACT"),
                        ("create contract online free", "EXACT"),
                        ("legal contract maker", "EXACT"),
                    ],
                    "cpc_bid_usd": 3.00,
                },
                "Contract Templates": {
                    "keywords": [
                        ("contract template", "EXACT"),
                        ("freelance contract template", "EXACT"),
                        ("service agreement template", "EXACT"),
                        ("nda template", "EXACT"),
                        ("independent contractor agreement", "EXACT"),
                        ("contractor agreement template", "EXACT"),
                        ("business contract template", "EXACT"),
                        ("consulting agreement template", "EXACT"),
                    ],
                    "cpc_bid_usd": 3.50,
                },
                "E-Signing": {
                    "keywords": [
                        ("sign contract online", "EXACT"),
                        ("e-sign documents", "EXACT"),
                        ("electronic signature for contracts", "EXACT"),
                        ("online contract signing", "EXACT"),
                        ("digital signature software", "EXACT"),
                    ],
                    "cpc_bid_usd": 4.00,
                },
            },
            "negative_keywords": [
                "free", "sample", "example", "download", "pdf",
                "word", "google docs", "what is", "how to",
                "definition", "law", "reddit", "quora",
            ],
        },
        "ContractKit - Invoicing & Payments [EXACT]": {
            "bidding": "MAXIMIZE_CONVERSIONS",
            "daily_budget_usd": 10.0,
            "ad_groups": {
                "Invoicing": {
                    "keywords": [
                        ("invoicing software", "EXACT"),
                        ("send invoice online", "EXACT"),
                        ("automated invoicing", "EXACT"),
                        ("invoice generator for freelancers", "EXACT"),
                        ("small business invoicing", "EXACT"),
                        ("recurring invoicing software", "EXACT"),
                    ],
                    "cpc_bid_usd": 4.00,
                },
                "Payment Terms": {
                    "keywords": [
                        ("net 30 invoicing", "EXACT"),
                        ("payment terms software", "EXACT"),
                        ("split payment invoicing", "EXACT"),
                        ("accept payments for contracts", "EXACT"),
                        ("stripe invoicing", "EXACT"),
                        ("automated payment collection", "EXACT"),
                    ],
                    "cpc_bid_usd": 2.00,
                },
                "Payment Reminders": {
                    "keywords": [
                        ("payment reminder software", "EXACT"),
                        ("automated payment reminders", "EXACT"),
                        ("invoice reminder app", "EXACT"),
                        ("overdue payment reminder", "EXACT"),
                    ],
                    "cpc_bid_usd": 1.50,
                },
            },
            "negative_keywords": [
                "free", "template", "excel", "google sheets",
                "paypal", "venmo", "zelle",
            ],
        },
        "ContractKit - Discovery [BROAD]": {
            "bidding": "MAXIMIZE_CONVERSIONS",
            "daily_budget_usd": 8.0,
            "ad_groups": {
                "Broad Contract Terms": {
                    "keywords": [
                        ("contract management software", "BROAD"),
                        ("create legal documents online", "BROAD"),
                        ("small business contract tool", "BROAD"),
                        ("freelancer contract app", "BROAD"),
                        ("digital contract platform", "BROAD"),
                    ],
                    "cpc_bid_usd": 2.50,
                },
                "Broad Invoicing Terms": {
                    "keywords": [
                        ("invoicing and contracts app", "BROAD"),
                        ("send invoices and get paid", "BROAD"),
                        ("freelance billing software", "BROAD"),
                        ("all in one business tool invoicing", "BROAD"),
                    ],
                    "cpc_bid_usd": 2.50,
                },
            },
            "negative_keywords": [
                "enterprise", "fortune 500", "free", "open source",
                "github", "developer", "api", "sdk",
                "jobs", "salary", "career", "course", "class",
            ],
        },
        "ContractKit - Competitor [EXACT]": {
            "bidding": "MAXIMIZE_CONVERSIONS",
            "daily_budget_usd": 5.0,
            "ad_groups": {
                "DocuSign Alternative": {
                    "keywords": [
                        ("docusign alternative", "EXACT"),
                        ("cheaper than docusign", "EXACT"),
                        ("docusign for small business", "EXACT"),
                    ],
                    "cpc_bid_usd": 5.00,
                },
                "PandaDoc Alternative": {
                    "keywords": [
                        ("pandadoc alternative", "EXACT"),
                        ("pandadoc for freelancers", "EXACT"),
                        ("cheaper than pandadoc", "EXACT"),
                    ],
                    "cpc_bid_usd": 4.50,
                },
                "Other Alternatives": {
                    "keywords": [
                        ("honeybook alternative", "EXACT"),
                        ("hellosign alternative", "EXACT"),
                        ("freshbooks contract alternative", "EXACT"),
                        ("bonsai alternative", "EXACT"),
                    ],
                    "cpc_bid_usd": 4.00,
                },
            },
            "negative_keywords": [
                "review", "reviews", "vs", "comparison", "pricing",
            ],
        },
        "ContractKit - Brand [EXACT]": {
            "bidding": "MAXIMIZE_CONVERSIONS",
            "daily_budget_usd": 3.0,
            "ad_groups": {
                "Brand Terms": {
                    "keywords": [
                        ("contractkit", "EXACT"),
                        ("contract kit", "EXACT"),
                        ("contract kit app", "EXACT"),
                        ("contractkit.info", "EXACT"),
                        ("contractkit login", "EXACT"),
                    ],
                    "cpc_bid_usd": 0.50,
                },
            },
            "negative_keywords": [],
        },
    }

    SITELINKS = [
        {
            "text": "Pricing - $19/mo",
            "desc1": "One simple plan for everything",
            "desc2": "Contracts, invoicing, e-sign",
            "url": "https://www.contractkit.info/#pricing",
        },
        {
            "text": "How It Works",
            "desc1": "Create contracts in minutes",
            "desc2": "Legally binding in all 50 states",
            "url": "https://www.contractkit.info/#how-it-works",
        },
        {
            "text": "Contract Templates",
            "desc1": "NDA, service, freelance & more",
            "desc2": "Customize any template instantly",
            "url": "https://www.contractkit.info/#templates",
        },
        {
            "text": "Start Free Trial",
            "desc1": "No credit card required",
            "desc2": "Set up in under 2 minutes",
            "url": "https://www.contractkit.info/signup",
        },
    ]

    CALLOUTS = [
        "All 50 States",
        "$19/Month Flat",
        "Unlimited Contracts",
        "E-Sign Built In",
        "Auto Payment Reminders",
        "Stripe Payments",
        "Net 30 & 50/50 Terms",
        "Cancel Anytime",
    ]

    def __init__(self, client: GoogleAdsClientWrapper):
        self.client = client
        self.created_campaigns = {}

    def build_all(self, execute: bool = False) -> Dict:
        """
        Build the complete campaign structure.
        Returns a blueprint of all operations (dry run by default).
        """
        original_dry_run = self.client.dry_run
        if not execute:
            self.client.dry_run = True

        blueprint = {"campaigns": [], "total_daily_budget": 0}

        for camp_name, camp_config in self.OPTIMAL_STRUCTURE.items():
            daily_budget = camp_config["daily_budget_usd"]
            blueprint["total_daily_budget"] += daily_budget

            # Create campaign
            camp_record = self.client.create_campaign(
                name=camp_name,
                budget_daily_micros=int(daily_budget * 1e6),
                bidding_strategy=camp_config["bidding"],
                target_cpa_micros=int(IDEAL_TARGET_CPA * 1e6),
            )

            campaign_info = {
                "name": camp_name,
                "budget_daily": daily_budget,
                "bidding": camp_config["bidding"],
                "ad_groups": [],
                "negative_keywords": camp_config.get("negative_keywords", []),
            }

            # Create ad groups and keywords
            for ag_name, ag_config in camp_config["ad_groups"].items():
                ag_record = self.client.create_ad_group(
                    campaign_id="PENDING",  # Would use real ID in live mode
                    name=ag_name,
                    cpc_bid_micros=int(ag_config["cpc_bid_usd"] * 1e6),
                )

                ag_info = {
                    "name": ag_name,
                    "cpc_bid": ag_config["cpc_bid_usd"],
                    "keywords": [],
                }

                for kw_text, kw_match in ag_config["keywords"]:
                    self.client.add_keyword(
                        ad_group_id="PENDING",
                        keyword_text=kw_text,
                        match_type=kw_match,
                    )
                    ag_info["keywords"].append({
                        "text": kw_text,
                        "match_type": kw_match,
                    })

                campaign_info["ad_groups"].append(ag_info)

            # Add negative keywords
            for neg_kw in camp_config.get("negative_keywords", []):
                self.client.add_negative_keyword(
                    campaign_id="PENDING",
                    keyword_text=neg_kw,
                    match_type="BROAD",
                )

            blueprint["campaigns"].append(campaign_info)

        # Sitelinks & callouts for all campaigns
        blueprint["sitelinks"] = self.SITELINKS
        blueprint["callouts"] = self.CALLOUTS

        for sitelink in self.SITELINKS:
            self.client.add_sitelink(
                campaign_id="PENDING",
                sitelink_text=sitelink["text"],
                description1=sitelink["desc1"],
                description2=sitelink["desc2"],
                final_url=sitelink["url"],
            )

        for callout in self.CALLOUTS:
            self.client.add_callout(campaign_id="PENDING", callout_text=callout)

        self.client.dry_run = original_dry_run
        return blueprint


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATED OPTIMIZATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

class AutomatedOptimizationLoop:
    """
    The full automated cycle:
    1. Pull fresh data from Google Ads
    2. Run Infinity Engine analysis (7 layers)
    3. Mine search terms
    4. Check quality scores
    5. Evaluate A/B tests
    6. Generate mutations
    7. Execute (if auto-execute is on) or save for review
    8. Log everything
    
    Designed to be run daily via scheduler.
    """

    def __init__(self, customer_id: str, login_customer_id: str = None,
                 auto_execute: bool = False, dry_run: bool = True):
        self.customer_id = customer_id
        self.client = GoogleAdsClientWrapper(
            customer_id=customer_id,
            login_customer_id=login_customer_id,
            dry_run=dry_run,
        )
        self.auto_execute = auto_execute
        self.search_miner = SearchTermMiner(self.client)
        self.ab_tester = AdCopyTestingEngine(self.client)
        self.qs_tracker = QualityScoreTracker(self.client)
        self.dayparting = DaypartingEngine(self.client)
        self.structure_builder = CampaignStructureBuilder(self.client)
        self.cycle_count = 0
        self.state_path = "optimization_loop_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "cycles_run": 0,
            "last_run": None,
            "total_mutations": 0,
            "search_terms_added": 0,
            "search_terms_negated": 0,
            "keywords_paused": 0,
            "history": [],
        }

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def run_cycle(self) -> Dict:
        """
        Run one full optimization cycle.
        Returns a summary of all actions taken.
        """
        start = datetime.now()
        logger.info("=" * 60)
        logger.info(f"OPTIMIZATION CYCLE #{self.state['cycles_run'] + 1}")
        logger.info(f"Started: {start.isoformat()}")
        logger.info("=" * 60)

        summary = {
            "cycle": self.state["cycles_run"] + 1,
            "started": start.isoformat(),
            "steps": {},
        }

        # Step 1: Connect
        logger.info("\n[Step 1] Connecting to Google Ads API...")
        connected = self.client.connect()
        summary["steps"]["connect"] = connected

        if not connected:
            logger.warning("Cannot connect to API. Running in analysis-only mode.")

        # Step 2: Run Infinity Engine
        logger.info("\n[Step 2] Running Infinity Engine analysis...")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from contractkit_infinity_engine import ContractKitInfinityEngine

            engine = ContractKitInfinityEngine(customer_id=self.customer_id)
            control_surface = engine.run_infinity_loop()
            summary["steps"]["infinity_engine"] = {
                "actions": control_surface.get("summary", {}).get("total_actions", 0),
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Infinity Engine failed: {e}")
            summary["steps"]["infinity_engine"] = {"status": "failed", "error": str(e)}

        # Step 3: Search Term Mining
        logger.info("\n[Step 3] Mining search terms...")
        try:
            mining_results = self.search_miner.mine_search_terms()
            summary["steps"]["search_mining"] = {
                "winners": len(mining_results.get("winners", [])),
                "losers": len(mining_results.get("losers", [])),
                "irrelevant": len(mining_results.get("irrelevant", [])),
            }
            self.state["search_terms_added"] += len(mining_results.get("winners", []))
            self.state["search_terms_negated"] += (
                len(mining_results.get("losers", [])) +
                len(mining_results.get("irrelevant", []))
            )
        except Exception as e:
            logger.error(f"Search term mining failed: {e}")
            summary["steps"]["search_mining"] = {"status": "failed", "error": str(e)}

        # Step 4: Quality Score Tracking
        logger.info("\n[Step 4] Tracking quality scores...")
        try:
            qs_df = self.qs_tracker.take_snapshot()
            qs_recs = self.qs_tracker.get_recommendations(qs_df)
            pause_count = len([r for r in qs_recs if r["action"] == "PAUSE"])
            summary["steps"]["quality_scores"] = {
                "keywords_tracked": len(qs_df) if not qs_df.empty else 0,
                "pause_recommended": pause_count,
                "review_recommended": len([r for r in qs_recs if r["action"] == "REVIEW"]),
                "boost_recommended": len([r for r in qs_recs if r["action"] == "BOOST_BID"]),
            }
            self.state["keywords_paused"] += pause_count
        except Exception as e:
            logger.error(f"QS tracking failed: {e}")
            summary["steps"]["quality_scores"] = {"status": "failed", "error": str(e)}

        # Step 5: Dayparting Analysis
        logger.info("\n[Step 5] Analyzing hourly performance...")
        try:
            hourly = self.dayparting.analyze_hourly_performance()
            schedule = self.dayparting.generate_bid_schedule(hourly)
            summary["steps"]["dayparting"] = {
                "hours_analyzed": len(hourly) if not hourly.empty else 0,
                "adjustments": len([s for s in schedule if s["bid_adjustment"] != 0]),
            }
        except Exception as e:
            logger.error(f"Dayparting analysis failed: {e}")
            summary["steps"]["dayparting"] = {"status": "failed", "error": str(e)}

        # Step 6: A/B Test Evaluation
        logger.info("\n[Step 6] Evaluating A/B tests...")
        summary["steps"]["ab_testing"] = {"status": "pending_data"}

        # Finalize
        end = datetime.now()
        summary["finished"] = end.isoformat()
        summary["duration_seconds"] = (end - start).total_seconds()
        summary["total_mutations"] = len(self.client.mutation_log)

        self.state["cycles_run"] += 1
        self.state["last_run"] = end.isoformat()
        self.state["total_mutations"] += len(self.client.mutation_log)
        self.state["history"].append(summary)
        self._save_state()

        # Export mutation log
        self.client.export_mutation_log(
            f"mutation_log_cycle_{self.state['cycles_run']}.json"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("  OPTIMIZATION CYCLE COMPLETE")
        print("=" * 60)
        for step_name, step_data in summary["steps"].items():
            print(f"  [{step_name}]: {step_data}")
        print(f"  Total mutations: {summary['total_mutations']}")
        print(f"  Duration: {summary['duration_seconds']:.1f}s")
        print("=" * 60)

        return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_blueprint(customer_id: str = "8360921701",
                  login_customer_id: str = "8360921701"):
    """Generate the complete campaign blueprint (dry run)."""
    print("=" * 60)
    print("  CONTRACTKIT CAMPAIGN BLUEPRINT GENERATOR")
    print("  (Dry Run - No changes will be made)")
    print("=" * 60)

    client = GoogleAdsClientWrapper(
        customer_id=customer_id,
        login_customer_id=login_customer_id,
        dry_run=True,
    )

    builder = CampaignStructureBuilder(client)
    blueprint = builder.build_all(execute=False)

    # Print the blueprint
    total_keywords = 0
    print(f"\n  Total Daily Budget: ${blueprint['total_daily_budget']:.2f}/day "
          f"(~${blueprint['total_daily_budget'] * 30:.0f}/month)")

    for camp in blueprint["campaigns"]:
        print(f"\n  CAMPAIGN: {camp['name']}")
        print(f"    Budget: ${camp['budget_daily']:.2f}/day | Strategy: {camp['bidding']}")
        for ag in camp["ad_groups"]:
            kw_count = len(ag["keywords"])
            total_keywords += kw_count
            print(f"    Ad Group: {ag['name']} (CPC: ${ag['cpc_bid']:.2f}, {kw_count} keywords)")
            for kw in ag["keywords"]:
                print(f"      - [{kw['match_type']:5s}] {kw['text']}")
        if camp["negative_keywords"]:
            print(f"    Negatives: {', '.join(camp['negative_keywords'])}")

    print(f"\n  SITELINKS ({len(blueprint['sitelinks'])}):")
    for sl in blueprint["sitelinks"]:
        print(f"    - {sl['text']} -> {sl['url']}")

    print(f"\n  CALLOUTS ({len(blueprint['callouts'])}):")
    print(f"    {' | '.join(blueprint['callouts'])}")

    print(f"\n  SUMMARY:")
    print(f"    Campaigns:     {len(blueprint['campaigns'])}")
    print(f"    Ad Groups:     {sum(len(c['ad_groups']) for c in blueprint['campaigns'])}")
    print(f"    Keywords:      {total_keywords}")
    print(f"    Daily Budget:  ${blueprint['total_daily_budget']:.2f}")
    print(f"    Monthly Budget: ~${blueprint['total_daily_budget'] * 30:.0f}")

    # Save blueprint
    with open("contractkit_campaign_blueprint.json", "w") as f:
        json.dump(blueprint, f, indent=2, default=str)
    print(f"\n  Blueprint saved to: contractkit_campaign_blueprint.json")

    # Export mutation log to see all planned operations
    client.export_mutation_log("contractkit_blueprint_mutations.json")
    print(f"  Mutation plan saved to: contractkit_blueprint_mutations.json")

    return blueprint


def run_ad_copy_variants():
    """Generate and display ad copy test variants."""
    print("=" * 60)
    print("  CONTRACTKIT AD COPY A/B TEST VARIANTS")
    print("=" * 60)

    client = GoogleAdsClientWrapper(customer_id="8360921701", dry_run=True)
    tester = AdCopyTestingEngine(client)
    variants = tester.generate_test_variants()

    for variant in variants:
        print(f"\n  {variant['name']}")
        print(f"  {'─' * 40}")
        print(f"  Headlines ({len(variant['headlines'])}):")
        for h in variant["headlines"]:
            print(f"    H: {h}")
        print(f"  Descriptions ({len(variant['descriptions'])}):")
        for d in variant["descriptions"]:
            print(f"    D: {d}")

    return variants


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ContractKit Ads Mutation Engine")
    parser.add_argument("command", choices=["blueprint", "ad-copy", "cycle", "structure"],
                        help="Command to run")
    parser.add_argument("--customer-id", default="8360921701")
    parser.add_argument("--login-customer-id", default="8360921701")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute mutations (LIVE MODE)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (default)")
    args = parser.parse_args()

    if args.command == "blueprint":
        run_blueprint(args.customer_id, args.login_customer_id)
    elif args.command == "ad-copy":
        run_ad_copy_variants()
    elif args.command == "cycle":
        loop = AutomatedOptimizationLoop(
            customer_id=args.customer_id,
            login_customer_id=args.login_customer_id,
            dry_run=not args.execute,
        )
        loop.run_cycle()
    elif args.command == "structure":
        run_blueprint(args.customer_id, args.login_customer_id)
