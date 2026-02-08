# ContractKit Ads Engine - Future Roadmap

## Current System (Built)

| Module | File | Status |
|--------|------|--------|
| 7-Layer Infinity Engine | `contractkit_infinity_engine.py` | Ready |
| Live Analysis (product-specific) | `contractkit_live_analysis.py` | Ready |
| Mutation Engine (full CRUD) | `ads_mutation_engine.py` | Ready (needs Basic Access) |
| Campaign Structure Builder | `ads_mutation_engine.py` | Blueprint generated |
| Ad Copy A/B Testing | `ads_mutation_engine.py` | Variants generated |
| Search Term Mining | `ads_mutation_engine.py` | Ready |
| Quality Score Tracker | `ads_mutation_engine.py` | Ready |
| Dayparting & Device Bids | `ads_mutation_engine.py` | Ready |
| Meta-Optimizer | `meta_optimizer.py` | First cycle complete |
| Runner Script | `run_infinity_loop.py` | Ready |

## Blocker: Google Ads API Basic Access

All live features are blocked until Google approves the Developer Token for Basic Access.
Check status: Google Ads > Tools > API Center

---

## Phase 1: Immediate (Once Basic Access Approved)

### 1.1 Go Live with Mutations
- [ ] Test API connection with real customer ID
- [ ] Run Infinity Engine on real data
- [ ] Review campaign blueprint with real performance data
- [ ] Deploy optimal campaign structure (5 campaigns, 12 ad groups, 63 keywords)
- [ ] Add all sitelinks, callouts, and structured snippets
- [ ] Deploy 3 RSA ad copy variants for A/B testing

### 1.2 Conversion Tracking Audit
- [ ] Verify subscription purchase conversion is tracked
- [ ] Enable enhanced conversions
- [ ] Set conversion value to $19
- [ ] Add micro-conversions (signup started, pricing page viewed)

### 1.3 First Optimization Cycle
- [ ] Pull 7+ days of real data
- [ ] Run full 7-layer analysis
- [ ] Execute search term mining (add winners, negate losers)
- [ ] Review Quality Score recommendations
- [ ] Apply dayparting bid adjustments

---

## Phase 2: Short-Term Enhancements (Weeks 2-4)

### 2.1 Audience Expansion
- Implement Customer Match lists (upload existing customer emails)
- Create Similar Audiences from converters
- Add In-Market audiences: Business Services, Legal Services, Accounting
- Add Affinity audiences: Small Business Owners, Freelancers
- Test Remarketing campaigns for website visitors who didn't subscribe

### 2.2 Geo-Targeting Intelligence
- Analyze conversion rates by state/metro area
- Increase bids in high-converting states
- Decrease bids in low-performing areas
- Test state-specific ad copy ("Legal Contracts in Texas - $19/mo")

### 2.3 Landing Page Optimization
- Build dedicated landing pages per campaign theme:
  - /contracts for contract creation campaigns
  - /invoicing for invoicing campaigns
  - /alternative for competitor campaigns
- A/B test landing page variants
- Implement page speed optimizations

### 2.4 Responsive Search Ad Optimization
- Monitor asset performance reports (which headlines/descriptions win)
- Pin winning headlines to position 1
- Replace underperforming assets monthly
- Test long vs short headlines

---

## Phase 3: Advanced (Months 2-3)

### 3.1 Performance Max Campaign
- Create a PMax campaign alongside Search campaigns
- Supply all creative assets (images, logos, videos, headlines, descriptions)
- Let Google's ML find new converting audiences across Search, Display, YouTube, Gmail, Discover
- Monitor cannibalization with Search campaigns

### 3.2 YouTube Ads (Video)
- Create 15-30 second explainer video: "Create Legal Contracts in 60 Seconds"
- Target: Small business owners, freelancers searching for contract tools
- Use TrueView for Action format (drives subscriptions)
- Retarget website visitors with video ads

### 3.3 Display Remarketing
- Build remarketing audiences:
  - Visited site, didn't sign up (7-day, 30-day, 90-day windows)
  - Started signup, didn't complete
  - Viewed pricing page
- Create display ads for each audience with appropriate messaging
- Frequency cap: 3 impressions per user per day

### 3.4 Smart Bidding Experiments
- Run campaign experiments to test:
  - Maximize Conversions vs. Target CPA ($15)
  - Target CPA ($15) vs. Target CPA ($20) vs. Target CPA ($23)
  - Target ROAS at different thresholds
- Use the Infinity Engine's causal simulation to predict optimal target CPA

---

## Phase 4: Above & Beyond (Months 3-6)

### 4.1 LTV-Based Bidding
- Track actual customer lifetime value (not just first $19)
- If average customer stays 4 months = $76 LTV, bid accordingly
- Implement offline conversion imports to feed LTV data back to Google
- This alone could 3-4x your acceptable CPA and open up many more keywords

### 4.2 Competitor Intelligence Engine
- Build an automated competitor monitoring system:
  - Track competitor ad copy changes via Auction Insights
  - Monitor competitor impression share trends
  - Alert when a new competitor enters your keyword space
  - Auto-adjust bids when competition intensity changes
- Use the Discovery Engine's regime change detection for this

### 4.3 Seasonal & Trend Intelligence
- Integrate Google Trends data for contract/invoicing search volume
- Predict seasonal spikes (tax season, new year, Q4 business planning)
- Auto-increase budgets during high-demand periods
- Auto-decrease during known low periods (holidays)

### 4.4 Cross-Channel Attribution
- Integrate Google Analytics 4 data
- Track the full funnel: Ad Click -> Site Visit -> Signup -> First Contract -> Payment
- Understand which keywords lead to the most engaged subscribers
- Feed this back into the Infinity Engine for better optimization

### 4.5 AI-Generated Ad Copy
- Use LLM to generate ad copy variations based on:
  - Top-performing search terms
  - Competitor ad messaging
  - Product feature updates
  - Seasonal themes
- Auto-test generated copy through the A/B testing engine
- Human review before deployment

### 4.6 Automated Budget Pacing
- Build a budget pacing algorithm that:
  - Distributes monthly budget optimally across days
  - Spends more on high-converting days (weekdays)
  - Spends less on weekends if data supports it
  - Never runs out of budget before month end
  - Accelerates spend when CPA is below target

### 4.7 Predictive Churn Integration
- Once you have enough subscriber data:
  - Build a churn prediction model
  - Identify which ad keywords produce high-churn vs. sticky subscribers
  - Optimize for subscriber quality, not just quantity
  - Negative keywords that bring tire-kickers

### 4.8 Multi-Touch Attribution Model
- Build a custom attribution model using the Infinity Engine's Shapley framework:
  - First Click: Which keyword introduced the user?
  - Last Click: Which keyword closed the sale?
  - Position-Based: 40/20/40 weighting
  - Data-Driven: Use actual conversion path data
- This reveals hidden value in "assist" keywords that current models miss

---

## Phase 5: Theoretical Frontier (6+ Months)

### 5.1 Causal Reinforcement Learning for Bidding
- Replace rule-based bidding with a causal RL agent
- The agent learns the causal structure of: Bid -> Impression -> Click -> Conversion
- Uses do-calculus (already built in Layer 5) to simulate counterfactual bids
- Optimizes for long-term LTV, not just immediate conversions
- This is beyond what Google's own Smart Bidding does

### 5.2 Natural Language Query Expansion
- Use embeddings to find semantically similar keywords
- "contract maker" -> discover "agreement generator", "deal creator", "pact builder"
- Expand keyword universe beyond what Google's Keyword Planner suggests
- Validate new keywords through the Discovery Engine before adding

### 5.3 Dynamic Creative Optimization
- Real-time ad copy assembly based on:
  - User's search query
  - Time of day
  - Device type
  - Geographic location
  - Weather (seriously - rainy days = more desk time = more contract work)
- Uses the Attribution Engine to weight which combinations drive conversions

### 5.4 Auction-Theoretic Bid Optimization
- Model the Google Ads auction as a game-theoretic system
- Use Nash Equilibrium analysis to find optimal bids given competitor behavior
- Predict competitor bid ranges from Auction Insights data
- Find bid levels where you maximize value while competitors overpay

### 5.5 Federated Learning Across Multiple Clients
- If ContractKit adds more products or serves multiple verticals:
  - Train a shared optimization model across all campaigns
  - Each campaign's data improves the others
  - Privacy-preserving: no individual campaign data is exposed
  - The more campaigns, the smarter the system gets

---

## Running the System

### Daily Automation
```bash
# Full optimization cycle (dry run - review before executing)
python ads_mutation_engine.py cycle --customer-id 8360921701

# Full optimization cycle (LIVE - auto-execute changes)
python ads_mutation_engine.py cycle --customer-id 8360921701 --execute

# Generate campaign blueprint
python ads_mutation_engine.py blueprint --customer-id 8360921701

# Generate ad copy variants
python ads_mutation_engine.py ad-copy

# Run Infinity Engine analysis
python run_infinity_loop.py --customer-id 8360921701
```

### Scheduling (Windows Task Scheduler)
Create a daily task that runs `ads_mutation_engine.py cycle` at 6 AM:
1. Open Task Scheduler
2. Create Basic Task -> "ContractKit Ads Optimization"
3. Trigger: Daily at 06:00
4. Action: Start Program -> `python`
5. Arguments: `C:\ContractKit-Ads-Automation\ads_mutation_engine.py cycle --customer-id 8360921701`

---

## Key Metrics to Track

| Metric | Current | Target (30d) | Target (90d) |
|--------|---------|--------------|--------------|
| CPA | $75+ (learning) | < $23 | < $15 |
| ROAS | 0.25x | > 1.0x | > 2.0x |
| CTR | 2.0% | > 4.0% | > 5.0% |
| Conv Rate | ~1% | > 3% | > 5% |
| Quality Score | Unknown | > 6 avg | > 7 avg |
| Monthly Subscribers | 0 | 10+ | 50+ |
