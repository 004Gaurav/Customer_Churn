# Business Insights — Customer Churn

## Executive Summary
- Churn risk appears most sensitive to customer **age**, **activity status**, and **balance**.
- Older, less active customers show higher churn tendency.
- Regional differences suggest location-specific drivers.

## Key Insights
1. **Age is the strongest churn signal**
   - Churn increases with age in the dataset.
   - Retention efforts should prioritize older segments.

2. **Active members churn less**
   - `IsActiveMember` is negatively associated with churn.
   - Engagement programs can reduce churn risk.

3. **Balance is mildly associated with churn**
   - Higher balances show a slight positive relationship with churn.
   - High-balance customers should receive proactive retention outreach.

4. **Geography shows differences in churn**
   - One-hot correlations indicate regional variation (e.g., Germany differs from France/Spain).
   - Investigate local market factors and tailor regional strategies.

5. **Gender has a weak effect**
   - Gender-related correlations are small and likely not a major driver.
   - Avoid over-indexing on gender for targeting.

## Operational Recommendations
- **Segmented retention campaigns**: prioritize older + inactive + high-balance customers.
- **Engagement lift**: targeted onboarding, usage nudges, loyalty benefits for inactive users.
- **Regional deep-dive**: analyze churn drivers by geography and adjust retention offers.

## Modeling Note
- The current model improves churn recall after threshold tuning, which helps flag more at-risk customers.
- For business use, focus on recall and F1 for churn (class 1) rather than accuracy alone.
