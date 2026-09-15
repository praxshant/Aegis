# Model Integration

Aegis implements a multi-model intelligence architecture. It currently hosts two specialized machine learning models.

---

## 1. Propensity Model (Next Purchase 30d)

### Provenance
- **Original Source Path**: `C:\Users\ACER\OneDrive\Desktop\Projects\Customer-Intelligence-Platform\backend\training\experiments\next_purchase_30d\model_HistGradientBoosting_Tuned.pkl`
- **Aegis Path**: `artifacts/models/next_purchase_30d/model_HistGradientBoosting_Tuned.pkl`
- **Artifact Hash (SHA-256)**: `23aa3acebaabc1e84818f8ea727e4693036ecaba03741f1b45b1a3de4d07f7bf`
- **Source Notebook**: `03_model_training.ipynb` (Aegis experiment)
- **Audit Status**: Audited and selected based on maximum PR-AUC in `evaluation_report.json`.

### Model Contract
- **Model Type**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Target**: `TARGET_next_purchase_30d` (Predicts the probability a customer will make a purchase in the next 30 days)
- **Missing Values**: Handled natively by `HistGradientBoostingClassifier`.

### Exact Feature List & Ordering
The model was trained on exactly 19 features:
1. `FEATURE_recency_days`
2. `FEATURE_customer_lifetime_days`
3. `FEATURE_purchase_order_count`
4. `FEATURE_purchase_line_count`
5. `FEATURE_return_line_count`
6. `FEATURE_cancellation_line_count`
7. `FEATURE_total_gross_purchase_spend`
8. `FEATURE_total_return_amount`
9. `FEATURE_total_net_spend`
10. `FEATURE_average_order_value`
11. `FEATURE_purchase_frequency`
12. `FEATURE_return_rate`
13. `FEATURE_unique_products_purchased`
14. `FEATURE_net_spend_30d`
15. `FEATURE_net_spend_90d`
16. `FEATURE_purchase_order_count_30d`
17. `FEATURE_purchase_order_count_90d`
18. `FEATURE_is_active_30d`
19. `FEATURE_is_active_90d`

---

## 2. Next-Basket Prediction Model

### Provenance
- **Original Source Path**: `C:\Users\ACER\OneDrive\Desktop\Projects\Customer-Intelligence-Platform\backend\models\artifacts\next_basket\next_basket_model_corrected.pkl`
- **Aegis Path**: `artifacts/models/next_basket/next_basket_model_corrected.pkl`
- **Artifact Hash (SHA-256)**: `238a44c50b10d825c2417ae8e278465d9366afa1651bc826cd1b2d81d0e4ea9c`
- **Source Notebook**: `instacart_next_basket_training2.ipynb` (Phase 9: Leakage Corrected)
- **Audit Status**: Evaluated on holdout data. NDCG@10: 0.524, Recall@10: 0.534.

### Model Contract
- **Model Type**: `lightgbm.Booster`
- **Target**: Next basket repurchase probability for candidate products.
- **Data Adapter Requirement**: This model requires an Instacart-compatible product taxonomy (`aisle_id`, `department_id`) and order grouping. Aegis handles this via `backend/models/next_basket_features.py`.

### Exact Feature List & Ordering
The model expects exactly 21 features (after 3 leakage features were removed):
1. `user_total_orders`
2. `user_avg_basket_size`
3. `user_avg_days_between_orders`
4. `user_reorder_rate`
5. `product_total_orders`
6. `product_reorder_rate`
7. `product_unique_users`
8. `product_popularity`
9. `user_product_order_count`
10. `user_product_reorder_count`
11. `user_product_last_order_number`
12. `user_product_reorder_rate`
13. `user_product_orders_since_last_purchase`
14. `user_product_avg_gap`
15. `aisle_id`
16. `department_id`
17. `user_aisle_order_count`
18. `user_department_order_count`
19. `user_department_share`
20. `user_order_number`
21. `user_recent_cart_size`

---

## 3. Required Runtime Logic
- **Do not train**: Aegis must only load and execute inference from these artifacts.
- **Handling NaN**: Missing numeric behavioral features must be handled natively by LightGBM and HistGradientBoostingClassifier.
- **Fail Gracefully**: If required intelligence cannot be generated (e.g. missing product taxonomy for Next-Basket), a controlled fallback `UNAVAILABLE` state must be used.
- **Architecture**: The `IntelligenceEngine` runs all models, merges their outputs into the `CustomerIntelligence` payload, and feeds it into the LLM Agent.
