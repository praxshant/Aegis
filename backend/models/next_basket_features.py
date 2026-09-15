"""
next_basket_features.py
-----------------------
Data adapter for the Instacart Next-Basket model.
Transforms generic Aegis transaction data into the 21-feature contract
expected by the LightGBM next-basket model.

Includes explicit mapping from generic product IDs to Instacart taxonomy
(aisle_id, department_id) when a catalog mapping is available.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

logger = logging.getLogger("NextBasketFeatures")

# The exact 21 features expected by the model (leakage-corrected)
NEXT_BASKET_FEATURES = [
    'user_total_orders',
    'user_avg_basket_size',
    'user_avg_days_between_orders',
    'user_reorder_rate',
    'product_total_orders',
    'product_reorder_rate',
    'product_unique_users',
    'product_popularity',
    'user_product_order_count',
    'user_product_reorder_count',
    'user_product_last_order_number',
    'user_product_reorder_rate',
    'user_product_orders_since_last_purchase',
    'user_product_avg_gap',
    'aisle_id',
    'department_id',
    'user_aisle_order_count',
    'user_department_order_count',
    'user_department_share',
    'user_order_number',
    'user_recent_cart_size'
]

N_POPULAR_CANDIDATES = 50


class FeatureBuildError(Exception):
    """Raised when features cannot be built due to missing or invalid data."""
    pass


def _map_transactions_to_instacart_schema(
    customer_id: str,
    tx_df: pd.DataFrame,
    product_catalog: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms raw Aegis transactions into Instacart-style 'orders' and 'order_products' dataframes.
    
    Aegis schema:
        transaction_timestamp, transaction_type, invoice, product_id, ...
        
    Instacart schema required for features:
        orders: order_id, user_id, order_number, days_since_prior_order
        prior_detail: order_id, product_id, reordered, aisle_id, department_id
    """
    if tx_df.empty:
        raise FeatureBuildError("Empty transaction history")
        
    # Filter to purchases only
    if 'transaction_type' in tx_df.columns:
        purchases = tx_df[tx_df['transaction_type'] == 'purchase'].copy()
    else:
        purchases = tx_df.copy()
        
    if purchases.empty:
        raise FeatureBuildError("No purchase transactions found")

    # Sort by time
    purchases['transaction_timestamp'] = pd.to_datetime(purchases['transaction_timestamp'])
    purchases = purchases.sort_values('transaction_timestamp')
    
    # 1. Group by invoice to form orders
    # We use factorize to create integer order_ids
    purchases['order_id'] = pd.factorize(purchases['invoice'])[0] + 1
    
    # Create orders metadata
    orders = purchases.groupby('order_id').agg(
        transaction_timestamp=('transaction_timestamp', 'min')
    ).reset_index()
    orders = orders.sort_values('transaction_timestamp')
    
    # Add order_number
    orders['order_number'] = np.arange(1, len(orders) + 1)
    orders['user_id'] = 1 # Dummy user ID for this customer
    
    # Calculate days_since_prior_order
    orders['days_since_prior_order'] = orders['transaction_timestamp'].diff().dt.total_seconds() / (24 * 3600)
    
    # 2. Create prior_detail
    # We need to map generic product_id to an integer product_id, and assign aisle/department.
    # If a product_catalog mapping is provided (via adapter), we join it. 
    # If not, we have to synthesize it, which might lead to UNAVAILABLE if strict mapping is required.
    
    prior_detail = purchases[['order_id', 'product_id', 'transaction_timestamp']].copy()
    prior_detail['user_id'] = 1
    
    if product_catalog is not None and not product_catalog.empty:
        # Build lookup dict: raw product_id string -> (instacart_product_id, aisle_id, department_id)
        # Supports BOTH 'PROD_101' style aegis IDs AND direct Instacart integer IDs in the CSV
        lookup = {}
        for _, row in product_catalog.iterrows():
            insta_id  = row['instacart_product_id']
            aisle_id  = row['aisle_id']
            dept_id   = row['department_id']
            # Key 1: aegis-style ID (e.g. 'PROD_101')
            lookup[str(row['aegis_product_id'])] = (insta_id, aisle_id, dept_id)
            # Key 2: direct instacart integer ID (e.g. '24852')
            lookup[str(int(insta_id))] = (insta_id, aisle_id, dept_id)

        # Map every row in prior_detail
        prior_detail['product_id_str'] = (
            prior_detail['product_id']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
        )
        
        logger.warning(
            f"DEBUG MAPPING: lookup keys sample={list(lookup.keys())[:5]}, "
            f"TX sample={prior_detail['product_id_str'].head(3).tolist()}"
        )

        prior_detail['mapped_product_id'] = prior_detail['product_id_str'].map(
            lambda x: lookup[x][0] if x in lookup else None
        )
        prior_detail['aisle_id'] = prior_detail['product_id_str'].map(
            lambda x: lookup[x][1] if x in lookup else None
        )
        prior_detail['department_id'] = prior_detail['product_id_str'].map(
            lambda x: lookup[x][2] if x in lookup else None
        )
        prior_detail = prior_detail.drop(columns=['product_id', 'product_id_str'])
        prior_detail = prior_detail.rename(columns={'mapped_product_id': 'product_id'})

        unmapped = prior_detail['product_id'].isna().sum()
        if unmapped > 0:
            logger.warning(f"Failed to map {unmapped} products to Instacart IDs for customer {customer_id}")
            # Drop unmapped products for the purpose of next-basket prediction
            prior_detail = prior_detail.dropna(subset=['product_id'])
            
        if prior_detail.empty:
            raise FeatureBuildError("No products could be mapped to Instacart taxonomy.")
            
        # Ensure integer types
        prior_detail['product_id'] = prior_detail['product_id'].astype(int)
        prior_detail['aisle_id'] = prior_detail['aisle_id'].fillna(1).astype(int)
        prior_detail['department_id'] = prior_detail['department_id'].fillna(1).astype(int)
    else:
        raise FeatureBuildError(
            "product_catalog is required for Instacart next-basket model. "
            "Aegis transactions lack aisle_id/department_id taxonomy."
        )

    # Bring in order_number to calculate 'reordered'
    prior_detail = prior_detail.merge(orders[['order_id', 'order_number']], on='order_id', how='left')
    
    # Calculate 'reordered' (1 if product was purchased in an earlier order, 0 otherwise)
    # Sort by time to ensure cumulative logic works
    prior_detail = prior_detail.sort_values(['product_id', 'order_number'])
    prior_detail['first_order_number'] = prior_detail.groupby('product_id')['order_number'].transform('min')
    prior_detail['reordered'] = (prior_detail['order_number'] > prior_detail['first_order_number']).astype(int)
    
    # We also need days_since_prior_order on the prior_detail
    prior_detail = prior_detail.merge(orders[['order_id', 'days_since_prior_order']], on='order_id', how='left')
    
    return orders, prior_detail


def build_user_features(history_detail: pd.DataFrame, history_orders: pd.DataFrame) -> pd.DataFrame:
    basket_sizes = history_detail.groupby("order_id").size().rename("basket_size")
    orders_with_size = history_orders.merge(basket_sizes, left_on="order_id", right_index=True, how="left")

    agg = orders_with_size.groupby("user_id").agg(
        user_total_orders=("order_id", "nunique"),
        user_avg_basket_size=("basket_size", "mean"),
        user_avg_days_between_orders=("days_since_prior_order", "mean"),
    )
    reorder_stats = history_detail.groupby("user_id")["reordered"].mean().rename("user_reorder_rate")
    return agg.join(reorder_stats, how="left").reset_index()


def build_product_features(history_detail: pd.DataFrame) -> pd.DataFrame:
    agg = history_detail.groupby("product_id").agg(
        product_total_orders=("order_id", "nunique"),
        product_reorder_rate=("reordered", "mean"),
        product_unique_users=("user_id", "nunique"),
    ).reset_index()
    agg["product_popularity"] = agg["product_total_orders"].rank(pct=True)
    return agg


def build_user_product_features(history_detail: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    up = (
        history_detail[["user_id", "product_id", "order_id", "order_number", "reordered"]]
        .groupby(["user_id", "product_id"], sort=False, observed=True)
        .agg(
            user_product_order_count=("order_id", "nunique"),
            user_product_reorder_count=("reordered", "sum"),
            user_product_last_order_number=("order_number", "max"),
        )
        .reset_index()
    )

    up = up.merge(user_features[["user_id", "user_total_orders"]], on="user_id", how="left", sort=False)
    up["user_product_reorder_rate"] = up["user_product_reorder_count"] / up["user_product_order_count"].clip(lower=1)
    up["user_product_orders_since_last_purchase"] = up["user_total_orders"] - up["user_product_last_order_number"]
    up = up.drop(columns=["user_total_orders"])

    order_days = history_detail[["user_id", "order_id", "order_number", "days_since_prior_order"]].drop_duplicates(subset=["user_id", "order_id"], keep="first")
    order_days = order_days.sort_values(["user_id", "order_number"], kind="mergesort")
    order_days["days_since_prior_order"] = order_days["days_since_prior_order"].fillna(0).astype("float32")
    order_days["days_since_first_order"] = order_days.groupby("user_id", sort=False)["days_since_prior_order"].cumsum().astype("float32")
    order_days = order_days[["user_id", "order_id", "days_since_first_order"]]

    timeline = (
        history_detail[["user_id", "product_id", "order_id"]]
        .drop_duplicates(subset=["user_id", "product_id", "order_id"], keep="first")
        .merge(order_days, on=["user_id", "order_id"], how="left", sort=False)
    )
    timeline = timeline.sort_values(["user_id", "product_id", "days_since_first_order"], kind="mergesort")
    timeline["gap"] = timeline.groupby(["user_id", "product_id"], sort=False, observed=True)["days_since_first_order"].diff().astype("float32")

    gap = timeline.groupby(["user_id", "product_id"], sort=False, observed=True)["gap"].mean().rename("user_product_avg_gap").reset_index()
    up = up.merge(gap, on=["user_id", "product_id"], how="left", sort=False)

    return up


def build_category_affinity(history_detail: pd.DataFrame):
    aisle_aff = history_detail.groupby(["user_id", "aisle_id"]).size().rename("user_aisle_order_count").reset_index()
    dept_counts = history_detail.groupby(["user_id", "department_id"]).size().rename("user_department_order_count").reset_index()
    user_totals = history_detail.groupby("user_id").size().rename("user_total_items")
    dept_counts = dept_counts.merge(user_totals, on="user_id", how="left")
    dept_counts["user_department_share"] = dept_counts["user_department_order_count"] / dept_counts["user_total_items"]
    dept_counts = dept_counts.drop(columns=["user_total_items"])
    return aisle_aff, dept_counts


def build_basket_behavior(history_detail: pd.DataFrame, history_orders: pd.DataFrame) -> pd.DataFrame:
    basket_sizes = history_detail.groupby("order_id").size().rename("basket_size")
    orders_with_size = history_orders.merge(basket_sizes, left_on="order_id", right_index=True, how="left")
    last_order = orders_with_size.sort_values("order_number").groupby("user_id").tail(1)
    return last_order[["user_id", "order_number", "basket_size"]].rename(
        columns={"order_number": "user_order_number", "basket_size": "user_recent_cart_size"}
    )


def generate_candidates(history_detail: pd.DataFrame, product_features: pd.DataFrame, n_popular: int) -> pd.DataFrame:
    purchased = history_detail[["user_id", "product_id"]].drop_duplicates()
    popular_products = product_features.sort_values("product_total_orders", ascending=False).head(n_popular)["product_id"].tolist()
    users = history_detail["user_id"].unique()
    popular_grid = pd.MultiIndex.from_product([users, popular_products], names=["user_id", "product_id"]).to_frame(index=False)
    return pd.concat([purchased, popular_grid], ignore_index=True).drop_duplicates()


def assemble_features_for_next_basket(
    customer_id: str,
    tx_df: pd.DataFrame,
    product_catalog: Optional[pd.DataFrame] = None,
    global_product_features: Optional[pd.DataFrame] = None
) -> Tuple[str, Optional[pd.DataFrame], str]:
    """
    Adapter entrypoint.
    Returns: (status: str, candidates_df: DataFrame, reason: str)
      status: "COMPLETE" or "UNAVAILABLE"
    """
    try:
        orders, prior_detail = _map_transactions_to_instacart_schema(customer_id, tx_df, product_catalog)
    except FeatureBuildError as e:
        logger.info(f"NextBasket UNAVAILABLE for {customer_id}: {e}")
        return "UNAVAILABLE", None, str(e)
        
    if len(orders) < 2:
        reason = "Insufficient order history (< 2 invoices)."
        logger.info(f"NextBasket UNAVAILABLE for {customer_id}: {reason}")
        return "UNAVAILABLE", None, reason

    # 1. Build components
    user_feats = build_user_features(prior_detail, orders)
    
    # If we have a global product feature table, use it. Otherwise, compute locally.
    # In a real environment, product_feats is pre-computed globally. For this demo,
    # we compute it from the user's history if global is missing.
    if global_product_features is not None and not global_product_features.empty:
        product_feats = global_product_features
    else:
        product_feats = build_product_features(prior_detail)
        
    up_feats = build_user_product_features(prior_detail, user_feats)
    aisle_aff, dept_aff = build_category_affinity(prior_detail)
    basket_feats = build_basket_behavior(prior_detail, orders)
    
    # 2. Generate candidates
    candidates = generate_candidates(prior_detail, product_feats, N_POPULAR_CANDIDATES)
    
    # 3. Assemble full frame
    frame = candidates.merge(user_feats, on="user_id", how="left", sort=False)
    frame = frame.merge(product_feats, on="product_id", how="left", sort=False)
    frame = frame.merge(up_feats, on=["user_id", "product_id"], how="left", sort=False)
    
    # We need aisle_id and department_id for candidates.
    # We can get this from the product_catalog.
    if product_catalog is not None:
        cat_subset = product_catalog[['instacart_product_id', 'aisle_id', 'department_id']].drop_duplicates()
        cat_subset = cat_subset.rename(columns={'instacart_product_id': 'product_id'})
        frame = frame.merge(cat_subset, on="product_id", how="left", sort=False)
    else:
        # Fallback (should not happen if we passed validation above)
        frame['aisle_id'] = 1
        frame['department_id'] = 1
        
    frame = frame.merge(aisle_aff, on=["user_id", "aisle_id"], how="left", sort=False)
    frame = frame.merge(dept_aff, on=["user_id", "department_id"], how="left", sort=False)
    frame = frame.merge(basket_feats, on="user_id", how="left", sort=False)

    # Ensure all 21 features are present and filled
    missing_cols = [c for c in NEXT_BASKET_FEATURES if c not in frame.columns]
    if missing_cols:
        logger.error(f"Missing features after assembly: {missing_cols}")
        return "UNAVAILABLE", None, f"Feature assembly failed. Missing {missing_cols}"
        
    frame[NEXT_BASKET_FEATURES] = frame[NEXT_BASKET_FEATURES].fillna(0)
    
    return "COMPLETE", frame, ""
