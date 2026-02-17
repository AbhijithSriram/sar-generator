import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib
import os

def load_data(db_path='aml_data.db'):
    """Load data from database"""
    conn = sqlite3.connect(db_path)

    customers = pd.read_sql('SELECT * FROM customers', conn)
    transactions = pd.read_sql('SELECT * FROM transactions', conn)
    alerts = pd.read_sql('SELECT * FROM alerts', conn)

    conn.close()
    return customers, transactions, alerts

def engineer_features(customers_df, transactions_df):
    """Create features for ALL customers (for training classifier)"""
    print("Engineering features for all customers...")

    features_list = []

    for idx, customer in customers_df.iterrows():
        customer_txns = transactions_df[transactions_df['customer_id'] == customer['customer_id']]

        if len(customer_txns) == 0:
            continue

        # Temporal features
        customer_txns = customer_txns.copy()
        customer_txns['transaction_date'] = pd.to_datetime(customer_txns['transaction_date'])
        date_range = (customer_txns['transaction_date'].max() - customer_txns['transaction_date'].min()).days
        if date_range == 0:
            date_range = 1

        # Transaction patterns
        deposits = customer_txns[customer_txns['transaction_type'] == 'deposit']
        withdrawals = customer_txns[customer_txns['transaction_type'] == 'withdrawal']

        # Cash transactions
        cash_txns = customer_txns[customer_txns['method'] == 'cash']
        wire_txns = customer_txns[customer_txns['method'] == 'wire']

        # International transactions
        intl_txns = customer_txns[customer_txns['country'] != 'USA']

        # High-risk countries
        high_risk_countries = ['China', 'Russia', 'UAE', 'Panama', 'Cyprus', 'Cayman Islands',
                               'British Virgin Islands', 'Seychelles', 'Belize', 'Malta',
                               'Luxembourg', 'Mauritius', 'Hong Kong', 'Singapore', 'Switzerland']
        high_risk_txns = customer_txns[customer_txns['country'].isin(high_risk_countries)]

        # Structuring indicators
        just_under_10k = customer_txns[
            (customer_txns['amount'] >= 7000) &
            (customer_txns['amount'] < 10000)
        ]

        # Rapid movement
        customer_txns_sorted = customer_txns.sort_values('transaction_date')
        time_diffs = customer_txns_sorted['transaction_date'].diff()
        rapid_txns = (time_diffs < pd.Timedelta(days=2)).sum() if len(time_diffs) > 1 else 0

        # Avg days between transactions
        avg_days = 0
        if len(time_diffs.dropna()) > 0:
            avg_days_td = time_diffs.dropna().mean()
            avg_days = avg_days_td.days if hasattr(avg_days_td, 'days') else 0

        features = {
            'customer_id': customer['customer_id'],

            # Customer profile features
            'customer_type_business': 1 if customer['customer_type'] == 'business' else 0,
            'annual_income': customer['annual_income'],
            'account_age_days': (pd.Timestamp('2025-12-31') - pd.to_datetime(customer['account_open_date'])).days,
            'risk_rating_high': 1 if customer['risk_rating'] == 'high' else 0,

            # Transaction volume features
            'total_transactions': len(customer_txns),
            'total_volume': customer_txns['amount'].sum(),
            'avg_transaction_amount': customer_txns['amount'].mean(),
            'max_transaction_amount': customer_txns['amount'].max(),
            'std_transaction_amount': customer_txns['amount'].std() if len(customer_txns) > 1 else 0,
            'transaction_date_range_days': date_range,

            # Transaction type ratios
            'num_deposits': len(deposits),
            'num_withdrawals': len(withdrawals),
            'deposit_withdrawal_ratio': len(deposits) / (len(withdrawals) + 1),
            'total_deposits': deposits['amount'].sum() if len(deposits) > 0 else 0,
            'total_withdrawals': withdrawals['amount'].sum() if len(withdrawals) > 0 else 0,

            # Method features
            'num_cash_txns': len(cash_txns),
            'num_wire_txns': len(wire_txns),
            'pct_cash': len(cash_txns) / len(customer_txns),
            'pct_wire': len(wire_txns) / len(customer_txns),
            'total_cash_amount': cash_txns['amount'].sum() if len(cash_txns) > 0 else 0,

            # International features
            'num_intl_txns': len(intl_txns),
            'num_high_risk_country_txns': len(high_risk_txns),
            'pct_intl': len(intl_txns) / len(customer_txns),
            'total_intl_amount': intl_txns['amount'].sum() if len(intl_txns) > 0 else 0,

            # Structuring indicators
            'num_just_under_10k': len(just_under_10k),
            'pct_just_under_10k': len(just_under_10k) / len(customer_txns),

            # Velocity features
            'num_rapid_sequence_txns': rapid_txns,
            'avg_days_between_txns': avg_days,

            # Profile inconsistency
            'volume_to_income_ratio': customer_txns['amount'].sum() / (customer['annual_income'] + 1),

            # Number of unique counterparties
            'num_unique_counterparties': customer_txns['counterparty'].nunique(),

            # Number of unique locations
            'num_unique_locations': customer_txns['location'].nunique(),

            # Target
            'is_suspicious': int(customer['is_suspicious']),
            'typology': customer['typology'] if customer['is_suspicious'] else 'normal'
        }

        features_list.append(features)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx+1:,} / {len(customers_df):,} customers")

    return pd.DataFrame(features_list)

def train_risk_classifier(features_df):
    """Train XGBoost classifier for risk scoring"""
    print("\nTraining risk classification model...")

    # Prepare features and target
    # Exclude risk_rating_high as it directly leaks the label (set during data generation)
    exclude_cols = ['customer_id', 'is_suspicious', 'typology', 'risk_rating_high', 'risk_score']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    X = features_df[feature_cols].values
    y = features_df['is_suspicious'].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Suspicious']))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Get risk scores for ALL customers
    risk_scores = model.predict_proba(X)[:, 1]
    features_df['risk_score'] = risk_scores

    # SHAP explanations
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save model and explainer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/risk_classifier.pkl')
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')

    # Also save in root for backward compatibility
    joblib.dump(model, 'risk_classifier.pkl')
    joblib.dump(explainer, 'shap_explainer.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')

    print("[OK] Model and explainer saved")

    return features_df, model, explainer, shap_values, feature_cols

def save_enriched_alerts(features_df, alerts_df, db_path='aml_data.db'):
    """Save enriched alerts back to database"""
    print("\nSaving enriched alerts...")

    conn = sqlite3.connect(db_path)

    # Merge risk scores into alerts via customer_id
    alert_features = features_df[features_df['is_suspicious'] == 1][['customer_id', 'risk_score', 'typology']]

    # Merge with alerts
    alerts_enriched = alerts_df.copy()
    alerts_enriched = alerts_enriched.merge(
        alert_features,
        on='customer_id',
        how='left',
        suffixes=('', '_feat')
    )

    # Use the feature typology if alert_type was already there
    if 'typology_feat' in alerts_enriched.columns:
        alerts_enriched.drop(columns=['typology_feat'], inplace=True)

    alerts_enriched.to_sql('alerts', conn, if_exists='replace', index=False)
    conn.close()

    print("[OK] Alerts updated with risk scores")
    return alerts_enriched

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("ALERT CLASSIFICATION & RISK SCORING")
    print("=" * 60)

    # Load data
    customers_df, transactions_df, alerts_df = load_data()
    print(f"Loaded {len(customers_df):,} customers, {len(transactions_df):,} transactions, {len(alerts_df):,} alerts")

    # Engineer features for ALL customers
    features_df = engineer_features(customers_df, transactions_df)
    features_df.to_csv('alert_features.csv', index=False)
    print(f"\nEngineered {len(features_df.columns)} features for {len(features_df):,} customers")

    # Train classifier
    features_df, model, explainer, shap_values, feature_cols = train_risk_classifier(features_df)

    # Save enriched alerts
    alerts_enriched = save_enriched_alerts(features_df, alerts_df)

    # Save features with scores
    features_df.to_csv('alert_features_scored.csv', index=False)

    # Print top risk alerts
    print("\n" + "=" * 60)
    print("TOP 10 HIGHEST RISK ALERTS")
    print("=" * 60)

    suspicious = features_df[features_df['is_suspicious'] == 1].nlargest(10, 'risk_score')
    for _, row in suspicious.iterrows():
        cust = customers_df[customers_df['customer_id'] == row['customer_id']].iloc[0]
        print(f"  Customer {int(row['customer_id']):4d} | {cust['name']:30s} | "
              f"{row['typology']:20s} | Risk: {row['risk_score']:.2%}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 60)
    print(f"\nFeature importance (top 10):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    for _, row in importance.iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.4f}")
