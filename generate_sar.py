import pandas as pd
import numpy as np
import sqlite3
import json
import shap
import joblib
import re
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def load_alert_data(alert_id, db_path='aml_data.db'):
    """Load all data for a specific alert"""
    conn = sqlite3.connect(db_path)

    # Get alert
    alert = pd.read_sql(f"SELECT * FROM alerts WHERE alert_id = {alert_id}", conn).iloc[0]

    # Get customer
    customer = pd.read_sql(f"SELECT * FROM customers WHERE customer_id = {alert['customer_id']}", conn).iloc[0]

    # Get transactions
    transactions = pd.read_sql(f"SELECT * FROM transactions WHERE customer_id = {alert['customer_id']}", conn)

    conn.close()

    return alert, customer, transactions

def format_customer_info(customer):
    """Format customer information for SAR narrative"""
    ctype = 'SSN' if customer['customer_type'] == 'individual' else 'EIN'
    info = f"""Customer Name: {customer['name']}
Customer Type: {customer['customer_type'].title()}
{ctype}: {customer['ssn_ein']}
Occupation/Business: {customer['occupation']}
Address: {customer['address']}, {customer['city']}, {customer['state']} {customer['zip_code']}
Annual Income: ${customer['annual_income']:,.2f}
Account Opening Date: {customer['account_open_date']}
Risk Rating: {customer['risk_rating'].title()}"""
    return info

def format_transaction_summary(transactions, top_n=20):
    """Format transaction summary for SAR narrative"""
    transactions = transactions.copy()
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    transactions = transactions.sort_values('transaction_date', ascending=False)

    # Overall statistics
    deposits = transactions[transactions['transaction_type'] == 'deposit']
    withdrawals = transactions[transactions['transaction_type'] == 'withdrawal']
    total_deposits = deposits['amount'].sum()
    total_withdrawals = withdrawals['amount'].sum()
    num_transactions = len(transactions)
    date_range_start = transactions['transaction_date'].min().strftime('%Y-%m-%d')
    date_range_end = transactions['transaction_date'].max().strftime('%Y-%m-%d')

    # Cash transactions
    cash_txns = transactions[transactions['method'] == 'cash']
    num_cash = len(cash_txns)
    total_cash = cash_txns['amount'].sum()

    # Wire transactions
    wire_txns = transactions[transactions['method'] == 'wire']
    num_wire = len(wire_txns)

    # International transactions
    intl_txns = transactions[transactions['country'] != 'USA']
    num_intl = len(intl_txns)

    summary = f"""Total Transactions: {num_transactions}
Date Range: {date_range_start} to {date_range_end}
Total Deposits: ${total_deposits:,.2f} ({len(deposits)} transactions)
Total Withdrawals: ${total_withdrawals:,.2f} ({len(withdrawals)} transactions)
Net Flow: ${total_deposits - total_withdrawals:,.2f}

Cash Transactions: {num_cash} totaling ${total_cash:,.2f}
Wire Transfers: {num_wire}
International Transactions: {num_intl} across {intl_txns['country'].nunique()} countries"""

    if num_intl > 0:
        summary += f"\nCountries involved: {', '.join(intl_txns['country'].unique()[:10])}"

    # Add top transactions
    summary += f"\n\nTop {min(top_n, len(transactions))} Most Recent/Significant Transactions:"

    top_txns = transactions.head(top_n)
    for _, txn in top_txns.iterrows():
        line = f"\n- {txn['transaction_date'].strftime('%Y-%m-%d')}: {txn['transaction_type'].title()} ${txn['amount']:,.2f} via {txn['method']}"
        if pd.notna(txn.get('counterparty')) and txn['counterparty']:
            line += f" (Counterparty: {txn['counterparty']}, Country: {txn['country']})"
        summary += line

    return summary

def format_alert_details(alert, features_df, explainer, feature_cols):
    """Format alert details with risk score and key risk factors"""
    # Find the customer's features
    alert_features = features_df[features_df['customer_id'] == alert['customer_id']]

    if len(alert_features) == 0:
        return f"""Alert ID: {alert['alert_id']}
Alert Type: {alert['alert_type'].replace('_', ' ').title()}
Alert Date: {alert['alert_date']}
Severity: {alert['severity'].title()}"""

    alert_features = alert_features.iloc[0]

    # Get SHAP explanation
    X_alert = alert_features[feature_cols].values.reshape(1, -1)
    shap_values = explainer.shap_values(X_alert)

    # Get top 5 risk drivers
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', ascending=False)

    top_drivers = shap_importance.head(5)

    risk_score = alert_features.get('risk_score', 0.0)

    details = f"""Alert ID: {alert['alert_id']}
Alert Type: {alert['alert_type'].replace('_', ' ').title()}
Alert Date: {alert['alert_date']}
Severity: {alert['severity'].title()}
Risk Score: {risk_score:.2%}

Top Risk Drivers (SHAP Analysis):"""

    for _, row in top_drivers.iterrows():
        feature_name = row['feature'].replace('_', ' ').title()
        feature_value = alert_features[row['feature']]
        shap_val = row['shap_value']
        details += f"\n- {feature_name}: {feature_value:.2f} (SHAP impact: {shap_val:+.3f})"

    return details, top_drivers

def generate_sar_narrative(alert_id, vectorstore, llm, features_df, explainer, feature_cols, db_path='aml_data.db'):
    """Generate complete SAR narrative with audit trail"""
    print(f"\n{'=' * 60}")
    print(f"GENERATING SAR FOR ALERT {alert_id}")
    print(f"{'=' * 60}")

    # Load data
    alert, customer, transactions = load_alert_data(alert_id, db_path)

    # Format inputs
    customer_info = format_customer_info(customer)
    transaction_summary = format_transaction_summary(transactions)
    alert_result = format_alert_details(alert, features_df, explainer, feature_cols)

    if isinstance(alert_result, tuple):
        alert_details, top_drivers = alert_result
    else:
        alert_details = alert_result
        top_drivers = pd.DataFrame()

    alert_features = features_df[features_df['customer_id'] == alert['customer_id']]
    risk_score = alert_features.iloc[0].get('risk_score', 0.0) if len(alert_features) > 0 else 0.0

    print(f"\n1. Customer: {customer['name']}")
    print(f"2. Typology: {alert['alert_type']}")
    print(f"3. Total Transactions: {len(transactions)}")
    print(f"4. Risk Score: {risk_score:.2%}")

    # Retrieve relevant templates
    query = f"{alert['alert_type'].replace('_', ' ')} suspicious activity"
    retrieved_docs = vectorstore.similarity_search(query, k=2)

    # Build context from retrieved templates
    context = "\n\n---\n\n".join([
        f"Template ({doc.metadata['typology']}):\n{doc.page_content[:800]}"
        for doc in retrieved_docs
    ])

    # Create prompt
    prompt_text = f"""You are a compliance officer writing a Suspicious Activity Report (SAR) narrative for FinCEN.

Use the following SAR template examples as guidance for structure and tone:
{context}

Based on the customer information and transaction data below, write a complete SAR narrative following the 5W+H structure.

Customer Information:
{customer_info}

Transaction Summary:
{transaction_summary}

Alert Details:
{alert_details}

CRITICAL REQUIREMENTS:
1. Use ONLY the information provided above - do not fabricate any details
2. Follow this EXACT structure with these section headers: INTRODUCTION, WHO, WHAT, WHEN, WHERE, WHY SUSPICIOUS, HOW, CONCLUSION
3. Keep the narrative professional and factual
4. Use specific amounts, dates, and numbers from the data
5. Explain WHY the activity is suspicious based on the typology
6. Length: 400-600 words
7. Start directly with "INTRODUCTION" - no preamble

SAR NARRATIVE:
"""

    print("\n5. Generating narrative with Mistral 7B...")
    print("   (This may take 30-90 seconds...)")

    # Generate narrative
    narrative = llm.invoke(prompt_text)

    # Clean up the narrative
    narrative = narrative.strip()

    print(f"\n[OK] Narrative generated!")
    print(f"   Length: {len(narrative.split())} words")

    # Create audit trail
    audit_trail = {
        'alert_id': int(alert_id),
        'generation_timestamp': datetime.now().isoformat(),
        'model': 'mistral:7b',
        'temperature': 0.1,

        # Data lineage
        'customer_id': int(customer['customer_id']),
        'customer_name': customer['name'],
        'num_transactions_analyzed': len(transactions),
        'transaction_date_range': f"{transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}",

        # Model decision
        'risk_score': float(risk_score),
        'typology_detected': alert['alert_type'],
        'shap_top_features': top_drivers.to_dict('records') if len(top_drivers) > 0 else [],

        # RAG retrieval
        'templates_retrieved': [
            {
                'typology': doc.metadata['typology'],
                'template_id': doc.metadata.get('template_id', 'unknown')
            }
            for doc in retrieved_docs
        ],

        # LLM generation
        'prompt_template_used': True,
        'generation_params': {
            'temperature': 0.1,
            'max_tokens': 2000
        },

        # Output
        'narrative': narrative,
        'narrative_word_count': len(narrative.split()),

        # Human review status
        'status': 'draft',
        'reviewed_by': None,
        'approved_by': None,
        'filed_date': None
    }

    return narrative, audit_trail

def validate_sar_compliance(narrative):
    """Validate SAR narrative meets FinCEN requirements"""
    print("\n6. Validating compliance...")

    required_sections = [
        'INTRODUCTION', 'WHO', 'WHAT', 'WHEN', 'WHERE',
        'WHY SUSPICIOUS', 'HOW', 'CONCLUSION'
    ]

    compliance_check = {
        'has_all_sections': True,
        'missing_sections': [],
        'word_count_ok': False,
        'has_specific_amounts': False,
        'has_specific_dates': False
    }

    narrative_upper = narrative.upper()

    # Check sections
    for section in required_sections:
        if section not in narrative_upper:
            compliance_check['has_all_sections'] = False
            compliance_check['missing_sections'].append(section)

    # Word count (300-1000 words is acceptable for a POC)
    word_count = len(narrative.split())
    compliance_check['word_count_ok'] = 200 <= word_count <= 1000
    compliance_check['word_count'] = word_count

    # Check for specific amounts ($ signs)
    compliance_check['has_specific_amounts'] = '$' in narrative

    # Check for specific dates
    date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}'
    compliance_check['has_specific_dates'] = bool(re.search(date_pattern, narrative))

    # Overall compliance
    compliance_check['compliant'] = (
        compliance_check['has_all_sections'] and
        compliance_check['word_count_ok'] and
        compliance_check['has_specific_amounts'] and
        compliance_check['has_specific_dates']
    )

    # Section count
    sections_found = sum(1 for s in required_sections if s in narrative_upper)
    compliance_check['sections_found'] = sections_found
    compliance_check['sections_total'] = len(required_sections)

    # Print results
    if compliance_check['compliant']:
        print(f"   [OK] SAR is COMPLIANT with FinCEN requirements")
    else:
        print(f"   [!!] SAR has compliance issues:")
        if not compliance_check['has_all_sections']:
            print(f"     - Missing sections: {', '.join(compliance_check['missing_sections'])}")
        if not compliance_check['word_count_ok']:
            print(f"     - Word count {word_count} outside recommended 200-1000 range")
        if not compliance_check['has_specific_amounts']:
            print("     - No specific dollar amounts found")
        if not compliance_check['has_specific_dates']:
            print("     - No specific dates found")

    print(f"   Sections found: {sections_found}/{len(required_sections)}")
    print(f"   Word count: {word_count}")

    return compliance_check

def save_sar_and_audit_trail(alert_id, narrative, audit_trail, compliance_check):
    """Save SAR narrative and audit trail"""
    os.makedirs('outputs', exist_ok=True)

    # Create SAR document
    sar_document = {
        'alert_id': alert_id,
        'narrative': narrative,
        'audit_trail': audit_trail,
        'compliance_check': compliance_check,
        'generated_at': datetime.now().isoformat()
    }

    # Save to JSON
    filename = f'outputs/sar_alert_{alert_id}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sar_document, f, indent=2, ensure_ascii=False)

    print(f"\n7. SAR saved to {filename}")

    # Save narrative as text file
    text_filename = f'outputs/sar_narrative_{alert_id}.txt'
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(f"SUSPICIOUS ACTIVITY REPORT - NARRATIVE\n")
        f.write(f"Alert ID: {alert_id}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(narrative)

    print(f"8. Narrative saved to {text_filename}")

    return sar_document

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("SAR NARRATIVE GENERATION ENGINE")
    print("=" * 60)

    # Load RAG components
    print("\nLoading components...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory='./chroma_db',
        embedding_function=embeddings
    )

    model_name="llama-3.1-8b-instant", 
    temperature=0.1

    features_df = pd.read_csv('alert_features_scored.csv')
    explainer = joblib.load('shap_explainer.pkl')
    feature_cols = joblib.load('feature_columns.pkl')

    print(f"[OK] All components loaded")
    print(f"  - Vector store: {vectorstore._collection.count()} templates")
    print(f"  - Features: {len(features_df)} customers scored")
    print(f"  - Model: mistral:7b")

    # Generate SAR for alert ID 0 (first suspicious customer)
    alert_id = 0

    narrative, audit_trail = generate_sar_narrative(
        alert_id,
        vectorstore,
        llm,
        features_df,
        explainer,
        feature_cols
    )

    # Validate compliance
    compliance_check = validate_sar_compliance(narrative)

    # Save
    sar_document = save_sar_and_audit_trail(
        alert_id,
        narrative,
        audit_trail,
        compliance_check
    )

    print("\n" + "=" * 60)
    print("SAR GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nNarrative Preview:")
    print("-" * 60)
    print(narrative[:800])
    if len(narrative) > 800:
        print("...")
