import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
import joblib
import glob
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from generate_sar import (
    generate_sar_narrative, validate_sar_compliance,
    save_sar_and_audit_trail, load_alert_data,
    format_customer_info, format_transaction_summary
)

# Page config
st.set_page_config(
    page_title="SAR Narrative Generator",
    page_icon="shield",
    layout="wide"
)

# Custom CSS - force metric text to be readable on all themes
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
@st.cache_resource
def load_all_data():
    conn = sqlite3.connect('aml_data.db')
    customers = pd.read_sql('SELECT * FROM customers', conn)
    transactions = pd.read_sql('SELECT * FROM transactions', conn)
    alerts = pd.read_sql('SELECT * FROM alerts', conn)
    conn.close()

    features_df = pd.read_csv('alert_features_scored.csv')
    explainer = joblib.load('shap_explainer.pkl')
    feature_cols = joblib.load('feature_columns.pkl')

    # 1. RAG components - Use HuggingFace for Cloud compatibility
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Initialize the vectorstore (CRITICAL: This was missing in your snippet)
    vectorstore = Chroma(
        persist_directory='./chroma_db',
        embedding_function=embeddings
    )

    # 3. Use st.secrets for the API key to avoid GitHub push protection errors
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="mistral-7b-instruct-v0.1", 
        temperature=0.1
    )

    return customers, transactions, alerts, features_df, explainer, feature_cols, vectorstore, llm

customers_df, transactions_df, alerts_df, features_df, explainer, feature_cols, vectorstore, llm = load_all_data()

# Sidebar
st.sidebar.title("SAR Generator")
st.sidebar.markdown("AI-Powered Suspicious Activity Report Generation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Alert Dashboard", "Generate SAR", "Audit Trail Viewer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info**")
st.sidebar.markdown(f"- Model: Mistral 7B")
st.sidebar.markdown(f"- Customers: {len(customers_df):,}")
st.sidebar.markdown(f"- Transactions: {len(transactions_df):,}")
st.sidebar.markdown(f"- Alerts: {len(alerts_df):,}")

# ============================================
# ALERT DASHBOARD
# ============================================
if page == "Alert Dashboard":
    st.title("AML Alert Monitoring Dashboard")
    st.markdown("Real-time monitoring of suspicious activity alerts across 10 typologies")

    # Merge alerts with risk scores
    alerts_with_scores = alerts_df.copy()
    if 'risk_score' not in alerts_with_scores.columns:
        alerts_with_scores = alerts_with_scores.merge(
            features_df[['customer_id', 'risk_score']],
            on='customer_id',
            how='left'
        )

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_alerts = len(alerts_df)
    high_risk = len(alerts_df[alerts_df['severity'] == 'high'])
    pending = len(alerts_df[alerts_df['status'] == 'open'])
    total_volume = alerts_df['total_amount'].sum()

    # Count generated SARs
    sar_files = glob.glob("outputs/sar_alert_*.json")
    sars_generated = len(sar_files)

    col1.metric("Total Alerts", f"{total_alerts:,}")
    col2.metric("High Severity", f"{high_risk:,}")
    col3.metric("Pending Review", f"{pending:,}")
    col4.metric("SARs Generated", f"{sars_generated:,}")
    col5.metric("Total Suspicious Volume", f"${total_volume:,.0f}")

    st.markdown("---")

    # Two column layout for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Risk score distribution
        st.subheader("Risk Score Distribution")

        if 'risk_score' in alerts_with_scores.columns:
            fig = px.histogram(
                alerts_with_scores,
                x='risk_score',
                nbins=20,
                labels={'risk_score': 'Risk Score', 'count': 'Number of Alerts'},
                color_discrete_sequence=['#ff4b4b']
            )
            fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                         annotation_text="High Risk Threshold")
            fig.update_layout(height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Typology distribution
        st.subheader("Alert Distribution by Typology")
        typology_counts = alerts_df['alert_type'].value_counts()

        fig = px.bar(
            x=typology_counts.index,
            y=typology_counts.values,
            labels={'x': 'Typology', 'y': 'Number of Alerts'},
            color=typology_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=350, showlegend=False, margin=dict(t=30, b=30))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Transaction volume by typology
    st.subheader("Total Suspicious Volume by Typology")
    volume_by_type = alerts_df.groupby('alert_type')['total_amount'].sum().sort_values(ascending=True)

    fig = px.bar(
        x=volume_by_type.values,
        y=volume_by_type.index,
        orientation='h',
        labels={'x': 'Total Amount ($)', 'y': 'Typology'},
        color=volume_by_type.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Top alerts table
    st.subheader("Top 20 High-Risk Alerts")

    if 'risk_score' in alerts_with_scores.columns:
        top_alerts = alerts_with_scores.nlargest(20, 'risk_score').copy()
    else:
        top_alerts = alerts_with_scores.head(20).copy()

    # Merge with customer names
    top_alerts = top_alerts.merge(
        customers_df[['customer_id', 'name']],
        on='customer_id',
        how='left'
    )

    display_cols = ['alert_id', 'name', 'alert_type', 'total_amount', 'num_transactions']
    if 'risk_score' in top_alerts.columns:
        top_alerts['risk_pct'] = (top_alerts['risk_score'] * 100).round(1)
        display_cols.append('risk_pct')

    display_cols.append('status')

    display_df = top_alerts[display_cols].copy()
    display_df.columns = ['Alert ID', 'Customer', 'Typology', 'Total Amount ($)',
                          '# Transactions'] + \
                         (['Risk Score (%)'] if 'risk_pct' in top_alerts.columns else []) + \
                         ['Status']

    st.dataframe(display_df, hide_index=True, use_container_width=True)


# ============================================
# GENERATE SAR
# ============================================
elif page == "Generate SAR":
    st.title("SAR Narrative Generator")
    st.markdown("AI-powered Suspicious Activity Report generation with full audit trail")

    # Alert selector
    col1, col2 = st.columns([1, 2])

    with col1:
        alert_id = st.number_input(
            "Select Alert ID",
            min_value=0,
            max_value=len(alerts_df)-1,
            value=0
        )

    # Show alert preview
    alert_preview = alerts_df[alerts_df['alert_id'] == alert_id]
    if len(alert_preview) > 0:
        alert_row = alert_preview.iloc[0]
        cust = customers_df[customers_df['customer_id'] == alert_row['customer_id']]

        with col2:
            if len(cust) > 0:
                cust_row = cust.iloc[0]
                st.markdown(f"**Customer:** {cust_row['name']} | "
                           f"**Typology:** {alert_row['alert_type'].replace('_', ' ').title()} | "
                           f"**Amount:** ${alert_row['total_amount']:,.2f}")

    # Generate button
    if st.button("Generate SAR Narrative", type="primary"):
        with st.spinner("Analyzing alert data and generating SAR narrative with Mistral 7B..."):
            try:
                # Load alert data for display
                alert, customer, txns = load_alert_data(alert_id)

                # Display alert info
                st.subheader("Alert Information")
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                info_col1.metric("Customer", customer['name'])
                info_col2.metric("Typology", alert['alert_type'].replace('_', ' ').title())
                info_col3.metric("Total Amount", f"${alert['total_amount']:,.2f}")
                info_col4.metric("Transactions", f"{alert['num_transactions']}")

                # Generate narrative
                narrative, audit_trail = generate_sar_narrative(
                    alert_id, vectorstore, llm, features_df, explainer, feature_cols
                )

                # Validate
                compliance_check = validate_sar_compliance(narrative)

                # Save
                sar_document = save_sar_and_audit_trail(
                    alert_id, narrative, audit_trail, compliance_check
                )

                # Display narrative
                st.subheader("Generated SAR Narrative")

                if compliance_check['compliant']:
                    st.success(f"SAR is COMPLIANT - {compliance_check['sections_found']}/{compliance_check['sections_total']} sections | {compliance_check['word_count']} words")
                else:
                    issues = []
                    if compliance_check['missing_sections']:
                        issues.append(f"Missing: {', '.join(compliance_check['missing_sections'])}")
                    if not compliance_check['word_count_ok']:
                        issues.append(f"Word count: {compliance_check['word_count']}")
                    st.warning(f"Review needed - {' | '.join(issues)}")

                st.text_area(
                    "SAR Narrative (Editable)",
                    value=narrative,
                    height=500,
                    key="sar_narrative"
                )

                # Compliance checklist
                st.subheader("Compliance Checklist")
                check_col1, check_col2 = st.columns(2)

                with check_col1:
                    st.write("**Structural Requirements:**")
                    sections_ok = "PASS" if compliance_check['has_all_sections'] else "FAIL"
                    st.write(f"- All 5W+H sections present: {sections_ok}")
                    word_ok = "PASS" if compliance_check['word_count_ok'] else "FAIL"
                    st.write(f"- Word count ({compliance_check['word_count']}): {word_ok}")

                with check_col2:
                    st.write("**Content Requirements:**")
                    amt_ok = "PASS" if compliance_check['has_specific_amounts'] else "FAIL"
                    st.write(f"- Specific dollar amounts: {amt_ok}")
                    date_ok = "PASS" if compliance_check['has_specific_dates'] else "FAIL"
                    st.write(f"- Specific dates: {date_ok}")

                # SHAP Risk Drivers
                st.subheader("Risk Drivers (SHAP Analysis)")
                if audit_trail.get('shap_top_features'):
                    shap_df = pd.DataFrame(audit_trail['shap_top_features'])
                    if len(shap_df) > 0:
                        shap_df['feature_display'] = shap_df['feature'].str.replace('_', ' ').str.title()
                        fig = px.bar(
                            shap_df,
                            x='shap_value',
                            y='feature_display',
                            orientation='h',
                            labels={'shap_value': 'SHAP Value (Impact on Risk)', 'feature_display': 'Feature'},
                            color='shap_value',
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                # Audit trail
                with st.expander("View Full Audit Trail", expanded=False):
                    # Remove the narrative from display (it's very long)
                    audit_display = {k: v for k, v in audit_trail.items() if k != 'narrative'}
                    st.json(audit_display)

                # Download buttons
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="Download Narrative (.txt)",
                        data=narrative,
                        file_name=f"sar_narrative_{alert_id}.txt",
                        mime="text/plain"
                    )
                with dl_col2:
                    sar_doc = {
                        'narrative': narrative,
                        'audit_trail': audit_trail,
                        'compliance_check': compliance_check
                    }
                    st.download_button(
                        label="Download Full SAR (.json)",
                        data=json.dumps(sar_doc, indent=2, ensure_ascii=False),
                        file_name=f"sar_complete_{alert_id}.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Error generating SAR: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# ============================================
# AUDIT TRAIL VIEWER
# ============================================
elif page == "Audit Trail Viewer":
    st.title("SAR Audit Trail Viewer")
    st.markdown("Complete transparency into SAR generation process - data lineage, model decisions, and compliance validation")

    # Load existing SARs
    sar_files = sorted(glob.glob("outputs/sar_alert_*.json"))

    if not sar_files:
        st.info("No SARs generated yet. Go to 'Generate SAR' to create your first one!")
    else:
        # Dropdown to select SAR
        sar_options = {}
        for f in sar_files:
            alert_num = os.path.basename(f).replace('sar_alert_', '').replace('.json', '')
            sar_options[f"Alert {alert_num}"] = f

        selected = st.selectbox("Select SAR to View", options=list(sar_options.keys()))

        if selected:
            with open(sar_options[selected], 'r', encoding='utf-8') as f:
                sar_doc = json.load(f)

            audit = sar_doc.get('audit_trail', {})
            compliance = sar_doc.get('compliance_check', {})

            # Metadata row
            st.subheader("SAR Metadata")
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            meta_col1.metric("Alert ID", audit.get('alert_id', 'N/A'))
            meta_col2.metric("Risk Score", f"{audit.get('risk_score', 0):.1%}")
            meta_col3.metric("Word Count", audit.get('narrative_word_count', 'N/A'))
            meta_col4.metric("Status", audit.get('status', 'N/A').title())

            st.markdown("---")

            # Two-column layout
            left_col, right_col = st.columns(2)

            with left_col:
                # Data lineage
                st.subheader("Data Lineage")
                st.write(f"**Customer ID:** {audit.get('customer_id', 'N/A')}")
                st.write(f"**Customer Name:** {audit.get('customer_name', 'N/A')}")
                st.write(f"**Transactions Analyzed:** {audit.get('num_transactions_analyzed', 'N/A')}")
                st.write(f"**Date Range:** {audit.get('transaction_date_range', 'N/A')}")

                # Model decision
                st.subheader("Model Decision")
                st.write(f"**Typology Detected:** {audit.get('typology_detected', 'N/A')}")
                st.write(f"**Risk Score:** {audit.get('risk_score', 0):.2%}")
                st.write(f"**Model:** {audit.get('model', 'N/A')}")
                st.write(f"**Temperature:** {audit.get('temperature', 'N/A')}")

            with right_col:
                # RAG retrieval
                st.subheader("Template Retrieval (RAG)")
                templates = audit.get('templates_retrieved', [])
                if templates:
                    for i, t in enumerate(templates):
                        st.write(f"**Template {i+1}:** {t.get('typology', 'unknown')} (ID: {t.get('template_id', 'N/A')})")
                else:
                    st.write("No templates recorded")

                # SHAP features
                st.subheader("Top SHAP Features")
                shap_feats = audit.get('shap_top_features', [])
                if shap_feats:
                    for feat in shap_feats[:5]:
                        feat_name = feat.get('feature', '').replace('_', ' ').title()
                        shap_val = feat.get('shap_value', 0)
                        st.write(f"- **{feat_name}:** {shap_val:+.3f}")
                else:
                    st.write("No SHAP data recorded")

                # Generation info
                st.subheader("LLM Generation")
                st.write(f"**Timestamp:** {audit.get('generation_timestamp', 'N/A')}")
                gen_params = audit.get('generation_params', {})
                st.write(f"**Temperature:** {gen_params.get('temperature', 'N/A')}")
                st.write(f"**Max Tokens:** {gen_params.get('max_tokens', 'N/A')}")

            st.markdown("---")

            # Compliance
            st.subheader("Compliance Validation")
            if compliance.get('compliant'):
                st.success(f"SAR meets FinCEN requirements - {compliance.get('sections_found', 0)}/{compliance.get('sections_total', 8)} sections, {compliance.get('word_count', 0)} words")
            else:
                missing = compliance.get('missing_sections', [])
                st.warning(f"Review required - Missing sections: {', '.join(missing) if missing else 'None'}")

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.write(f"- All sections present: {'PASS' if compliance.get('has_all_sections') else 'FAIL'}")
                st.write(f"- Word count acceptable: {'PASS' if compliance.get('word_count_ok') else 'FAIL'}")
            with comp_col2:
                st.write(f"- Specific amounts: {'PASS' if compliance.get('has_specific_amounts') else 'FAIL'}")
                st.write(f"- Specific dates: {'PASS' if compliance.get('has_specific_dates') else 'FAIL'}")

            st.markdown("---")

            # Narrative
            st.subheader("Generated Narrative")
            st.text_area(
                "SAR Narrative",
                value=sar_doc.get('narrative', ''),
                height=500,
                disabled=True
            )

            # Raw JSON
            with st.expander("View Raw Audit Trail JSON"):
                st.json(audit)
