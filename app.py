import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_processor import engineer_features, encode_categoricals, FEATURE_COLUMNS, FEATURE_LABELS

# Streamlit Page Configuration
st.set_page_config(
    page_title="FinFlow - Loan Approval & Credit Risk Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data and artifacts
@st.cache_data
def load_dataset():
    data_path = os.path.join("data", "loan_approval_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_models_and_metrics():
    models = {}
    model_files = {
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Logistic Regression": "logistic_regression.joblib"
    }
    for name, fname in model_files.items():
        p = os.path.join("models", fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
            
    scaler_path = os.path.join("models", "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    return models, scaler, metrics

# Load global resources
raw_df = load_dataset()
models_dict, scaler, metrics_json = load_models_and_metrics()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=70)
    st.markdown("## **FinFlow Analytics**")
    st.caption("Financial Data Analysis & Management System")
    st.markdown("---")
    
    st.markdown("### 🎓 Academic Context")
    st.write("**Course:** FFM (2603MD12)")
    st.write("**Class:** SY, Semester IV")
    st.write("**Activity:** CIE 2 Project")
    st.write("**Tech:** Machine Learning")
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    active_model_name = st.selectbox(
        "Default Underwriting Model",
        options=list(models_dict.keys()) if models_dict else ["None"],
        index=0
    )
    
    decision_threshold = st.slider(
        "Risk Cutoff Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Probability threshold above which a loan is flagged as Approved."
    )
    
    st.markdown("---")
    st.info("💡 **Financial Insight:** Lower thresholds increase loan acceptance volume but escalate Default/NPL risks.")

# Header
st.markdown('<div class="main-title">🏦 Financial Data Analysis & Management System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Loan Approval Prediction, Credit Risk Scoring & Capital Flow Optimization</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_eda, tab_predictor, tab_batch, tab_benchmark, tab_docs = st.tabs([
    "📈 Executive Dashboard",
    "🔍 Exploratory Data Analysis",
    "🎯 Real-Time Underwriter",
    "📁 Batch Loan Screening",
    "🏆 Model Performance & Testing",
    "📘 Project Academic Report"
])

# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
with tab_overview:
    if raw_df is not None:
        total_apps = len(raw_df)
        approved_count = (raw_df["loan_status"].str.strip() == "Approved").sum()
        rejected_count = total_apps - approved_count
        approval_rate = (approved_count / total_apps) * 100
        
        total_requested = raw_df["loan_amount"].sum()
        approved_volume = raw_df[raw_df["loan_status"].str.strip() == "Approved"]["loan_amount"].sum()
        avg_cibil = raw_df["cibil_score"].mean()
        avg_income = raw_df["income_annum"].mean()
        
        # High Level KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Loan Applications", f"{total_apps:,}")
        kpi2.metric("Approval Rate", f"{approval_rate:.1f}%", f"{approved_count} approved")
        kpi3.metric("Approved Capital Flow", f"₹{approved_volume/1e7:.2f} Cr", f"{(approved_volume/total_requested)*100:.1f}% of demand")
        kpi4.metric("Avg Portfolio CIBIL", f"{avg_cibil:.0f} pts")
        kpi5.metric("Avg Applicant Income", f"₹{avg_income/1e5:.1f} Lakhs")
        
        st.markdown("---")
        
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.markdown("#### 🎯 Loan Approval Ratio")
            fig_pie = px.pie(
                raw_df, 
                names="loan_status", 
                hole=0.55,
                color="loan_status",
                color_discrete_map={"Approved": "#10B981", "Rejected": "#EF4444"}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
            fig_pie.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            st.markdown("#### 💳 CIBIL Credit Score Distribution by Status")
            fig_cibil = px.histogram(
                raw_df,
                x="cibil_score",
                color="loan_status",
                barmode="overlay",
                nbins=35,
                color_discrete_map={"Approved": "#10B981", "Rejected": "#EF4444"},
                labels={"cibil_score": "CIBIL Score (300 - 900)"}
            )
            fig_cibil.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cibil, use_container_width=True)
            
        col_c3, col_c4 = st.columns([1, 1])
        
        with col_c3:
            st.markdown("#### 💰 Loan Amount vs Annual Income")
            fig_scatter = px.scatter(
                raw_df.sample(min(800, len(raw_df)), random_state=42),
                x="income_annum",
                y="loan_amount",
                color="loan_status",
                size="cibil_score",
                hover_data=["education", "self_employed", "loan_term"],
                color_discrete_map={"Approved": "#10B981", "Rejected": "#EF4444"},
                labels={"income_annum": "Annual Income (₹)", "loan_amount": "Loan Amount (₹)"}
            )
            fig_scatter.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_c4:
            st.markdown("#### 🏢 Collateral Asset Composition")
            asset_sums = {
                "Residential": raw_df["residential_assets_value"].sum(),
                "Commercial": raw_df["commercial_assets_value"].sum(),
                "Luxury Assets": raw_df["luxury_assets_value"].sum(),
                "Bank Balances": raw_df["bank_asset_value"].sum()
            }
            asset_df = pd.DataFrame(list(asset_sums.items()), columns=["Asset Category", "Total Value"])
            fig_bar = px.bar(
                asset_df,
                x="Asset Category",
                y="Total Value",
                color="Asset Category",
                text_auto='.2s',
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_bar.update_layout(height=360, showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.warning("Dataset not loaded. Please verify data/loan_approval_dataset.csv exists.")

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📊 In-Depth Financial Data Exploration")
    st.write("Analyze demographic distributions, financial ratios, and credit risk correlations.")
    
    if raw_df is not None:
        # Derived ratios for EDA
        eda_df = engineer_features(raw_df)
        
        # Filter controls
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            edu_filter = st.multiselect("Education Filter", options=eda_df["education"].unique(), default=eda_df["education"].unique())
        with f_col2:
            emp_filter = st.multiselect("Employment Filter", options=eda_df["self_employed"].unique(), default=eda_df["self_employed"].unique())
        with f_col3:
            term_filter = st.slider("Loan Tenure Range (Years)", int(eda_df["loan_term"].min()), int(eda_df["loan_term"].max()), (2, 20))
            
        filtered_eda = eda_df[
            (eda_df["education"].isin(edu_filter)) &
            (eda_df["self_employed"].isin(emp_filter)) &
            (eda_df["loan_term"].between(term_filter[0], term_filter[1]))
        ]
        st.caption(f"Showing **{len(filtered_eda):,}** records matching current filters.")
        
        st.markdown("---")
        
        row1_1, row1_2 = st.columns(2)
        with row1_1:
            st.markdown("#### 🛡️ Asset Coverage Ratio (ACR) vs Loan Status")
            st.caption("Ratio of total assets to requested loan amount. Higher values indicate strong collateral safety.")
            fig_acr = px.box(
                filtered_eda,
                x="loan_status",
                y="asset_coverage_ratio",
                color="loan_status",
                points="outliers",
                color_discrete_map={"Approved": "#10B981", "Rejected": "#EF4444"}
            )
            fig_acr.update_layout(height=340)
            st.plotly_chart(fig_acr, use_container_width=True)
            
        with row1_2:
            st.markdown("#### ⚖️ Debt-to-Income (DTI) Ratio vs Loan Status")
            st.caption("Annual loan repayment liability compared to annual income. Lower values indicate repayment capacity.")
            fig_dti = px.box(
                filtered_eda,
                x="loan_status",
                y="debt_to_income_ratio",
                color="loan_status",
                points="outliers",
                color_discrete_map={"Approved": "#10B981", "Rejected": "#EF4444"}
            )
            fig_dti.update_layout(height=340)
            st.plotly_chart(fig_dti, use_container_width=True)
            
        st.markdown("#### 🔗 Correlation Heatmap of Key Financial Factors")
        numeric_cols = [
            "income_annum", "loan_amount", "loan_term", "cibil_score", 
            "total_assets_value", "loan_to_income_ratio", "asset_coverage_ratio", "debt_to_income_ratio"
        ]
        corr_matrix = eda_df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        with st.expander("📄 View Filtered Financial Records"):
            st.dataframe(filtered_eda.head(100), use_container_width=True)

# ==============================================================================
# TAB 3: REAL-TIME LOAN UNDERWRITING PREDICTOR
# ==============================================================================
with tab_predictor:
    st.markdown("### 🎯 Single-Applicant Credit Underwriting & Risk Evaluation")
    st.write("Input applicant financial details, assets, and credit score to receive an immediate ML-driven underwriting decision.")
    
    with st.form("underwriting_form"):
        st.markdown("##### 1. Applicant Profile & Loan Request")
        u_col1, u_col2, u_col3, u_col4 = st.columns(4)
        with u_col1:
            f_dependents = st.number_input("Dependents Count", min_value=0, max_value=8, value=2, step=1)
        with u_col2:
            f_education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        with u_col3:
            f_self_emp = st.selectbox("Self Employed?", options=["No", "Yes"])
        with u_col4:
            f_loan_term = st.selectbox("Loan Tenure (Years)", options=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], index=4)
            
        u_col5, u_col6, u_col7 = st.columns(3)
        with u_col5:
            f_income = st.number_input("Annual Gross Income (₹)", min_value=100000, max_value=50000000, value=1200000, step=50000)
        with u_col6:
            f_loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=50000, max_value=100000000, value=3000000, step=100000)
        with u_col7:
            f_cibil = st.slider("CIBIL Credit Score", min_value=300, max_value=900, value=740, step=5,
                                help="300-550: Poor, 550-650: Moderate, 650-750: Good, 750+: Excellent")
            
        st.markdown("##### 2. Collateral Assets & Net Worth")
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            f_res_asset = st.number_input("Residential Asset Value (₹)", min_value=0, max_value=100000000, value=2500000, step=50000)
        with a_col2:
            f_comm_asset = st.number_input("Commercial Asset Value (₹)", min_value=0, max_value=100000000, value=800000, step=50000)
        with a_col3:
            f_lux_asset = st.number_input("Luxury Assets (Vehicles/Gold) (₹)", min_value=0, max_value=50000000, value=1500000, step=50000)
        with a_col4:
            f_bank_asset = st.number_input("Bank Deposits / Liquid Cash (₹)", min_value=0, max_value=50000000, value=600000, step=25000)
            
        submit_btn = st.form_submit_button("⚡ Run Credit Underwriting Analysis", use_container_width=True)
        
    if submit_btn:
        # Assemble input dictionary
        input_data = {
            "loan_id": ["APP_LIVE"],
            "no_of_dependents": [f_dependents],
            "education": [f_education],
            "self_employed": [f_self_emp],
            "income_annum": [f_income],
            "loan_amount": [f_loan_amount],
            "loan_term": [f_loan_term],
            "cibil_score": [f_cibil],
            "residential_assets_value": [f_res_asset],
            "commercial_assets_value": [f_comm_asset],
            "luxury_assets_value": [f_lux_asset],
            "bank_asset_value": [f_bank_asset]
        }
        
        applicant_df = pd.DataFrame(input_data)
        feat_df = engineer_features(applicant_df)
        clean_df = encode_categoricals(feat_df)
        
        X_applicant = clean_df[FEATURE_COLUMNS]
        
        # Select active model
        model = models_dict.get(active_model_name)
        if model is None:
            st.error("Selected ML model is unavailable.")
        else:
            if active_model_name == "Logistic Regression" and scaler is not None:
                X_input_scaled = scaler.transform(X_applicant)
                prob_approved = model.predict_proba(X_input_scaled)[0, 1]
            else:
                prob_approved = model.predict_proba(X_applicant)[0, 1]
                
            is_approved = prob_approved >= decision_threshold
            
            # Computed financial metrics
            total_assets = clean_df["total_assets_value"].iloc[0]
            acr = clean_df["asset_coverage_ratio"].iloc[0]
            lti = clean_df["loan_to_income_ratio"].iloc[0]
            dti = clean_df["debt_to_income_ratio"].iloc[0]
            
            st.markdown("---")
            st.markdown("### 📋 Credit Underwriting Assessment Result")
            
            res_left, res_right = st.columns([1.2, 1])
            
            with res_left:
                if is_approved:
                    st.success(f"### ✅ LOAN APPROVED\n**Status:** Recommended for Approval | Model: **{active_model_name}**")
                else:
                    st.error(f"### ❌ LOAN REJECTED\n**Status:** Decline / High Credit Risk | Model: **{active_model_name}**")
                    
                st.markdown(f"""
                - **Approval Probability:** `{prob_approved * 100:.2f}%` (Threshold: `{decision_threshold*100:.0f}%`)
                - **Default Risk Level:** `{'LOW' if prob_approved > 0.75 else ('MODERATE' if prob_approved >= 0.50 else 'HIGH / CRITICAL')}`
                - **Total Collateral Security:** ₹{total_assets:,.0f}
                - **Asset Coverage Ratio (ACR):** `{acr:.2f}x` (Collateral / Loan)
                - **Debt-to-Income (DTI):** `{dti * 100:.1f}%` of annual income
                """)
                
                # Detailed Underwriter Comments
                st.markdown("##### 🔍 Financial Rationale & Recommendations:")
                if f_cibil < 580:
                    st.warning("⚠️ **Subprime Credit Bureau Score:** CIBIL is below banking risk baseline (580). Credit history demonstrates previous delays or defaults.")
                else:
                    st.info("✔️ **Strong Credit Score:** CIBIL meets prime banking approval thresholds.")
                    
                if acr < 1.0:
                    st.warning("⚠️ **Under-Collateralized:** Total asset coverage is less than requested loan amount. Lender would face shortfall upon liquidation.")
                else:
                    st.info(f"✔️ **Sufficient Collateral Buffer:** Total net assets provide {acr:.1f}x coverage over the principal.")
                    
                if dti > 0.45:
                    st.warning("⚠️ **Elevated Debt Burden:** Annual debt repayment exceeds 45% of gross annual income, indicating cash flow vulnerability.")
                    
            with res_right:
                # Gauge Chart for Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_approved * 100,
                    title={"text": "Approval Confidence Score", "font": {"size": 18}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E3A8A"},
                        'steps': [
                            {'range': [0, decision_threshold * 100], 'color': "#FCA5A5"},
                            {'range': [decision_threshold * 100, 100], 'color': "#86EFAC"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': decision_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================================================================
# TAB 4: BATCH APPLICATION SCREENING
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Loan Applications Underwriting")
    st.write("Upload a CSV file containing multiple loan applicants or load the pre-configured sample dataset to screen applicants in bulk.")
    
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload CSV containing applicant data", type=["csv"])
    with b_col2:
        st.write("Or use pre-generated test applications:")
        load_sample = st.button("📂 Load Sample Batch File (8 Applicants)")
        
    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample:
        sample_path = os.path.join("data", "sample_batch_applications.csv")
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path)
        else:
            st.error("Sample batch CSV file not found.")
            
    if batch_df is not None:
        st.markdown(f"#### 📥 Loaded Applicants ({len(batch_df)} records)")
        st.dataframe(batch_df, use_container_width=True)
        
        if st.button("🚀 Process Batch Predictions", type="primary"):
            model = models_dict.get(active_model_name)
            if model is None:
                st.error("Model not available.")
            else:
                # Process features
                batch_feat = engineer_features(batch_df)
                batch_clean = encode_categoricals(batch_feat)
                X_batch = batch_clean[FEATURE_COLUMNS]
                
                if active_model_name == "Logistic Regression" and scaler is not None:
                    X_b_scaled = scaler.transform(X_batch)
                    probs = model.predict_proba(X_b_scaled)[:, 1]
                else:
                    probs = model.predict_proba(X_batch)[:, 1]
                    
                decisions = np.where(probs >= decision_threshold, "Approved", "Rejected")
                
                results_df = batch_df.copy()
                results_df["Approval_Probability"] = np.round(probs * 100, 2)
                results_df["Predicted_Status"] = decisions
                results_df["Risk_Tier"] = np.where(
                    probs >= 0.80, "Low Risk",
                    np.where(probs >= 0.50, "Moderate Risk", "High Risk")
                )
                results_df["Asset_Coverage_Ratio"] = np.round(batch_feat["asset_coverage_ratio"], 2)
                results_df["Debt_To_Income"] = np.round(batch_feat["debt_to_income_ratio"], 2)
                
                st.markdown("---")
                st.markdown("### 📊 Underwriting Results Summary")
                
                app_c = (results_df["Predicted_Status"] == "Approved").sum()
                rej_c = len(results_df) - app_c
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Batch Size", len(results_df))
                m2.metric("Approved Applications", app_c, f"{(app_c/len(results_df))*100:.1f}%")
                m3.metric("Declined Applications", rej_c, f"{(rej_c/len(results_df))*100:.1f}%")
                
                # Styled dataframe
                st.dataframe(
                    results_df[[
                        "loan_id", "cibil_score", "income_annum", "loan_amount", 
                        "Asset_Coverage_Ratio", "Debt_To_Income", "Approval_Probability", 
                        "Predicted_Status", "Risk_Tier"
                    ]],
                    use_container_width=True
                )
                
                csv_download = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Export Batch Decisions (CSV)",
                    data=csv_download,
                    file_name="loan_underwriting_decisions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================================
# TAB 5: MODEL PERFORMANCE & BENCHMARK
# ==============================================================================
with tab_benchmark:
    st.markdown("### 🏆 Comprehensive Model Performance & Comparative Benchmark")
    st.write("Rigorous quantitative evaluation of 4 classification algorithms trained with 80/20 stratified cross-validation.")
    
    if metrics_json and "metrics_summary" in metrics_json:
        summary = metrics_json["metrics_summary"]
        
        # Comparison Table
        table_rows = []
        for model_name, m in summary.items():
            table_rows.append({
                "Model Algorithm": model_name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1 Score": f"{m['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{m['roc_auc']:.4f}"
            })
        st.table(pd.DataFrame(table_rows))
        
        st.markdown("---")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 📉 Receiver Operating Characteristic (ROC) Curves")
            fig_roc = go.Figure()
            for model_name, m in summary.items():
                fig_roc.add_trace(go.Scatter(
                    x=m["fpr"],
                    y=m["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC = {m['roc_auc']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random Baseline"
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=380,
                legend=dict(x=0.45, y=0.15)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with col_m2:
            st.markdown(f"#### 🔲 Confusion Matrix: {active_model_name}")
            cm = summary[active_model_name]["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Rejected (0)", "Actual: Approved (1)"],
                columns=["Pred: Rejected (0)", "Pred: Approved (1)"]
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual")
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 🌟 Feature Importance Ranking (Random Forest)")
        rf_fi = summary["Random Forest"]["feature_importances"]
        fi_sorted = sorted(rf_fi.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(k, k) for k, _ in fi_sorted],
            "Importance Weight": [v for _, v in fi_sorted]
        })
        fig_fi = px.bar(
            fi_df,
            x="Importance Weight",
            y="Feature",
            orientation="h",
            color="Importance Weight",
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("""
        > [!NOTE]
        > **Financial Interpretability:** The empirical results confirm that **CIBIL Score** is the single highest predictor of creditworthiness (~32% weight), followed by derived solvency ratios: **Asset Coverage Ratio (ACR)** and **Debt-to-Income (DTI)**.
        """)

# ==============================================================================
# TAB 6: PROJECT DOCUMENTATION & VIVA PREPARATION
# ==============================================================================
with tab_docs:
    st.markdown("### 📘 University Project Documentation & Academic Alignment")
    st.write("Academic documentation prepared for the FFM CIE-2 Continuous Assessment.")
    
    with st.expander("📌 1. Problem Definition & Objectives", expanded=True):
        st.markdown("""
        - **Problem Statement:** Traditional retail and commercial bank loan underwriting involves manual verification of credit bureau reports, debt capacity, and collateral valuation. This induces operational latency (3–7 days), human cognitive bias, high operational underwriting expense, and risk of approving subprime borrowers resulting in Non-Performing Assets (NPAs).
        - **Key Objectives:**
          1. **Automate Credit Decisions:** Develop an intelligent classification system to evaluate loan applications in under 2 seconds.
          2. **Minimize Bad Debt / Credit Loss:** Ensure high classification precision and ROC-AUC (>0.95) to minimize False Positives (Type II errors in banking).
          3. **Synthesize Domain Financial Ratios:** Extract crucial financial metrics such as Asset Coverage Ratio (ACR), Loan-to-Income (LTI), and Debt-to-Income (DTI) to mirror institutional underwriting practices.
          4. **Provide Full Model Explainability:** Surface feature importance weights so banking regulators and applicants can verify the objective factors governing loan approvals.
        """)
        
    with st.expander("🗂️ 2. Dataset Description & Provenance"):
        st.markdown("""
        - **Source:** Kaggle / Financial Banking Credit Risk Benchmark.
        - **Record Count:** 4,269 historical loan application records.
        - **Class Balance:** ~74% Approved, ~26% Rejected (representative of operational banking portfolios).
        - **Features:**
          - Demographic: `no_of_dependents`, `education`, `self_employed`
          - Cash Flow: `income_annum`, `loan_amount`, `loan_term`
          - Bureau Rating: `cibil_score` (300 to 900)
          - Collateral Assets: `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
          - Target: `loan_status` (Approved / Rejected)
        - **Ethical Compliance:** Anonymized synthetic data containing no personally identifiable individual records (PII) or confidential banking credentials.
        """)

    with st.expander("🏗️ 3. System Architecture & Financial Data Flow"):
        st.markdown("""
        ```mermaid
        flowchart TD
            A["Raw Loan Applications (CSV / Form)"] --> B["Data Ingestion & Validation Layer"]
            B --> C["Feature Engineering Engine<br/>(Total Assets, ACR, LTI, DTI Ratios)"]
            C --> D["StandardScaler / Categorical Encoding"]
            D --> E["Trained ML Classifiers<br/>(Random Forest, Gradient Boosting, Logistic Regression)"]
            E --> F["Risk Scoring & Threshold Engine (p >= 0.50)"]
            F --> G["Underwriting Decision & Advisory Output<br/>(Approved / Declined + Rationale)"]
            G --> H["Streamlit Executive Dashboard & Audit Reports"]
        ```
        """)

    with st.expander("🔬 4. Technology Selection Justification"):
        st.markdown("""
        - **Why Machine Learning over Static Rules?** Traditional banks use hard threshold cutoff rules (e.g., CIBIL > 650). Such rules reject creditworthy applicants with borderline CIBIL scores who possess substantial collateral coverage. ML models learn non-linear combinations of collateral buffer, income leverage, and credit behavior, maximizing lending book yield without escalating default risk.
        - **Random Forest & Gradient Boosting:** Excel at non-linear interactions and tabular data, yielding robust generalization and resilience to outliers.
        - **Logistic Regression:** Offers calibrated probabilistic output and compliance with banking fair-lending transparency regulations.
        """)
        
    with st.expander("🎓 5. Viva Voce & Defense Q&A Guide"):
        st.markdown("""
        **Q1: What is the biggest financial risk when deploying an ML model in loan approval?**
        > *Answer:* False Positives (predicting Approved for an applicant who subsequently defaults). In banking, a single default results in 100% loss of unpaid principal, whereas a False Negative only forfeits the marginal net interest income (NIM). Hence, models must be tuned with high precision and strict threshold calibration.

        **Q2: Why was CIBIL score the most prominent feature?**
        > *Answer:* CIBIL directly reflects historical debt repayment discipline, past defaults, write-offs, and credit utilization. Past financial behavior is empirically the strongest indicator of future credit default probability.

        **Q3: How do the derived ratios enhance predictive accuracy?**
        > *Answer:* Raw asset numbers alone do not indicate solvency. Asset Coverage Ratio (Collateral / Loan) normalizes applicant wealth against requested debt size, enabling models to gauge recovery prospects in liquidation events.
        """)

    st.success("🎓 University CIE-2 Academic Documentation successfully integrated into the system dashboard.")

st.markdown("---")
st.caption("Developed for FFM Course (2603MD12) | Financial Data Analysis and Management System")
