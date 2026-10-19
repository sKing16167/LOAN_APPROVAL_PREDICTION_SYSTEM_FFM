import pandas as pd
import numpy as np

def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data and extracts high-impact financial ratios.
    """
    df = df_input.copy()
    
    # Strip whitespace from column names and string values if any
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Derived Financial Ratios
    # 1. Total Collateral Assets
    df["total_assets_value"] = (
        df["residential_assets_value"] + 
        df["commercial_assets_value"] + 
        df["luxury_assets_value"] + 
        df["bank_asset_value"]
    )
    
    # 2. Loan to Income Ratio (LTI) - Leverage indicator
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1.0)
    
    # 3. Asset Coverage Ratio (ACR) - Solvency indicator (Collateral / Loan)
    df["asset_coverage_ratio"] = df["total_assets_value"] / (df["loan_amount"] + 1.0)
    
    # 4. Estimated Annual Debt Service & Debt-to-Income (DTI)
    df["annual_debt_service"] = df["loan_amount"] / df["loan_term"]
    df["debt_to_income_ratio"] = df["annual_debt_service"] / (df["income_annum"] + 1.0)
    
    # 5. Liquid Asset Ratio (Bank / Total Assets)
    df["liquidity_ratio"] = df["bank_asset_value"] / (df["total_assets_value"] + 1.0)
    
    return df

def encode_categoricals(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features into binary numerical values.
    """
    df = df_input.copy()
    
    if "education" in df.columns:
        df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0}).fillna(0).astype(int)
        
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0}).fillna(0).astype(int)
        
    return df

FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets_value",
    "loan_to_income_ratio",
    "asset_coverage_ratio",
    "debt_to_income_ratio",
    "liquidity_ratio"
]

FEATURE_LABELS = {
    "no_of_dependents": "Dependents Count",
    "education": "Education (Graduate=1)",
    "self_employed": "Self Employed",
    "income_annum": "Annual Income (INR)",
    "loan_amount": "Loan Amount (INR)",
    "loan_term": "Loan Term (Years)",
    "cibil_score": "CIBIL / Credit Score",
    "residential_assets_value": "Residential Asset (INR)",
    "commercial_assets_value": "Commercial Asset (INR)",
    "luxury_assets_value": "Luxury Asset (INR)",
    "bank_asset_value": "Bank Asset (INR)",
    "total_assets_value": "Total Collateral Assets (INR)",
    "loan_to_income_ratio": "Loan-to-Income (LTI)",
    "asset_coverage_ratio": "Asset Coverage Ratio (ACR)",
    "debt_to_income_ratio": "Debt-to-Income (DTI)",
    "liquidity_ratio": "Liquidity Ratio"
}
