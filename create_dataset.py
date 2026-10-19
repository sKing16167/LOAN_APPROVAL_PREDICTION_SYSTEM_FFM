import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_loan_dataset(n_samples=4269):
    """
    Generates a realistic Financial Loan Approval Prediction Dataset
    modeled after the standard Kaggle/Banking Credit Risk Benchmark.
    """
    loan_ids = [f"LP_{i+1:05d}" for i in range(n_samples)]
    
    # Demographics and profile
    dependents = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05])
    education = np.random.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(["No", "Yes"], size=n_samples, p=[0.80, 0.20])
    
    # Financial incomes (in INR, yearly)
    # Graduates tend to have slightly higher median income
    base_income = np.random.lognormal(mean=14.5, sigma=0.6, size=n_samples)
    income_annum = np.where(education == "Graduate", base_income * 1.25, base_income)
    income_annum = np.clip(np.round(income_annum, -3), 200000, 15000000).astype(int)
    
    # Loan amount requested (correlated with income)
    loan_multiplier = np.random.uniform(1.2, 4.5, size=n_samples)
    loan_amount = np.clip(np.round(income_annum * loan_multiplier, -3), 100000, 45000000).astype(int)
    
    # Loan tenure in years (e.g. 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    loan_term = np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], size=n_samples, 
                                 p=[0.05, 0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05])
    
    # CIBIL Score: 300 to 900
    cibil_score = np.clip(np.random.normal(loc=630, scale=140, size=n_samples), 300, 900).astype(int)
    
    # Asset Values (Residential, Commercial, Luxury, Bank)
    # Asset wealth correlates with income
    res_mult = np.random.uniform(0.5, 4.0, size=n_samples)
    residential_assets_value = np.clip(np.round(income_annum * res_mult, -3), 0, 30000000).astype(int)
    
    # Commercial assets (many have 0 or moderate)
    has_comm = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    comm_mult = np.random.uniform(0.3, 3.0, size=n_samples) * has_comm
    commercial_assets_value = np.clip(np.round(income_annum * comm_mult, -3), 0, 25000000).astype(int)
    
    # Luxury assets (vehicles, gold, lifestyle)
    lux_mult = np.random.uniform(0.4, 2.5, size=n_samples)
    luxury_assets_value = np.clip(np.round(income_annum * lux_mult, -3), 100000, 20000000).astype(int)
    
    # Bank assets (deposits, liquidity)
    bank_mult = np.random.uniform(0.3, 2.0, size=n_samples)
    bank_asset_value = np.clip(np.round(income_annum * bank_mult, -3), 50000, 18000000).astype(int)
    
    # Calculate underwriting score for realistic ground truth loan approval
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    asset_coverage_ratio = total_assets / (loan_amount + 1)
    dti_ratio = (loan_amount / loan_term) / (income_annum + 1)
    
    # Latent credit score
    # CIBIL is key determinant in banking (>550-600 is usually cutoff), plus asset coverage
    z = (
        (cibil_score - 580) / 75.0 * 2.8 +
        (asset_coverage_ratio - 1.2) * 1.5 -
        (dti_ratio - 0.4) * 2.2 +
        np.where(education == "Graduate", 0.35, -0.2) +
        np.where(self_employed == "Yes", -0.2, 0.1) -
        (dependents * 0.08) +
        np.random.normal(0, 0.65, size=n_samples) # stochastic bank underwriter variance
    )
    
    prob_approved = 1.0 / (1.0 + np.exp(-z))
    loan_status = np.where(prob_approved >= 0.50, "Approved", "Rejected")
    
    df = pd.DataFrame({
        "loan_id": loan_ids,
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "loan_status": loan_status
    })
    
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "loan_approval_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated with {len(df)} records at {csv_path}")
    print("Class distribution:")
    print(df["loan_status"].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    generate_loan_dataset()
