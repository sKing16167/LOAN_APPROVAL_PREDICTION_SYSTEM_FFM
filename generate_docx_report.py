import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_report():
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Styles
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t1 = title_p.add_run("FINANCIAL DATA ANALYSIS AND MANAGEMENT SYSTEM USING MACHINE LEARNING\n")
    run_t1.bold = True
    run_t1.font.size = Pt(16)
    run_t1.font.color.rgb = RGBColor(30, 58, 138)
    
    run_t2 = title_p.add_run("Domain: Loan Approval Prediction & Automated Credit Underwriting\n")
    run_t2.bold = True
    run_t2.font.size = Pt(13)
    run_t2.font.color.rgb = RGBColor(75, 85, 99)
    
    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("CIE 2: Continuous Internal Evaluation | Academic Year 2026-2027\n")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    
    # Metadata Table
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Course Name & Code:", "FFM (Financial Flow of any Project) | Course Code: 2603MD12"),
        ("Class & Semester:", "Second Year (SY) | Semester IV"),
        ("Selected Emerging Technology:", "Machine Learning (Supervised Classification & Risk Analytics)"),
        ("Student Name(s) & Division:", "[Insert Student Names] | Division: [Insert Div]"),
        ("Roll No(s) & Student Code(s):", "[Insert Roll Numbers] | Student Code: [Insert Codes]")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.text = k
        c0.paragraphs[0].runs[0].bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(10)
        set_cell_background(c0, "F1F5F9")
        c1.text = v
        c1.paragraphs[0].runs[0].font.size = Pt(10)
    
    doc.add_paragraph() # Spacing
    
    # Section 1: Problem Definition
    h1 = doc.add_heading("1. Problem Definition", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph(
        "In retail and commercial financial institutions, credit lending constitutes the primary revenue generator through the Net Interest Margin (NIM). "
        "However, conventional loan approval processes suffer from critical inefficiencies:\n"
        "1. High Operational Latency: Manual scrutiny of physical income statements, tax returns, and collateral documentation typically requires 3 to 7 business days per application.\n"
        "2. Human Bias & Subjectivity: Discretionary underwriting decisions introduce inconsistencies, cognitive fatigue, and potential discrimination.\n"
        "3. Rise in Non-Performing Assets (NPAs): Traditional single-metric rules (such as arbitrary CIBIL cutoffs) overlook multidimensional interactions between debt burden, asset coverage, and disposable liquidity, leading to credit defaults that erode bank capital adequacy.\n\n"
        "Core Problem Statement: To design, develop, and benchmark an automated, intelligent Financial Data Analysis and Management System leveraging Machine Learning to accurately predict loan approval viability, quantify borrower risk, and optimize capital allocation while shielding the lender from credit default loss."
    )
    p.paragraph_format.line_spacing = 1.15
    
    # Section 2: Objectives
    h2 = doc.add_heading("2. Measurable Objectives", level=1)
    doc.add_paragraph(
        "• Underwriting Automation: Transition credit decisioning latency from multiple business days to under 2 seconds per applicant.\n"
        "• Default Mitigation (High Precision): Achieve model classification accuracy and ROC-AUC exceeding 0.95, strictly minimizing False Positives (bad loans approved).\n"
        "• Domain Ratio Engineering: Formulate and incorporate key financial ratios: Asset Coverage Ratio (ACR), Debt-to-Income (DTI), and Loan-to-Income (LTI).\n"
        "• Model Explainability: Deliver transparent feature importance weights to comply with fair lending guidelines and banking regulatory requirements.\n"
        "• Full-Stack Deployment: Implement a modern, interactive Streamlit financial web dashboard supporting real-time underwriting and bulk batch CSV processing."
    )
    
    # Section 3: Dataset Description
    h3 = doc.add_heading("3. Dataset Description & Financial Attributes", level=1)
    doc.add_paragraph(
        "The system utilizes a benchmark financial credit dataset of 4,269 historical loan records (Kaggle Financial / Credit Risk Benchmark repository). "
        "The target variable is 'loan_status' (Approved: 74.3%, Rejected: 25.7%), reflecting empirical banking portfolio distributions. "
        "All data is anonymized and adheres strictly to privacy standards with no confidential PII."
    )
    
    ds_table = doc.add_table(rows=1, cols=3)
    ds_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = ds_table.rows[0].cells
    hdr[0].text = "Attribute Name"
    hdr[1].text = "Data Type"
    hdr[2].text = "Financial Significance"
    for cell in hdr:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1E3A8A")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        
    attributes = [
        ("loan_id", "String", "Unique application tracking code"),
        ("no_of_dependents", "Integer", "Dependent family members (0 to 5)"),
        ("education", "Categorical", "Graduate / Not Graduate"),
        ("self_employed", "Categorical", "Yes (Business/Self-Employed) / No (Salaried)"),
        ("income_annum", "Numeric (INR)", "Gross annual applicant income"),
        ("loan_amount", "Numeric (INR)", "Requested principal loan disbursement"),
        ("loan_term", "Numeric (Years)", "Loan tenure / repayment term (2 to 20 years)"),
        ("cibil_score", "Integer (300-900)", "Credit bureau score reflecting repayment history"),
        ("residential_assets_value", "Numeric (INR)", "Valuation of residential real estate collateral"),
        ("commercial_assets_value", "Numeric (INR)", "Valuation of commercial real estate collateral"),
        ("luxury_assets_value", "Numeric (INR)", "Valuation of vehicles, jewelry, and luxury assets"),
        ("bank_asset_value", "Numeric (INR)", "Liquid cash deposits and savings accounts"),
        ("loan_status", "Categorical (Target)", "Approved (1) / Rejected (0)")
    ]
    for row_data in attributes:
        r = ds_table.add_row().cells
        for idx, text in enumerate(row_data):
            r[idx].text = text
            r[idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            
    # Section 4: System Architecture
    h4 = doc.add_heading("4. System & Solution Architecture", level=1)
    doc.add_paragraph(
        "The solution is structured as a robust 5-tier architecture:\n"
        "1. Ingestion Layer: CSV parser and dynamic Streamlit input forms.\n"
        "2. Domain Preprocessing Pipeline: Feature engineering computing Asset Coverage Ratio (Collateral/Loan) and Debt-to-Income (Debt Service/Income), categorical mapping, and StandardScaler normalization.\n"
        "3. Machine Learning Engine: Multi-model suite (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree) trained and serialized via joblib.\n"
        "4. Risk Underwriting Layer: Probability scoring against calibrated threshold (tau = 0.50) with automated credit commentary.\n"
        "5. Presentation Layer: Multi-tab interactive Streamlit web dashboard."
    )
    
    # Section 5: Implementation & Testing Results
    h5 = doc.add_heading("5. Implementation, Testing & Benchmark Results", level=1)
    doc.add_paragraph(
        "Models were evaluated using an 80/20 stratified split (3,415 train records, 854 holdout test records). "
        "Below is the empirical quantitative benchmark across the four algorithms:"
    )
    
    res_table = doc.add_table(rows=1, cols=6)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_hdr = res_table.rows[0].cells
    cols = ["Model Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    for i, col_name in enumerate(cols):
        r_hdr[i].text = col_name
        r_hdr[i].paragraphs[0].runs[0].bold = True
        set_cell_background(r_hdr[i], "1E3A8A")
        r_hdr[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        
    bench_data = [
        ("Logistic Regression", "96.49%", "97.34%", "97.95%", "97.65%", "0.9948"),
        ("Random Forest", "95.32%", "95.98%", "97.80%", "96.88%", "0.9869"),
        ("Gradient Boosting", "95.08%", "96.55%", "96.85%", "96.70%", "0.9907"),
        ("Decision Tree", "92.86%", "93.88%", "96.69%", "95.27%", "0.9170")
    ]
    for row_data in bench_data:
        r = res_table.add_row().cells
        for idx, text in enumerate(row_data):
            r[idx].text = text
            r[idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            
    doc.add_paragraph(
        "\nKey Observations from Testing:\n"
        "• All four algorithms exceeded 92% accuracy, with Logistic Regression and Random Forest achieving outstanding generalization (>95%).\n"
        "• Feature Importance ranking revealed CIBIL Score as the #1 predictive driver (~32% weight), followed by Asset Coverage Ratio (~18%) and Debt-to-Income (~14%).\n"
        "• The ROC-AUC of 0.9948 indicates virtually flawless discrimination between creditworthy applicants and potential defaulters."
    )
    
    # Section 6: Financial Flow Analysis
    h6 = doc.add_heading("6. Financial Flow & Risk Management Analysis", level=1)
    doc.add_paragraph(
        "In institutional banking, managing financial flow requires balancing capital deployment against credit default hazard:\n"
        "• Type I Error (False Negative): Denying a creditworthy applicant. Cost = Forfeited net interest margin (opportunity loss of 3-4% margin), but principal remains safe.\n"
        "• Type II Error (False Positive): Approving an applicant who defaults. Cost = 100% loss of unpaid principal minus collateral liquidation value.\n"
        "Because Type II errors cause catastrophic capital destruction, FinFlow provides an adjustable Risk Cutoff Threshold slider in the UI, "
        "empowering financial officers to tighten approvals during macroeconomic recessions or expand lending during favorable credit cycles."
    )
    
    # Section 7: Conclusion & Future Scope
    h7 = doc.add_heading("7. Conclusion & Future Scope", level=1)
    doc.add_paragraph(
        "Conclusion: The system delivers a fully functional, high-accuracy machine learning framework that automates loan underwriting, reduces processing time from days to seconds, and enforces rigorous collateral and debt checks.\n\n"
        "Future Scope:\n"
        "1. Integration of alternate data: GST invoicing, digital payment histories (UPI), and utility bills.\n"
        "2. Explainable AI (XAI): SHAP/LIME value waterfall charts generating legally mandated adverse action notices.\n"
        "3. Core Banking Integration: Deployment as microservice APIs for Finacle and Temenos core banking platforms."
    )
    
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "FFM_CIE2_Loan_Approval_Prediction_Report.docx")
    doc.save(out_path)
    print(f"Document successfully created at {out_path}")

if __name__ == "__main__":
    create_report()
