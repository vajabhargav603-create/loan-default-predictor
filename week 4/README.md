   # Week 4: Loan Default Risk Prediction Web Application

An end-to-end Full-Stack Machine Learning Web Application for predicting loan defaults based on the **Loan Default Prediction** dataset (`Loan_default.csv`, 255,347 records).

Built following the **Computer Engineering Department ML Project SOP** (Week 4: Model Evaluation, Backend Integration & Web Interface).

---

## 📁 Project Architecture

```
week 4/
├── backend/
│   ├── app.py                  # Flask REST API server (Serves endpoints & web app)
│   ├── train_model.py          # Self-contained Scikit-Learn training pipeline
│   ├── model.joblib            # Trained balanced ensemble model
│   └── requirements.txt        # Python library dependencies
├── frontend/
│   ├── index.html              # Modern, responsive UI with form, gauge & insights
│   ├── style.css               # Glassmorphic dark theme CSS design system
│   └── app.js                  # Frontend controller, API client & fallback engine
├── run_app.bat                 # One-click Windows starter script
└── README.md                   # Setup and execution guide
```

---

## 🚀 How to Run the Application

You can start the entire application in two easy steps:

### Option A: Using the One-Click Batch Script (Easiest)
Simply double-click [`run_app.bat`](file:///d:/Darshan/Sem5/ML/Project/week%204/run_app.bat) in the `week 4` folder, or run:
```powershell
.\run_app.bat
```
This automatically launches the Flask server and opens `http://127.0.0.1:5000` in your web browser.

---

### Option B: Manual Command-Line Execution

#### Step 1: Open Terminal and Navigate to Backend
Open PowerShell or Command Prompt and run:
```powershell
cd "d:\Darshan\Sem5\ML\Project\week 4\backend"
```

#### Step 2: Install Dependencies (If not already installed)
```powershell
pip install -r requirements.txt
```

#### Step 3: Start the Flask Web Server
```powershell
python app.py
```

#### Step 4: Open the Website
Once you see the server running, open your browser and visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎯 Features & How to Test

1. **🎲 Fill Random Values**:
   - Click the **"🎲 Fill Random Values"** button at the top of the form.
   - All 16 fields (`Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`, `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`) will be automatically populated with realistic values sampled from the dataset distribution.

2. **⚡ Quick Risk Presets**:
   - **🛡️ Low Risk**: High income, high credit score (795), low DTI, co-signer present $\rightarrow$ predicts **Low Default Risk (Approval Recommended)**.
   - **⚠️ High Risk**: Low income, unemployed, high DTI (0.78), high interest rate (22.8%), low credit score (430) $\rightarrow$ predicts **High Default Risk (Rejection Advised)**.
   - **⚖️ Average**: Median borrower profile.

3. **📊 Live Real-Time Prediction**:
   - Click **"Predict Loan Default Risk"**.
   - View the animated **Default Probability Meter** (0% to 100%).
   - View the **Risk Tier Badge** (Low, Moderate, High).
   - View **Estimated Monthly EMI, Total Repayment, and Interest**.
   - View **Key Contributing Risk Factors** explaining *why* the model made that decision.

4. **📈 Data Insights (EDA) Tab**:
   - Displays full dataset statistics: 255,347 records, 11.61% default rate, class balance visualization, and feature ranges.

---

## 🔄 Optional: Re-training the Model

If you ever wish to re-train the model from scratch on `Loan_default.csv`, run:
```powershell
python train_model.py
```
This will train a `HistGradientBoostingClassifier` with balanced class weights on an 80/20 train/test split and update `model.joblib`.
