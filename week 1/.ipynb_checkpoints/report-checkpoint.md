# Week 1: Problem Definition & Dataset Exploration

## 1. Problem Statement
Loan default prediction is a classic and highly impactful problem in retail banking. When borrowers default on their loans, financial institutions face direct financial losses. Conversely, rejecting creditworthy borrowers limits interest revenue. The objective of this project is to analyze historical borrower data and build a classification model to predict whether a borrower will default on a loan (`Default = 1`) or not (`Default = 0`).

By leveraging machine learning, lenders can systematically identify high-risk applicants, optimize interest rates, establish co-signer requirements, and design tailored repayment programs, thereby minimizing credit risk and maximizing returns.

---

## 2. Dataset Summary

The dataset `Loan_default.csv` consists of **255,347** entries (rows) and **18** features (columns). 

### Feature Inventory
Here is the description of the columns present in the dataset:

| # | Column Name | Data Type | Description |
|---|-------------|-----------|-------------|
| 0 | `LoanID` | Object | Unique identifier for each loan transaction. |
| 1 | `Age` | Integer | Age of the borrower (18 to 69 years). |
| 2 | `Income` | Integer | Annual income of the borrower (in USD). |
| 3 | `LoanAmount` | Integer | Capital amount borrowed (in USD). |
| 4 | `CreditScore` | Integer | Creditworthiness rating (300 to 849). |
| 5 | `MonthsEmployed` | Integer | Duration of borrower's employment in months. |
| 6 | `NumCreditLines` | Integer | Number of open credit lines (1 to 4). |
| 7 | `InterestRate` | Float | Percentage rate of interest on the loan (2.0% to 25.0%). |
| 8 | `LoanTerm` | Integer | Duration of the loan (12, 24, 36, 48, or 60 months). |
| 9 | `DTIRatio` | Float | Debt-to-Income ratio (0.1 to 0.9). |
| 10 | `Education` | Object | Highest educational level attained (High School, Bachelor's, Master's, PhD). |
| 11 | `EmploymentType` | Object | Employment category (Full-time, Part-time, Self-employed, Unemployed). |
| 12 | `MaritalStatus` | Object | Marital status (Single, Married, Divorced). |
| 13 | `HasMortgage` | Object | Mortgage ownership indicator (Yes / No). |
| 14 | `HasDependents` | Object | Dependents indicator (Yes / No). |
| 15 | `LoanPurpose` | Object | Purpose of the loan (Auto, Business, Education, Home, Other). |
| 16 | `HasCoSigner` | Object | Co-signer presence indicator (Yes / No). |
| 17 | `Default` | Integer | Target variable (0 = Non-default, 1 = Defaulted). |

---

## 3. Dataset Exploration (20 Questions & Answers)

Here are the answers to the 20 analytical questions formulated to explore the data, as implemented in [exploration.ipynb](file:///d:/Darshan/Sem5/ML/Project/week%201/exploration.ipynb):

### **Q1: What is the total number of records and features in the dataset?**
* **Answer:** There are **255,347** records and **18** columns (17 features + 1 target).

### **Q2: What are the data types of each feature?**
* **Answer:** 
  * `Object` (Categorical/IDs): 8 columns (`LoanID`, `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`)
  * `Integer` (Discrete/Continuous): 8 columns (`Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `LoanTerm`, `Default`)
  * `Float` (Continuous): 2 columns (`InterestRate`, `DTIRatio`)

### **Q3: How many null (missing) values are there in each column?**
* **Answer:** There are **0** null values across all columns. The dataset is complete.

### **Q4: What is the proportion of loan defaults in the dataset (class balance/imbalance)?**
* **Answer:** 
  * **Non-Default (0):** 225,694 records (**88.39%**)
  * **Default (1):** 29,653 records (**11.61%**)
  * *Observation:* The target class is highly imbalanced. Evaluating models using accuracy alone would lead to misleading conclusions. Instead, we must track F1-Score and Precision-Recall AUC.

### **Q5: How many borrowers have a co-signer (HasCoSigner)?**
* **Answer:** 
  * **Yes (Co-signer present):** 127,701 borrowers (**50.01%**)
  * **No (No co-signer):** 127,646 borrowers (**49.99%**)

### **Q6: What is the distribution of the target variable Default across borrowers with vs without a co-signer?**
* **Answer:**
  * **Borrowers without a co-signer:** **12.87%** defaulted (16,423 defaults out of 127,646)
  * **Borrowers with a co-signer:** **10.36%** defaulted (13,230 defaults out of 127,701)
  * *Observation:* Co-signed loans show a ~2.5% lower default rate, confirming the risk-mitigating effect of a co-signer.

### **Q7: What are the unique categories in LoanPurpose and their counts?**
* **Answer:** 
  * `Business`: 51,298 (20.09%)
  * `Home`: 51,286 (20.08%)
  * `Education`: 51,005 (19.97%)
  * `Other`: 50,914 (19.94%)
  * `Auto`: 50,844 (19.91%)
  * *Observation:* The loan purposes are very evenly distributed.

### **Q8: What is the distribution of education levels among borrowers?**
* **Answer:** 
  * `Bachelor's`: 64,366 (25.21%)
  * `High School`: 63,903 (25.03%)
  * `Master's`: 63,541 (24.88%)
  * `PhD`: 63,537 (24.88%)

### **Q9: What are the minimum, maximum, mean, and median values of the Age feature?**
* **Answer:**
  * **Minimum:** 18.0 years
  * **Maximum:** 69.0 years
  * **Mean:** 43.50 years
  * **Median:** 43.0 years

### **Q10: What is the average income of borrowers who defaulted compared to those who did not?**
* **Answer:**
  * **Non-Defaulted (0):** $83,899.17
  * **Defaulted (1):** $71,844.72
  * *Observation:* Defaulters have a significantly lower average income (by $12,054.44, or -14.37%) compared to non-defaulters.

### **Q11: How many unique employment types are there in the dataset, and which one is the most common?**
* **Answer:** There are **4** unique employment types. The most common is `Part-time` with 64,161 records, though all are closely balanced:
  * `Part-time`: 64,161 (25.13%)
  * `Unemployed`: 63,824 (25.00%)
  * `Self-employed`: 63,706 (24.95%)
  * `Full-time`: 63,656 (24.93%)

### **Q12: What is the distribution of LoanTerm values?**
* **Answer:** Loans are issued in terms of 12, 24, 36, 48, and 60 months, and are evenly spread:
  * `48 months`: 51,166 (20.04%)
  * `60 months`: 51,154 (20.03%)
  * `36 months`: 51,061 (20.00%)
  * `24 months`: 51,009 (19.98%)
  * `12 months`: 50,957 (19.96%)

### **Q13: Is there any clear correlation between CreditScore and InterestRate?**
* **Answer:** The Pearson correlation coefficient is **-0.0020**, indicating **virtually zero correlation** in the raw data. 

### **Q14: What are the statistics (mean, median, std) of the LoanAmount feature?**
* **Answer:**
  * **Mean:** $127,578.87
  * **Median:** $127,556.00
  * **Standard Deviation:** $70,840.71

### **Q15: What proportion of borrowers have a mortgage (HasMortgage)?**
* **Answer:**
  * **Yes (Have Mortgage):** 127,677 borrowers (**50.00%**)
  * **No (Do not have Mortgage):** 127,670 borrowers (**50.00%**)

### **Q16: How many borrowers have dependents (HasDependents)?**
* **Answer:**
  * **Yes (Have Dependents):** 127,742 borrowers (**50.03%**)
  * **No (Do not have Dependents):** 127,605 borrowers (**49.97%**)

### **Q17: What is the distribution of the number of credit lines (NumCreditLines)?**
* **Answer:**
  * `1 line`: 63,554 borrowers (24.89%)
  * `2 lines`: 64,130 borrowers (25.12%)
  * `3 lines`: 63,834 borrowers (24.99%)
  * `4 lines`: 63,829 borrowers (24.99%)

### **Q18: What is the average Debt-to-Income (DTI) ratio (DTIRatio) for defaulted vs non-defaulted loans?**
* **Answer:**
  * **Non-Defaulted (0):** 0.4975
  * **Defaulted (1):** 0.5208
  * *Observation:* Borrowers who default have an average DTI ratio that is **4.68% higher** than those who do not default.

### **Q19: What is the distribution of employment length in months (MonthsEmployed)?**
* **Answer:**
  * **Minimum:** 0 months
  * **Maximum:** 119 months
  * **Mean:** 59.54 months
  * **Median:** 60.00 months
  * **Standard Deviation:** 34.64 months

### **Q20: Are there any duplicate records in the dataset?**
* **Answer:** 
  * There are **0** complete duplicate rows.
  * There are **0** duplicate `LoanID` entries. Every row represents a unique loan.

---

## 4. Initial Observations & Next Steps

1. **High Quality, Clean Data:** The dataset does not require complex imputations or deduplications since it contains zero missing values and zero duplicate entries.
2. **Predictors of Default:** Preliminary exploration suggests that `Income`, `DTIRatio`, and `HasCoSigner` are promising predictors of default. In Week 2, we will explore this further via bivariate boxplots and correlation matrices.
3. **Preprocessing Plan for Week 2:**
   - Numerical columns appear bounded and uniform/normal; we will run standard outlier detection (IQR) to confirm.
   - Categorical columns (such as `Education`, `EmploymentType`, `MaritalStatus`, `LoanPurpose`) will be One-Hot encoded.
   - Binary columns (`HasMortgage`, `HasDependents`, `HasCoSigner`) will be binary encoded (mapped to 0/1).
   - All numerical features will be normalized/scaled using standard scaling to ensure they are on the same scale for distance-based and gradient-based algorithms.
