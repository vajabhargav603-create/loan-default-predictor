"""
Week 4: Model Training and Evaluation Pipeline for Loan Default Prediction
Dataset: Loan_default.csv (255,347 records)
Features: 16 applicant credit and loan features
Target: Default (0: Non-default, 1: Default)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

def find_dataset_path():
    """Locate Loan_default.csv across typical paths."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'Loan_default.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Loan_default.csv'),
        os.path.join(os.path.dirname(__file__), 'Loan_default.csv'),
        os.path.join(os.getcwd(), 'Loan_default.csv'),
        os.path.join(os.getcwd(), '..', 'Loan_default.csv')
    ]
    for path in candidates:
        norm = os.path.abspath(path)
        if os.path.exists(norm):
            return norm
    raise FileNotFoundError("Could not locate Loan_default.csv. Please ensure it is in the project root directory.")

def train_and_evaluate(sample_size=None):
    dataset_path = find_dataset_path()
    print(f"[1/5] Loading dataset from: {dataset_path}")
    start_time = time.time()
    
    df = pd.read_csv(dataset_path)
    print(f"      Total records loaded: {len(df):,} | Columns: {list(df.columns)}")
    
    if sample_size and sample_size < len(df):
        print(f"      Sampling {sample_size:,} records for fast execution...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Feature definitions
    numeric_features = [
        'Age', 'Income', 'LoanAmount', 'CreditScore',
        'MonthsEmployed', 'NumCreditLines', 'InterestRate',
        'LoanTerm', 'DTIRatio'
    ]
    categorical_features = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]
    
    X = df[numeric_features + categorical_features]
    y = df['Default']
    
    default_rate = float(y.mean() * 100)
    print(f"[2/5] Target distribution: Default Rate = {default_rate:.2f}% (0: {(100-default_rate):.2f}%, 1: {default_rate:.2f}%)")
    
    # Preprocessor
    print("[3/5] Setting up preprocessing pipeline (StandardScaler + OneHotEncoder)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Model pipeline with balanced HistGradientBoostingClassifier
    classifier = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=150,
        learning_rate=0.08,
        max_depth=7,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Train/Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")
    
    print("[4/5] Training HistGradientBoostingClassifier...")
    pipeline.fit(X_train, y_train)
    
    print("[5/5] Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred) * 100
    roc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EVALUATION METRICS:")
    print("="*60)
    print(f" Accuracy : {acc:.2f}%")
    print(f" ROC-AUC  : {roc:.4f}")
    print(f" Precision: {prec:.2f}%")
    print(f" Recall   : {rec:.2f}%")
    print(f" F1-Score : {f1:.4f}")
    print(f" Confusion Matrix (TN, FP / FN, TP):\n {np.array(cm)}")
    print("="*60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Default (0)', 'Default (1)']))
    
    # Compute feature stats for random value generation
    feature_stats = {}
    for col in numeric_features:
        feature_stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std())
        }
    
    categorical_distribution = {}
    for col in categorical_features:
        vc = df[col].value_counts(normalize=True)
        categorical_distribution[col] = {
            'categories': list(vc.index),
            'probabilities': [float(p) for p in vc.values]
        }
    
    # Pick 10 representative test samples with ground truth for testing
    sample_indices = np.random.RandomState(42).choice(X_test.index, size=10, replace=False)
    test_samples = []
    for idx in sample_indices:
        sample = X_test.loc[idx].to_dict()
        sample['Default'] = int(y_test.loc[idx])
        # cast numpy types to standard python types
        for k, v in sample.items():
            if isinstance(v, (np.integer, int)):
                sample[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                sample[k] = round(float(v), 2)
        test_samples.append(sample)
    
    # Extract transformed feature names
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_transformed_features = numeric_features + cat_feature_names
    
    output_dir = os.path.dirname(__file__)
    model_path = os.path.join(output_dir, 'model.joblib')
    
    # Save trained pipeline
    joblib.dump(pipeline, model_path)
    print(f"[OK] Saved model pipeline to: {model_path}")
    print("\nTraining completed successfully! You can now run 'python app.py'.")

if __name__ == '__main__':
    train_and_evaluate()
