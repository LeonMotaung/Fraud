import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# 1. Feature Manifold Integration
portfolio = pd.read_csv('loan_portfolio.csv')
vintage = pd.read_csv('vintage_analysis.csv')

# Engineering 'Vintage Friction': Mapping cohorts to their historical default curves
portfolio['origination_q'] = pd.to_datetime(portfolio['origination_date']).dt.to_period('Q').astype(str)
vintage_map = vintage.groupby('vintage')['cumulative_default_rate'].max().to_dict()
portfolio['vintage_friction'] = portfolio['origination_q'].map(vintage_map).fillna(0)

# 2. Sovereign Feature Selection
# Corrected column names: 'dti' -> 'debt_to_equity', 'loan_amount' -> 'ead'
features = ['credit_score', 'debt_to_equity', 'ead', 'interest_coverage', 'vintage_friction']
X = portfolio[features]
y = portfolio['defaulted']

print("Balancing definitions via SMOTE...")
# 3. Neutralizing Imbalance (SMOTE)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Training Sovereign XGBoost Manifold...")
# 4. The High-Resonance Auditor: XGBoost
# scale_pos_weight already implicitly compensates for imbalance, but we also applied SME, 
# resulting in heavily weighted focus on detecting faults
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=6.1, 
    tree_method='hist', 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 5. Audit Validation
preds = model.predict(X_test)

rep = classification_report(y_test, preds)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

with open("report.txt", "w") as f:
    f.write(rep)
    f.write(f"\nROC-AUC: {auc:.4f}")

print("Saved to report.txt")
