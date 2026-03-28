import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('loan_portfolio.csv')
X = df[['credit_score', 'debt_to_equity', 'ead']]
y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
prob = log_reg.predict_proba(X_test)[:, 1]

rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
auc = roc_auc_score(y_test, prob)

print("--- SUMMARY ---")
print(f"Accuracy: {rep['accuracy']:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Class 1 (Defaults) - Precision: {rep['1']['precision']:.4f}, Recall: {rep['1']['recall']:.4f}")
print("---------------")
