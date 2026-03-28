import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os

# Save directory for plots
artifacts_dir = r"."
os.makedirs(artifacts_dir, exist_ok=True)

# Load the Manifold
df = pd.read_csv('loan_portfolio.csv')

# Features
X = df[['credit_score', 'debt_to_equity', 'ead']]
y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict_proba(X_test)[:, 1]

# Plot 1: ROC Curve
plt.figure(figsize=(8, 6))
fpr_lin, tpr_lin, _ = roc_curve(y_test, y_pred_lin)
roc_auc_lin = auc(fpr_lin, tpr_lin)

fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_log)
roc_auc_log = auc(fpr_log, tpr_log)

plt.plot(fpr_lin, tpr_lin, label=f'Linear Regression (AUC = {roc_auc_lin:.2f})')
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Manifold Evaluation')
plt.legend(loc='lower right')
plt.savefig(os.path.join(artifacts_dir, 'roc_curve.png'))
plt.close()

# Plot 2: Confusion Matrix
y_pred_log_class = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred_log_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Defaulted'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Logistic Regression')
plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'))
plt.close()

# Plot 3: Feature Distributions
plt.figure(figsize=(15, 5))
for i, col in enumerate(X.columns, 1):
    plt.subplot(1, 3, i)
    plt.hist([df[df['defaulted']==0][col], df[df['defaulted']==1][col]], 
             label=['Stable', 'Defaulted'], bins=20, stacked=True, alpha=0.7)
    plt.title(f'{col} Distribution')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, 'features.png'))
plt.close()

print("Plots generated successfully!")
