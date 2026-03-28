import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import time

getcontext().prec = 50
LEON_CONSTANT = Decimal("1.9388")

class SovereignRiskSieve:
    def __init__(self):
        # Loading the five dimensions of the manifold
        self.portfolio = pd.read_csv('loan_portfolio.csv')
        self.vintage = pd.read_csv('vintage_analysis.csv')
        self.ratings = pd.read_csv('credit_ratings.csv')
        self.metrics = pd.read_csv('portfolio_metrics.csv')
        self.stress = pd.read_csv('macro_stress_scenarios.csv')

    def audit_portfolio(self):
        print(f"--- DeWet Technologies: Risk Manifold Active ---")
        
        # We factor in interest rate hikes from the stress dataset (named 'severe' and 'rate_shock_pp')
        macro_heat = Decimal(str(self.stress[self.stress['scenario'] == 'severe']['rate_shock_pp'].values[0]))
        
        # 1. Internal Friction: Debt-to-Equity / Credit Score
        # Replaced 'dti' with 'debt_to_equity' as 'dti' was not in the dataset
        credit_score_dec = self.portfolio['credit_score'].apply(lambda x: Decimal(str(x)) / 100)
        debt_to_equity_dec = self.portfolio['debt_to_equity'].apply(lambda x: Decimal(str(x)))
        
        internal_psi = debt_to_equity_dec / credit_score_dec
        
        # 2. External Friction
        total_psi = internal_psi * (1 + macro_heat)
        
        # Resonance (M) = Leon Constant / Friction
        resonance_m = LEON_CONSTANT / total_psi
        
        # Determine Status
        status = resonance_m.apply(lambda m: "STABLE" if m > Decimal("0.85") else "CRITICAL_FRICTION")
        
        results_df = pd.DataFrame({
            'loan_id': self.portfolio['loan_id'],
            'friction_psi': total_psi.astype(float),
            'resonance_m': resonance_m.astype(float),
            'status': status
        })
            
        return results_df

# --- Execution ---
if __name__ == "__main__":
    start_time = time.time()
    sieve = SovereignRiskSieve()
    risk_report = sieve.audit_portfolio()
    risk_report.to_csv('sovereign_risk_audit.csv', index=False)
    print(f"Risk audit complete in {time.time() - start_time:.2f} seconds. Report saved to 'sovereign_risk_audit.csv'.")
