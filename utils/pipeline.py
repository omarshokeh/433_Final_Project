import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import pulp

DROP_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Score', 'CLTV', 'Churn Reason', 'Churn Label',
]
NUMERIC_FEATURES = ['Tenure Months', 'Monthly Charges', 'Total Charges']
TARGET           = 'Churn Value'
HORIZON_MONTHS   = 12


# ── Loading ───────────────────────────────────────────────────────────────────

def load_and_clean(source):
    """Load from UploadedFile or file path, clean, and return df."""
    name = getattr(source, 'name', str(source))
    if name.endswith('.csv'):
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df.dropna(subset=['Total Charges'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


# ── Model training ────────────────────────────────────────────────────────────

@st.cache_data
def train_model(df):
    """
    Train logistic regression pipeline.
    Returns: (fitted_pipeline, auc, report_dict, fpr_list, tpr_list, feat_df)
    Pipeline is refit on full dataset after evaluation.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
             categorical_features),
        ],
        remainder='drop',
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier',   LogisticRegression(class_weight='balanced',
                                            max_iter=1000, random_state=42)),
    ])

    # Evaluate on hold-out
    pipeline.fit(X_train, y_train)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = pipeline.predict(X_test)
    auc     = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred,
                                    target_names=['No Churn', 'Churn'],
                                    output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Refit on full dataset for downstream scoring
    pipeline.fit(X, y)

    # Top-15 features by |coefficient|
    ohe_names = (
        pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    all_names = NUMERIC_FEATURES + ohe_names
    coefs     = pipeline.named_steps['classifier'].coef_[0]
    feat_df   = (
        pd.DataFrame({'feature': all_names, 'coef': coefs})
        .assign(abs_coef=lambda d: d['coef'].abs())
        .nlargest(15, 'abs_coef')
        .sort_values('abs_coef')
    )

    return pipeline, auc, report, fpr.tolist(), tpr.tolist(), feat_df


# ── Scoring ───────────────────────────────────────────────────────────────────

@st.cache_data
def score_customers(df, _pipeline):
    """
    Attach churn_prob, LTV, expected_loss to df.
    _pipeline is prefixed with _ so Streamlit does not hash it.
    """
    out = df.copy()
    X   = out.drop(columns=[TARGET])
    out['churn_prob']    = _pipeline.predict_proba(X)[:, 1]
    out['LTV']           = out['Monthly Charges'] * HORIZON_MONTHS
    out['expected_loss'] = out['churn_prob'] * out['LTV']
    return out


# ── Optimization helpers ──────────────────────────────────────────────────────

def _solve_knapsack(eligible, budget):
    n = len(eligible)
    if n == 0 or budget <= 0:
        return eligible.iloc[0:0].copy()
    c = eligible['c_i'].values
    d = eligible['delta_i'].values
    prob = pulp.LpProblem('knapsack', pulp.LpMaximize)
    x    = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(n)]
    prob += pulp.lpSum(d[i] * x[i] for i in range(n))
    prob += pulp.lpSum(c[i] * x[i] for i in range(n)) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    mask = [pulp.value(x[i]) == 1 for i in range(n)]
    return eligible[mask].copy()


def _greedy(eligible, budget, sort_col):
    ordered   = eligible.sort_values(sort_col, ascending=False)
    remaining = budget
    rows      = []
    for _, row in ordered.iterrows():
        if row['c_i'] <= remaining:
            rows.append(row)
            remaining -= row['c_i']
    if rows:
        return pd.DataFrame(rows).reset_index(drop=True)
    return eligible.iloc[0:0].copy()


# ── Optimization ──────────────────────────────────────────────────────────────

@st.cache_data
def run_optimization(scored_df, budget, alpha, delta):
    """
    Run binary knapsack (PuLP) and two greedy heuristics.
    Returns: (opt_targeted, heur_targeted, vd_targeted, eligible, summary_df)
    """
    df            = scored_df.copy()
    df['c_i']     = alpha * df['Monthly Charges']
    df['delta_i'] = delta * df['churn_prob'] * df['LTV'] - df['c_i']
    eligible      = df[df['delta_i'] > 0].copy().reset_index(drop=True)
    eligible['value_density'] = eligible['delta_i'] / eligible['c_i']

    opt_t  = _solve_knapsack(eligible, budget)
    heur_t = _greedy(eligible, budget, 'churn_prob')
    vd_t   = _greedy(eligible, budget, 'value_density')

    rows = []
    for name, t in [('Optimizer (PuLP)',         opt_t),
                    ('Heuristic (Greedy)',        heur_t),
                    ('Heuristic (Value Density)', vd_t)]:
        n    = len(t)
        cost = t['c_i'].sum()     if n > 0 else 0.0
        gain = t['delta_i'].sum() if n > 0 else 0.0
        roi  = (gain / cost * 100) if cost > 0 else 0.0
        rows.append({
            'Method'             : name,
            'Customers Targeted' : n,
            'Budget Used ($)'    : round(cost, 2),
            'Net Gain ($)'       : round(gain, 2),
            'ROI (%)'            : round(roi,  1),
        })
    return opt_t, heur_t, vd_t, eligible, pd.DataFrame(rows)


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

@st.cache_data
def run_sensitivity(scored_df, alpha, delta, max_budget=50_000, step=2_500):
    """
    Sweep budget from 0 to max_budget in `step` increments.
    Returns sensitivity_df with columns:
        budget, opt_gain, opt_n, heur_gain, heur_n, gain_gap
    """
    df            = scored_df.copy()
    df['c_i']     = alpha * df['Monthly Charges']
    df['delta_i'] = delta * df['churn_prob'] * df['LTV'] - df['c_i']
    eligible      = df[df['delta_i'] > 0].copy().reset_index(drop=True)

    records = []
    for b in range(0, max_budget + 1, step):
        opt  = _solve_knapsack(eligible, b)
        heur = _greedy(eligible, b, 'churn_prob')
        o_gain = opt['delta_i'].sum()  if len(opt)  > 0 else 0.0
        h_gain = heur['delta_i'].sum() if len(heur) > 0 else 0.0
        records.append({
            'budget'   : b,
            'opt_gain' : o_gain,
            'opt_n'    : len(opt),
            'heur_gain': h_gain,
            'heur_n'   : len(heur),
            'gain_gap' : o_gain - h_gain,
        })
    return pd.DataFrame(records)
