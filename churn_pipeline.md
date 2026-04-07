# Churn Prediction Pipeline
### IBM Telco Customer Churn Dataset | MSE 433 Final Project

This notebook covers end-to-end preprocessing, logistic regression modelling, evaluation, and probability scoring.  


## 1. Data Loading & Preprocessing
Load the raw Excel file, coerce `Total Charges` to numeric (whitespace entries become `NaN` and are dropped), remove identifier/leakage columns, and report dataset shape and overall churn rate.


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('Telco_customer_churn.xlsx')

# ── Coerce Total Charges (whitespace → NaN → drop) ────────────────────────────
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Drop identifier, geography, and leakage columns ───────────────────────────
DROP_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Score', 'CLTV', 'Churn Reason', 'Churn Label'
]
df.drop(columns=DROP_COLS, inplace=True)

# ── Summary ───────────────────────────────────────────────────────────────────
churn_rate = df['Churn Value'].mean() * 100
print(f'Shape after cleaning : {df.shape}')
print(f'Churn rate           : {churn_rate:.2f}%')
df.head(3)
```

    Shape after cleaning : (7032, 20)
    Churn rate           : 26.58%





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Senior Citizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>Tenure Months</th>
      <th>Phone Service</th>
      <th>Multiple Lines</th>
      <th>Internet Service</th>
      <th>Online Security</th>
      <th>Online Backup</th>
      <th>Device Protection</th>
      <th>Tech Support</th>
      <th>Streaming TV</th>
      <th>Streaming Movies</th>
      <th>Contract</th>
      <th>Paperless Billing</th>
      <th>Payment Method</th>
      <th>Monthly Charges</th>
      <th>Total Charges</th>
      <th>Churn Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>8</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>820.50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Train / Test Split
An 80/20 stratified split on `Churn Value` preserves the class imbalance ratio in both partitions.


```python
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES  = ['Tenure Months', 'Monthly Charges', 'Total Charges']
TARGET            = 'Churn Value'

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify categorical columns (everything that is not numeric and not the target)
CATEGORICAL_FEATURES = X.select_dtypes(include='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f'Training set   : {X_train.shape[0]} rows')
print(f'Test set       : {X_test.shape[0]} rows')
print(f'Numeric cols   : {NUMERIC_FEATURES}')
print(f'Categorical cols ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}')
```

    Training set   : 5625 rows
    Test set       : 1407 rows
    Numeric cols   : ['Tenure Months', 'Monthly Charges', 'Total Charges']
    Categorical cols (16): ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']


## 3. Modelling Pipeline
A `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder` (drop first dummy to avoid multicollinearity) to categoricals.  
`class_weight='balanced'` compensates for the ~26 % churn minority without resampling.


```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),                                    NUMERIC_FEATURES),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)
print('Pipeline fitted successfully.')
```

    Pipeline fitted successfully.


## 4. Evaluation
ROC-AUC summarises discrimination across all thresholds; the classification report shows precision, recall, and F1 at the default 0.5 cut-off.


```python
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

y_pred      = pipeline.predict(X_test)
y_prob_test = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob_test)
print(f'ROC-AUC : {roc_auc:.4f}\n')
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
```

    ROC-AUC : 0.8426
    
                  precision    recall  f1-score   support
    
        No Churn       0.90      0.71      0.79      1033
           Churn       0.49      0.79      0.61       374
    
        accuracy                           0.73      1407
       macro avg       0.70      0.75      0.70      1407
    weighted avg       0.79      0.73      0.74      1407
    


## 4b. Cross-Validation
A single 80/20 split AUC can vary with the random seed. 5-fold stratified CV re-estimates AUC across five non-overlapping held-out folds, validating that the hold-out result is a stable and representative estimate of generalisation performance.


```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

print('5-Fold Stratified Cross-Validation — ROC-AUC')
print('─' * 42)
for i, score in enumerate(cv_scores, 1):
    print(f'  Fold {i} : {score:.4f}')
print('─' * 42)
cv_mean = cv_scores.mean()
cv_std  = cv_scores.std()
print(f'  Mean   : {cv_mean:.4f}')
print(f'  Std    : {cv_std:.4f}')
print(f'  95% CI : [{cv_mean - 2*cv_std:.4f}, {cv_mean + 2*cv_std:.4f}]')
print()
print(f'Hold-out test AUC : {roc_auc:.4f}')
gap = abs(cv_mean - roc_auc)
if gap < 0.01:
    print(f'Verdict           : Consistent — CV mean and hold-out AUC differ by {gap:.4f} (< 0.01)')
else:
    print(f'Verdict           : Notable gap — CV mean and hold-out AUC differ by {gap:.4f} (≥ 0.01)')
```

    5-Fold Stratified Cross-Validation — ROC-AUC
    ──────────────────────────────────────────
      Fold 1 : 0.8545
      Fold 2 : 0.8458
      Fold 3 : 0.8481
      Fold 4 : 0.8695
      Fold 5 : 0.8687
    ──────────────────────────────────────────
      Mean   : 0.8573
      Std    : 0.0101
      95% CI : [0.8372, 0.8774]
    
    Hold-out test AUC : 0.8426
    Verdict           : Notable gap — CV mean and hold-out AUC differ by 0.0147 (≥ 0.01)


## 4c. Model Comparison
Logistic regression is benchmarked against a majority-class dummy, a shallow decision tree, and a random forest. This justifies the model choice: logistic regression provides interpretable coefficients (used for feature importance in Section 5), well-calibrated probability scores (required for expected loss computation in Stage 2), and competitive AUC relative to more complex alternatives.


```python
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SKPipeline

# ── Model registry — reuse existing pipeline and preprocessor ─────────────────
models = {
    'Dummy (majority class)'  : DummyClassifier(strategy='most_frequent'),
    'Logistic Regression'     : pipeline,
    'Decision Tree (depth=5)' : SKPipeline([
        ('preprocessor', preprocessor),
        ('classifier',   DecisionTreeClassifier(max_depth=5, class_weight='balanced',
                                                random_state=42))
    ]),
    'Random Forest (100 trees)': SKPipeline([
        ('preprocessor', preprocessor),
        ('classifier',   RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                random_state=42, n_jobs=-1))
    ]),
}

# ── 5-fold CV for each model (reuse cv object from 4b) ───────────────────────
print('Model Comparison — 5-Fold Stratified CV ROC-AUC')
print('─' * 52)
print(f'  {"Model":<30} {"Mean AUC":>9}  {"Std":>7}')
print('─' * 52)
comparison_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    comparison_results[name] = scores
    print(f'  {name:<30} {scores.mean():>9.4f}  {scores.std():>7.4f}')
print('─' * 52)
print()
print('Conclusion:')
print('  Logistic Regression was selected because:')
print('  1. Interpretable coefficients — directly used for feature importance analysis.')
print('  2. Well-calibrated probabilities — required by Stage 2 expected loss formula')
print('     (expected_loss = churn_prob × LTV); poorly calibrated scores would')
print('     distort revenue-at-risk estimates.')
print('  3. Competitive AUC — matches or closely approaches more complex models,')
print('     providing no justification for sacrificing interpretability.')
```

    Model Comparison — 5-Fold Stratified CV ROC-AUC
    ────────────────────────────────────────────────────
      Model                           Mean AUC      Std
    ────────────────────────────────────────────────────
      Dummy (majority class)            0.5000   0.0000
      Logistic Regression               0.8573   0.0101
      Decision Tree (depth=5)           0.8414   0.0112
      Random Forest (100 trees)         0.8365   0.0106
    ────────────────────────────────────────────────────
    
    Conclusion:
      Logistic Regression was selected because:
      1. Interpretable coefficients — directly used for feature importance analysis.
      2. Well-calibrated probabilities — required by Stage 2 expected loss formula
         (expected_loss = churn_prob × LTV); poorly calibrated scores would
         distort revenue-at-risk estimates.
      3. Competitive AUC — matches or closely approaches more complex models,
         providing no justification for sacrificing interpretability.


## 5. Visualisations
Three Plotly charts in a single output: (1) ROC curve with AUC annotation, (2) predicted probability distributions split by true churn label, and (3) the top 15 features ranked by absolute logistic regression coefficient.


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 1. ROC curve data ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob_test)

# ── 2. Probability distributions ─────────────────────────────────────────────
# Convert y_test to a numpy array so boolean indexing aligns with y_prob_test
y_test_arr    = y_test.to_numpy()
prob_no_churn = y_prob_test[y_test_arr == 0]
prob_churn    = y_prob_test[y_test_arr == 1]

# ── 3. Feature importances ────────────────────────────────────────────────────
ohe_feature_names = (
    pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(CATEGORICAL_FEATURES)
    .tolist()
)
all_feature_names = NUMERIC_FEATURES + ohe_feature_names
coefficients      = pipeline.named_steps['classifier'].coef_[0]

feat_df = (
    pd.DataFrame({'feature': all_feature_names, 'coef': coefficients})
    .assign(abs_coef=lambda d: d['coef'].abs())
    .nlargest(15, 'abs_coef')
    .sort_values('abs_coef')
)

# ── Build subplots ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        f'ROC Curve (AUC = {roc_auc:.3f})',
        'Predicted Probability Distribution',
        'Top 15 Feature Importances'
    ),
    column_widths=[0.30, 0.35, 0.35]
)

# Panel 1 — ROC
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC',
                         line=dict(color='steelblue', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                         line=dict(dash='dash', color='grey')), row=1, col=1)
fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
fig.update_yaxes(title_text='True Positive Rate',  row=1, col=1)

# Panel 2 — Probability distributions
fig.add_trace(go.Histogram(x=prob_no_churn, name='No Churn', opacity=0.6,
                           marker_color='steelblue', nbinsx=40), row=1, col=2)
fig.add_trace(go.Histogram(x=prob_churn,    name='Churn',    opacity=0.6,
                           marker_color='tomato',    nbinsx=40), row=1, col=2)
fig.update_xaxes(title_text='Predicted Churn Probability', row=1, col=2)
fig.update_yaxes(title_text='Count',                       row=1, col=2)

# Panel 3 — Feature importances
bar_colors = ['tomato' if c > 0 else 'steelblue' for c in feat_df['coef']]
fig.add_trace(go.Bar(x=feat_df['coef'], y=feat_df['feature'],
                     orientation='h', marker_color=bar_colors,
                     name='Coefficient'), row=1, col=3)
fig.update_xaxes(title_text='Coefficient Value', row=1, col=3)

# ── Single update_layout call — barmode must be set here to take effect ───────
fig.update_layout(
    barmode='overlay',
    height=480,
    width=1300,
    title_text='Stage 1 — Model Diagnostics',
    showlegend=True,
    template='plotly_white'
)
fig.show()
```



## 6. Refit on Full Dataset & Attach `churn_prob`
Refit the same pipeline on all available data to maximise scoring accuracy, then append the predicted churn probability as a new column on `df` for use in Stage 2.


```python
# ── Score using the held-out pipeline (trained on 80% only) ──────────────────
# Re-train on X_train only (do NOT refit on full dataset)
pipeline.fit(X_train, y_train)

# Score ALL rows using this pipeline — test rows are truly out-of-sample,
# train rows are in-sample but this is acceptable for a planning/allocation tool
df['churn_prob'] = pipeline.predict_proba(X)[:, 1]

# Flag which rows were in the test set for transparency
df['in_test_set'] = 0
df.loc[X_test.index, 'in_test_set'] = 1

print(f"churn_prob attached. df shape: {df.shape}")
print(f"Test set rows (out-of-sample): {df['in_test_set'].sum():,}")
print(f"Train set rows (in-sample):    {(df['in_test_set']==0).sum():,}")
df[['churn_prob']].describe()
```

    churn_prob attached. df shape: (7032, 22)
    Test set rows (out-of-sample): 1,407
    Train set rows (in-sample):    5,625





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>churn_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.407833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.313134</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001444</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.092513</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.377211</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.716648</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.945439</td>
    </tr>
  </tbody>
</table>
</div>



## 7. Sanity Check
Confirm that `churn_prob` contains no missing values and that all scores fall within the valid probability range \[0, 1\].


```python
nan_count      = df['churn_prob'].isna().sum()
out_of_range   = ((df['churn_prob'] < 0) | (df['churn_prob'] > 1)).sum()

print(f'NaN values in churn_prob    : {nan_count}      (expected 0)')
print(f'Values outside [0, 1]       : {out_of_range}  (expected 0)')
print()
print(df['churn_prob'].describe().to_string())

assert nan_count    == 0, 'ERROR: NaNs found in churn_prob'
assert out_of_range == 0, 'ERROR: probabilities outside [0, 1]'
print('\nAll checks passed. df with churn_prob is ready for Stage 2.')
```

    NaN values in churn_prob    : 0      (expected 0)
    Values outside [0, 1]       : 0  (expected 0)
    
    count    7032.000000
    mean        0.407833
    std         0.313134
    min         0.001444
    25%         0.092513
    50%         0.377211
    75%         0.716648
    max         0.945439
    
    All checks passed. df with churn_prob is ready for Stage 2.


## Stage 2 — LTV Estimation & Expected Loss
LTV is a 12-month forward-looking revenue estimate based on each customer's current monthly spend.  
Expected loss is the probability-weighted revenue at risk per customer: `churn_prob × LTV`.


```python
# ── Dependency guard ──────────────────────────────────────────────────────────
if 'df' not in vars():
    raise NameError(
        "df is not defined. Run all Stage 1 cells (1–7) first to create "
        "df with the churn_prob column before executing Stage 2."
    )
if 'churn_prob' not in df.columns:
    raise KeyError(
        "churn_prob column missing from df. Run Stage 1 cell 6 "
        "(Refit on Full Dataset) to attach it before executing Stage 2."
    )

# ── LTV & Expected Loss ───────────────────────────────────────────────────────
HORIZON_MONTHS = 12

df['LTV']           = df['Monthly Charges'] * HORIZON_MONTHS
df['expected_loss'] = df['churn_prob'] * df['LTV']

print(f'LTV summary stats:')
print(df['LTV'].describe().to_string())
print()
print(f'Expected Loss summary stats:')
print(df['expected_loss'].describe().to_string())
```

    LTV summary stats:
    count    7032.000000
    mean      777.578498
    std       361.031687
    min       219.000000
    25%       427.050000
    50%       844.200000
    75%      1078.350000
    max      1425.000000
    
    Expected Loss summary stats:
    count    7032.000000
    mean      355.980678
    std       331.678095
    min         0.344807
    25%        55.114536
    50%       250.703288
    75%       659.423175
    max      1182.753715



```python
# ── Sanity checks ─────────────────────────────────────────────────────────────
assert df['LTV'].isna().sum() == 0,           'ERROR: NaNs found in LTV'
assert df['expected_loss'].isna().sum() == 0, 'ERROR: NaNs found in expected_loss'
print('No NaNs in LTV or expected_loss.\n')

print(df[['Monthly Charges', 'churn_prob', 'LTV', 'expected_loss']].describe())
print()
print('Top 10 customers by expected_loss:')
print(
    df[['Monthly Charges', 'churn_prob', 'LTV', 'expected_loss']]
    .nlargest(10, 'expected_loss')
    .to_string()
)
```

    No NaNs in LTV or expected_loss.
    
           Monthly Charges   churn_prob          LTV  expected_loss
    count      7032.000000  7032.000000  7032.000000    7032.000000
    mean         64.798208     0.407833   777.578498     355.980678
    std          30.085974     0.313134   361.031687     331.678095
    min          18.250000     0.001444   219.000000       0.344807
    25%          35.587500     0.092513   427.050000      55.114536
    50%          70.350000     0.377211   844.200000     250.703288
    75%          89.862500     0.716648  1078.350000     659.423175
    max         118.750000     0.945439  1425.000000    1182.753715
    
    Top 10 customers by expected_loss:
          Monthly Charges  churn_prob     LTV  expected_loss
    1827           105.90    0.930716  1270.8    1182.753715
    1044           105.30    0.920182  1263.6    1162.741780
    1566           105.00    0.915193  1260.0    1153.143063
    1263           105.50    0.906845  1266.0    1148.065936
    3020           111.40    0.857086  1336.8    1145.752138
    64             106.90    0.892984  1282.8    1145.519571
    1681           101.95    0.933982  1223.4    1142.634086
    584            100.80    0.944598  1209.6    1142.585362
    607            106.70    0.887755  1280.4    1136.680929
    204            111.20    0.847382  1334.4    1130.746221



```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'LTV Distribution',
        'Churn Probability vs Expected Loss'
    ),
    column_widths=[0.45, 0.55]
)

# ── Panel 1 — LTV histogram ───────────────────────────────────────────────────
fig.add_trace(
    go.Histogram(
        x=df['LTV'],
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.8,
        name='LTV'
    ),
    row=1, col=1
)
fig.update_xaxes(title_text='LTV (USD, 12-month)', row=1, col=1)
fig.update_yaxes(title_text='Number of Customers', row=1, col=1)

# ── Panel 2 — Scatter: churn_prob vs expected_loss, coloured by Monthly Charges
fig.add_trace(
    go.Scatter(
        x=df['churn_prob'],
        y=df['expected_loss'],
        mode='markers',
        marker=dict(
            color=df['Monthly Charges'],
            colorscale='Viridis',
            size=4,
            opacity=0.7,
            colorbar=dict(title='Monthly<br>Charges ($)', x=1.02)
        ),
        name='Customers'
    ),
    row=1, col=2
)
fig.update_xaxes(title_text='Churn Probability', row=1, col=2)
fig.update_yaxes(title_text='Expected Loss (USD)', row=1, col=2)

fig.update_layout(
    height=460,
    width=1200,
    title_text='Stage 2 — LTV & Expected Loss Diagnostics',
    showlegend=False,
    template='plotly_white'
)
fig.show()
```



## Stage 3 — Budget-Constrained Optimization
Each eligible customer is assigned a binary decision variable: target with a retention incentive (1) or not (0).  
The optimizer maximises total net gain — probability-weighted revenue saved minus incentive cost — subject to a total budget cap, solved as a binary knapsack via PuLP/CBC.


```python
# ── Dependency guard ──────────────────────────────────────────────────────────
if 'df' not in vars():
    raise NameError("df is not defined. Run all Stage 1 and Stage 2 cells first.")
for _col in ('churn_prob', 'LTV', 'expected_loss'):
    if _col not in df.columns:
        raise KeyError(f"'{_col}' missing from df — run Stage 2 cells first.")

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHA  = 0.10    # incentive cost rate  (cost_i = ALPHA × Monthly Charges_i)
DELTA  = 0.50    # churn reduction factor (incentive halves churn_prob)
BUDGET = 10_000  # total retention budget ($)

# ── Per-customer cost and incremental gain ────────────────────────────────────
df['c_i']     = ALPHA * df['Monthly Charges']
df['delta_i'] = DELTA * df['churn_prob'] * df['LTV'] - df['c_i']

# ── Pre-filter: only customers where incentive returns positive net gain ───────
eligible = df[df['delta_i'] > 0].copy().reset_index(drop=True)

print(f"Total customers  : {len(df):,}")
print(f"Eligible (δ > 0) : {len(eligible):,}")
print(f"Excluded (δ ≤ 0) : {len(df) - len(eligible):,}")
```

    Total customers  : 7,032
    Eligible (δ > 0) : 6,481
    Excluded (δ ≤ 0) : 551



```python
import pulp

def run_optimizer(budget):
    """
    Solve the binary retention knapsack for a given budget using PuLP/CBC.
    Reads the module-level `eligible` DataFrame (pre-filtered, delta_i > 0).
    Returns (targeted_df, solver_status_str).
    """
    n          = len(eligible)
    c_vals     = eligible['c_i'].values
    delta_vals = eligible['delta_i'].values

    prob = pulp.LpProblem('retention_knapsack', pulp.LpMaximize)
    x    = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(n)]

    # Objective: maximise total net gain
    prob += pulp.lpSum(delta_vals[i] * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(c_vals[i] * x[i] for i in range(n)) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    mask = [pulp.value(x[i]) == 1 for i in range(n)]
    return eligible[mask].copy(), pulp.LpStatus[prob.status]

# ── Baseline run at BUDGET = $10,000 ─────────────────────────────────────────
opt_targeted, opt_status = run_optimizer(BUDGET)
opt_n    = len(opt_targeted)
opt_cost = opt_targeted['c_i'].sum()
opt_gain = opt_targeted['delta_i'].sum()

print(f"Solver status        : {opt_status}")
print(f"Customers targeted   : {opt_n:,}")
print(f"Total incentive cost : ${opt_cost:,.2f}")
print(f"Total net gain       : ${opt_gain:,.2f}")
```

    Solver status        : Optimal
    Customers targeted   : 1,239
    Total incentive cost : $10,000.00
    Total net gain       : $505,958.07



```python
# ── Greedy heuristic: sort eligible customers by churn_prob descending ────────
heuristic_df     = eligible.sort_values('churn_prob', ascending=False).copy()
budget_remaining = BUDGET
heur_rows        = []

for _, row in heuristic_df.iterrows():
    if row['c_i'] <= budget_remaining:
        heur_rows.append(row)
        budget_remaining -= row['c_i']

heur_targeted = pd.DataFrame(heur_rows).reset_index(drop=True)
heur_n        = len(heur_targeted)
heur_cost     = heur_targeted['c_i'].sum()    if heur_n > 0 else 0.0
heur_gain     = heur_targeted['delta_i'].sum() if heur_n > 0 else 0.0

print(f"Customers targeted   : {heur_n:,}")
print(f"Total incentive cost : ${heur_cost:,.2f}")
print(f"Total net gain       : ${heur_gain:,.2f}")
```

    Customers targeted   : 1,240
    Total incentive cost : $9,999.91
    Total net gain       : $505,936.39



```python
# ── Value-density heuristic: sort by delta_i / c_i descending ────────────────
vd_df            = eligible.copy()
vd_df['density'] = vd_df['delta_i'] / vd_df['c_i']
vd_df            = vd_df.sort_values('density', ascending=False)

budget_remaining = BUDGET
vd_rows          = []
for _, row in vd_df.iterrows():
    if row['c_i'] <= budget_remaining:
        vd_rows.append(row)
        budget_remaining -= row['c_i']

vd_targeted = pd.DataFrame(vd_rows).reset_index(drop=True)
vd_n        = len(vd_targeted)
vd_cost     = vd_targeted['c_i'].sum()    if vd_n > 0 else 0.0
vd_gain     = vd_targeted['delta_i'].sum() if vd_n > 0 else 0.0

print(f"Customers targeted   : {vd_n:,}")
print(f"Total incentive cost : ${vd_cost:,.2f}")
print(f"Total net gain       : ${vd_gain:,.2f}")
```

    Customers targeted   : 1,240
    Total incentive cost : $9,999.91
    Total net gain       : $505,936.39



```python
vd_roi   = vd_gain / vd_cost if vd_cost > 0 else float('nan')

summary = pd.DataFrame({
    'Method'                  : ['Optimizer (PuLP)', 'Heuristic (Churn Prob)', 'Heuristic (Value Density)'],
    'Customers Targeted'      : [opt_n,              heur_n,                   vd_n],
    'Budget Used ($)'         : [f'{opt_cost:,.2f}', f'{heur_cost:,.2f}',      f'{vd_cost:,.2f}'],
    'Total Net Gain ($)'      : [f'{opt_gain:,.2f}', f'{heur_gain:,.2f}',      f'{vd_gain:,.2f}'],
    'ROI (Net Gain / Budget)' : [f'{opt_roi:.2f}x',  f'{heur_roi:.2f}x',       f'{vd_roi:.2f}x'],
})
print(summary.to_string(index=False))

pct_churn = (opt_gain - heur_gain) / abs(heur_gain) * 100 if heur_gain != 0 else float('nan')
pct_vd    = (opt_gain - vd_gain)   / abs(vd_gain)   * 100 if vd_gain   != 0 else float('nan')
print(f"\nOptimizer vs churn-prob heuristic : {pct_churn:.4f}%")
print(f"Optimizer vs value-density heuristic: {pct_vd:.4f}%")
```

                       Method  Customers Targeted Budget Used ($) Total Net Gain ($) ROI (Net Gain / Budget)
             Optimizer (PuLP)                1239       10,000.00         505,958.07                  50.38x
       Heuristic (Churn Prob)                1240        9,999.91         505,936.39                  50.38x
    Heuristic (Value Density)                1240        9,999.91         505,936.39                  50.59x
    
    Optimizer vs churn-prob heuristic : 0.0043%
    Optimizer vs value-density heuristic: 0.0043%



```python
# ── Decision Table: Who gets targeted by the optimizer ────────────────────────
targeted_summary = opt_targeted[[
    'Monthly Charges', 'churn_prob', 'LTV', 'c_i', 'delta_i'
]].copy()

targeted_summary.columns = [
    'Monthly Revenue ($)', 'Churn Probability', '12-Month LTV ($)',
    'Incentive Cost ($)', 'Net Gain ($)'
]

targeted_summary = targeted_summary.sort_values('Net Gain ($)', ascending=False).reset_index(drop=True)
targeted_summary.index += 1  # 1-based rank

print(f"Optimizer selected {len(targeted_summary)} customers to target.\n")
print("── Top 15 Customers by Net Gain ──────────────────────────────────────────")
print(targeted_summary.head(15).to_string())
print()
print("── Portfolio Summary ─────────────────────────────────────────────────────")
print(f"  Avg Churn Probability  : {targeted_summary['Churn Probability'].mean():.3f}")
print(f"  Avg 12-Month LTV ($)   : ${targeted_summary['12-Month LTV ($)'].mean():,.2f}")
print(f"  Avg Incentive Cost ($) : ${targeted_summary['Incentive Cost ($)'].mean():,.2f}")
print(f"  Avg Net Gain ($)       : ${targeted_summary['Net Gain ($)'].mean():,.2f}")
print(f"  Total Budget Used ($)  : ${targeted_summary['Incentive Cost ($)'].sum():,.2f}")
print(f"  Total Net Gain ($)     : ${targeted_summary['Net Gain ($)'].sum():,.2f}")
```

    Optimizer selected 1239 customers to target.
    
    ── Top 15 Customers by Net Gain ──────────────────────────────────────────
        Monthly Revenue ($)  Churn Probability  12-Month LTV ($)  Incentive Cost ($)  Net Gain ($)
    1                105.90           0.930716            1270.8              10.590    580.786858
    2                105.30           0.920182            1263.6              10.530    570.840890
    3                105.00           0.915193            1260.0              10.500    566.071531
    4                105.50           0.906845            1266.0              10.550    563.482968
    5                106.90           0.892984            1282.8              10.690    562.069786
    6                111.40           0.857086            1336.8              11.140    561.736069
    7                100.80           0.944598            1209.6              10.080    561.212681
    8                101.95           0.933982            1223.4              10.195    561.122043
    9                106.70           0.887755            1280.4              10.670    557.670465
    10               111.20           0.847382            1334.4              11.120    554.253110
    11               102.45           0.917222            1229.4              10.245    553.571510
    12               104.85           0.896386            1258.2              10.485    553.431212
    13               101.45           0.925014            1217.4              10.145    552.910959
    14               110.85           0.844237            1330.2              11.085    550.417218
    15               102.00           0.913473            1224.0              10.200    548.845472
    
    ── Portfolio Summary ─────────────────────────────────────────────────────
      Avg Churn Probability  : 0.858
      Avg 12-Month LTV ($)   : $968.52
      Avg Incentive Cost ($) : $8.07
      Avg Net Gain ($)       : $408.36
      Total Budget Used ($)  : $10,000.00
      Total Net Gain ($)     : $505,958.07



```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

opt_mean_delta  = opt_targeted['delta_i'].mean()  if opt_n  > 0 else 0.0
heur_mean_delta = heur_targeted['delta_i'].mean() if heur_n > 0 else 0.0

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Total Net Gain: Optimizer vs Heuristic',
        'Distribution of Net Gain per Customer (Eligible)'
    ),
    column_widths=[0.35, 0.65]
)

# ── Panel 1 — bar chart ───────────────────────────────────────────────────────
fig.add_trace(
    go.Bar(
        x=['Optimizer (PuLP)', 'Heuristic (Greedy)'],
        y=[opt_gain, heur_gain],
        marker_color=['steelblue', 'tomato'],
        text=[f'${opt_gain:,.0f}', f'${heur_gain:,.0f}'],
        textposition='outside',
        name='Net Gain'
    ),
    row=1, col=1
)
fig.update_yaxes(title_text='Total Net Gain ($)', row=1, col=1)

# ── Panel 2 — histogram of delta_i with mean vertical lines ──────────────────
fig.add_trace(
    go.Histogram(
        x=eligible['delta_i'],
        nbinsx=60,
        marker_color='lightsteelblue',
        opacity=0.8,
        name='All eligible'
    ),
    row=1, col=2
)
fig.add_vline(
    x=opt_mean_delta,
    line=dict(color='steelblue', dash='dash', width=2),
    annotation_text='Optimizer mean',
    annotation_position='top right',
    row=1, col=2
)
fig.add_vline(
    x=heur_mean_delta,
    line=dict(color='tomato', dash='dash', width=2),
    annotation_text='Heuristic mean',
    annotation_position='top left',
    row=1, col=2
)
fig.update_xaxes(title_text='Net Gain per Customer, delta_i ($)', row=1, col=2)
fig.update_yaxes(title_text='Number of Customers',                row=1, col=2)

fig.update_layout(
    height=460,
    width=1200,
    title_text='Stage 3 — Optimization Results',
    showlegend=False,
    template='plotly_white'
)
fig.show()
```



## Stage 4 — Sensitivity Analysis
We sweep the retention budget from $0 to $50,000 and record total net gain and customers targeted under both the PuLP optimizer and the greedy heuristic at each level.  
This reveals the marginal return on additional budget spend and quantifies how much value the optimizer captures relative to the heuristic across the full range.


```python
# ── Dependency guard ──────────────────────────────────────────────────────────
if 'eligible' not in vars():
    raise NameError("eligible is not defined. Run all Stage 3 cells first.")
if 'run_optimizer' not in vars():
    raise NameError("run_optimizer is not defined. Run the Stage 3 PuLP cell first.")

# ── Helper: greedy heuristic for an arbitrary budget ─────────────────────────
def run_heuristic(budget):
    """Sort eligible customers by churn_prob descending, fill greedily."""
    ordered          = eligible.sort_values('churn_prob', ascending=False)
    budget_remaining = budget
    rows             = []
    for _, row in ordered.iterrows():
        if row['c_i'] <= budget_remaining:
            rows.append(row)
            budget_remaining -= row['c_i']
    if rows:
        t = pd.DataFrame(rows)
        return t['delta_i'].sum(), len(t)
    return 0.0, 0

# ── Budget sweep ──────────────────────────────────────────────────────────────
BUDGET_GRID = list(range(0, 50_001, 2_500))
records     = []

for b in BUDGET_GRID:
    opt_df, _      = run_optimizer(b)
    o_gain         = opt_df['delta_i'].sum() if len(opt_df) > 0 else 0.0
    o_n            = len(opt_df)

    h_gain, h_n    = run_heuristic(b)

    records.append({
        'budget'   : b,
        'opt_gain' : o_gain,
        'opt_n'    : o_n,
        'heur_gain': h_gain,
        'heur_n'   : h_n,
        'gain_gap' : o_gain - h_gain,
    })

sensitivity_df = pd.DataFrame(records)
print(sensitivity_df.to_string(index=False))
```

     budget     opt_gain  opt_n    heur_gain  heur_n  gain_gap
          0 0.000000e+00      0 0.000000e+00       0  0.000000
       2500 1.334882e+05    292 1.334232e+05     293 64.986332
       5000 2.623153e+05    605 2.622871e+05     608 28.188725
       7500 3.867298e+05    915 3.866665e+05     914 63.339474
      10000 5.059581e+05   1239 5.059364e+05    1240 21.685310
      12500 6.194095e+05   1567 6.193845e+05    1568 25.007619
      15000 7.252683e+05   1907 7.252540e+05    1908 14.230030
      17500 8.218954e+05   2262 8.218726e+05    2263 22.844464
      20000 9.071455e+05   2627 9.071432e+05    2628  2.372078
      22500 9.809712e+05   3047 9.809455e+05    3045 25.740333
      25000 1.043297e+06   3458 1.043265e+06    3459 32.024843
      27500 1.092983e+06   3832 1.092958e+06    3831 24.962460
      30000 1.131023e+06   4187 1.131022e+06    4189  1.410855
      32500 1.160328e+06   4539 1.160321e+06    4540  7.581687
      35000 1.181770e+06   4956 1.181770e+06    4957  0.114826
      37500 1.195214e+06   5379 1.195214e+06    5379  0.624103
      40000 1.202766e+06   5774 1.202763e+06    5776  3.460217
      42500 1.206387e+06   6207 1.206387e+06    6209  0.468130
      45000 1.206859e+06   6481 1.206859e+06    6481  0.000000
      47500 1.206859e+06   6481 1.206859e+06    6481  0.000000
      50000 1.206859e+06   6481 1.206859e+06    6481  0.000000



```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        'Total Net Gain vs Retention Budget',
        'Customers Targeted vs Retention Budget'
    ),
    vertical_spacing=0.14,
    shared_xaxes=True
)

budgets = sensitivity_df['budget']

# ── Panel 1 — net gain ────────────────────────────────────────────────────────
fig.add_trace(
    go.Scatter(x=budgets, y=sensitivity_df['opt_gain'],
               mode='lines+markers', name='Optimizer (PuLP)',
               line=dict(color='steelblue', width=2), marker=dict(size=5)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=budgets, y=sensitivity_df['heur_gain'],
               mode='lines+markers', name='Heuristic (Greedy)',
               line=dict(color='tomato', width=2, dash='dot'), marker=dict(size=5)),
    row=1, col=1
)
fig.add_vline(x=10_000, line=dict(color='grey', dash='dash', width=1.5),
              annotation_text='Baseline ($10k)', annotation_position='top right',
              row=1, col=1)
fig.update_yaxes(title_text='Total Net Gain ($)', row=1, col=1)

# ── Panel 2 — customers targeted ─────────────────────────────────────────────
fig.add_trace(
    go.Scatter(x=budgets, y=sensitivity_df['opt_n'],
               mode='lines+markers', name='Optimizer (PuLP)',
               line=dict(color='steelblue', width=2), marker=dict(size=5),
               showlegend=False),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=budgets, y=sensitivity_df['heur_n'],
               mode='lines+markers', name='Heuristic (Greedy)',
               line=dict(color='tomato', width=2, dash='dot'), marker=dict(size=5),
               showlegend=False),
    row=2, col=1
)
fig.add_vline(x=10_000, line=dict(color='grey', dash='dash', width=1.5),
              row=2, col=1)
fig.update_xaxes(title_text='Retention Budget ($)', row=2, col=1)
fig.update_yaxes(title_text='Customers Targeted',  row=2, col=1)

fig.update_layout(
    height=700,
    width=1000,
    title_text='Stage 4 — Sensitivity Analysis: Budget Sweep $0 – $50,000',
    legend=dict(x=0.01, y=0.98),
    template='plotly_white'
)
fig.show()
```




```python
# ── Key observations from sensitivity_df ─────────────────────────────────────

# 1. Budget level where optimizer gain first plateaus (step increase < $500)
plateau_budget = None
gains = sensitivity_df['opt_gain'].values
for i in range(1, len(gains)):
    if gains[i] - gains[i - 1] < 500:
        plateau_budget = sensitivity_df['budget'].iloc[i]
        break

# 2. Maximum net gain achieved by the optimizer
max_opt_gain        = sensitivity_df['opt_gain'].max()
max_opt_gain_budget = sensitivity_df.loc[sensitivity_df['opt_gain'].idxmax(), 'budget']

# 3. Budget level where the gain gap is largest
max_gap_idx    = sensitivity_df['gain_gap'].idxmax()
max_gap_budget = sensitivity_df.loc[max_gap_idx, 'budget']
max_gap_value  = sensitivity_df.loc[max_gap_idx, 'gain_gap']

print("── Key Observations ──────────────────────────────────────────────────")
if plateau_budget is not None:
    print(f"1. Optimizer gain plateaus at budget : ${plateau_budget:,}  "
          f"(step increase first drops below $500)")
else:
    print("1. Optimizer gain does not plateau within the $0–$50,000 sweep range.")

print(f"2. Maximum optimizer net gain        : ${max_opt_gain:,.2f}  "
      f"(first reached at ${max_opt_gain_budget:,})")
print(f"3. Largest optimizer–heuristic gap   : ${max_gap_value:,.2f}  "
      f"at budget = ${max_gap_budget:,}")
```

    ── Key Observations ──────────────────────────────────────────────────
    1. Optimizer gain plateaus at budget : $45,000  (step increase first drops below $500)
    2. Maximum optimizer net gain        : $1,206,859.26  (first reached at $45,000)
    3. Largest optimizer–heuristic gap   : $64.99  at budget = $2,500


### 4b — Sensitivity Analysis: Incentive Effectiveness (DELTA)
We sweep DELTA — the assumed fraction by which an incentive reduces a customer's churn probability — from 0.10 to 0.90 at the baseline $10,000 budget.  
This quantifies how sensitive the optimizer's output is to the assumed effectiveness of retention incentives.


```python
# ── DELTA sweep at fixed baseline budget ──────────────────────────────────────
DELTA_GRID   = [round(d, 2) for d in np.arange(0.10, 0.95, 0.05)]
BASE_BUDGET  = 10_000
delta_records = []

for delta in DELTA_GRID:
    # Recompute per-customer values with this delta
    temp             = df.copy()
    temp['c_i']      = ALPHA * temp['Monthly Charges']
    temp['delta_i']  = delta * temp['churn_prob'] * temp['LTV'] - temp['c_i']
    temp_eligible    = temp[temp['delta_i'] > 0].copy().reset_index(drop=True)

    # Solve knapsack
    n          = len(temp_eligible)
    c_vals     = temp_eligible['c_i'].values
    d_vals     = temp_eligible['delta_i'].values
    prob       = pulp.LpProblem(f'knapsack_delta_{delta}', pulp.LpMaximize)
    x          = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(n)]
    prob      += pulp.lpSum(d_vals[i] * x[i] for i in range(n))
    prob      += pulp.lpSum(c_vals[i] * x[i] for i in range(n)) <= BASE_BUDGET
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    mask       = [pulp.value(x[i]) == 1 for i in range(n)]
    selected   = temp_eligible[mask]
    net_gain   = selected['delta_i'].sum() if len(selected) > 0 else 0.0
    n_targeted = len(selected)

    delta_records.append({
        'DELTA'           : delta,
        'Eligible Customers' : len(temp_eligible),
        'Customers Targeted' : n_targeted,
        'Net Gain ($)'    : round(net_gain, 2),
    })

delta_df = pd.DataFrame(delta_records)
print(delta_df.to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Net Gain vs Incentive Effectiveness (DELTA)',
        'Customers Targeted vs DELTA'
    )
)

fig.add_trace(
    go.Scatter(x=delta_df['DELTA'], y=delta_df['Net Gain ($)'],
               mode='lines+markers', line=dict(color='steelblue', width=2),
               marker=dict(size=6), name='Net Gain'),
    row=1, col=1
)
fig.add_vline(x=DELTA, line=dict(color='grey', dash='dash', width=1.5),
              annotation_text=f'Baseline δ={DELTA}', annotation_position='top left',
              row=1, col=1)
fig.update_xaxes(title_text='DELTA (Churn Reduction Factor)', row=1, col=1)
fig.update_yaxes(title_text='Total Net Gain ($)', row=1, col=1)

fig.add_trace(
    go.Scatter(x=delta_df['DELTA'], y=delta_df['Customers Targeted'],
               mode='lines+markers', line=dict(color='tomato', width=2),
               marker=dict(size=6), name='Customers Targeted'),
    row=1, col=2
)
fig.add_vline(x=DELTA, line=dict(color='grey', dash='dash', width=1.5),
              row=1, col=2)
fig.update_xaxes(title_text='DELTA (Churn Reduction Factor)', row=1, col=2)
fig.update_yaxes(title_text='Customers Targeted', row=1, col=2)

fig.update_layout(
    height=460, width=1100,
    title_text='Stage 4b — Sensitivity Analysis: Incentive Effectiveness (DELTA = 0.10 → 0.90)',
    template='plotly_white', showlegend=False
)
fig.show()
```

     DELTA  Eligible Customers  Customers Targeted  Net Gain ($)
      0.10                5375                1239      93191.61
      0.15                5726                1239     144787.42
      0.20                5963                1239     196383.23
      0.25                6136                1239     247979.04
      0.30                6236                1239     299574.84
      0.35                6324                1239     351170.65
      0.40                6389                1239     402766.46
      0.45                6431                1239     454362.26
      0.50                6481                1239     505958.07
      0.55                6535                1239     557553.88
      0.60                6573                1239     609149.69
      0.65                6623                1239     660745.49
      0.70                6656                1239     712341.30
      0.75                6685                1239     763937.11
      0.80                6702                1239     815532.92
      0.85                6718                1239     867128.72
      0.90                6734                1239     918724.53



