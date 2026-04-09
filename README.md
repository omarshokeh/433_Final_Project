## Running the Notebook 
notebook already rendered in churn_pipeline.md (no plots)

```bash
pip install -r requirements.txt
jupyter notebook churn_pipeline.ipynb
```
Run all cells top to bottom. Each stage depends on the previous.

---

## Running the Streamlit App
Open codespace or local clone repo

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
No setup needed — the IBM Telco dataset loads automatically as demo data.
Upload your own `.xlsx` or `.csv` file to use custom customer data.

---

## Key Results
| Metric | Value |
|--------|-------|
| Model AUC (hold-out) | 0.8426 |
| Churn recall | 79% |
| Customers targeted @ $10k budget | 1,239 |
| Expected net gain | $505,958 |
| Return on investment | 50.38x |
| Budget plateau | $45,000 |

---

## Dependencies
See `requirements.txt`. Key libraries: scikit-learn, PuLP, Streamlit, Plotly, pandas.
