## Running the Notebook
The notebook is already rendered (no plots) in `churn_pipeline.md` for quick reference.

To run it interactively:

```bash
pip install -r requirements.txt
jupyter notebook churn_pipeline.ipynb
```

- Run all cells top to bottom — each stage depends on the previous
- The dataset loads from `data/Telco_customer_churn.xlsx` automatically
- Stage 1 trains the model and attaches `churn_prob` to the dataframe
- Stage 2 computes LTV and expected loss per customer
- Stage 3 runs the binary knapsack optimizer (requires PuLP/CBC)
- Stage 4 runs the budget and DELTA sensitivity sweeps — these take ~30–60 seconds due to repeated solver calls

> **Note:** If you're on GitHub Codespaces and Jupyter isn't found, use `python -m jupyter notebook churn_pipeline.ipynb`

---

## Running the Streamlit App
Open in Codespaces or clone locally, then:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

> **Note:** Use `python -m streamlit` instead of `streamlit` directly — this avoids PATH issues in Codespaces.

The app opens to the **Configure** page:
1. The IBM Telco dataset loads automatically as demo data
2. Use the sidebar slider to set a retention budget ($1k–$100k)
3. Expand **Advanced Settings** to adjust incentive cost rate (α) and retention effectiveness (δ)
4. Click **Run Analysis** — the full pipeline runs and results appear below
5. Download the target list as CSV using the button at the bottom

To use your own data, upload a `.xlsx` or `.csv` file in the sidebar — it replaces the demo dataset. Required columns match the IBM Telco schema (Monthly Charges, Churn Value, etc.).

Navigate to **Customer Intelligence** (sidebar) after running to see portfolio-level churn diagnostics.

---

## Dependencies
See `requirements.txt`. Key libraries:

| Library | Purpose |
|---|---|
| scikit-learn | Logistic regression, preprocessing, cross-validation |
| PuLP | Binary knapsack solver (CBC backend) |
| Streamlit | Interactive dashboard |
| Plotly | Charts and visualisations |
| pandas / openpyxl | Data loading and manipulation |
