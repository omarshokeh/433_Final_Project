import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import load_and_clean, train_model, score_customers, run_optimization

st.set_page_config(page_title='Retention Budget Planner', layout='wide')

DEMO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'Telco_customer_churn.xlsx',
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Settings')

    uploaded = st.file_uploader('Customer data (.xlsx or .csv)', type=['xlsx', 'csv'])
    source   = uploaded if uploaded is not None else (DEMO_PATH if os.path.exists(DEMO_PATH) else None)

    if uploaded is None and os.path.exists(DEMO_PATH):
        st.caption('✓ Demo data loaded — IBM Telco dataset (7,032 customers). Upload your own file to replace it.')

    budget = st.slider('Retention Budget ($)', 1_000, 100_000, 10_000, step=1_000,
                       format='$%d')

    with st.expander('Advanced Settings', expanded=False):
        alpha = st.slider('Incentive Cost Rate (% of monthly revenue)',
                          0.01, 0.30, 0.10, step=0.01, format='%.2f')
        delta = st.slider('Assumed Retention Effectiveness',
                          0.10, 0.90, 0.50, step=0.05, format='%.2f')

    run_clicked = st.button('Run Analysis', type='primary', use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title('Customer Retention Budget Planner')
st.write('Upload your customer data, set your budget, and get a prioritized list of customers to contact.')

if run_clicked:
    if source is None:
        st.error('Please upload a dataset or place Telco_customer_churn.xlsx in the data/ folder.')
        st.stop()

    st.text('Running analysis...')

    df        = load_and_clean(source)
    pipeline, _, _, _, _, _ = train_model(df)
    scored_df = score_customers(df, pipeline)
    opt_t, _, _, _, summary_df = run_optimization(scored_df, budget, alpha, delta)

    # Persist
    st.session_state['scored_df']   = scored_df
    st.session_state['opt_targeted'] = opt_t
    st.session_state['alpha']        = alpha
    st.session_state['delta']        = delta
    st.session_state['budget']       = budget
    st.session_state['ran']          = True

if st.session_state.get('ran'):
    opt_t = st.session_state['opt_targeted']

    n    = len(opt_t)
    cost = opt_t['c_i'].sum()     if n > 0 else 0.0
    gain = opt_t['delta_i'].sum() if n > 0 else 0.0
    roi  = (gain / cost)           if cost > 0 else 0.0

    st.success('Analysis complete — view results below')

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Customers to Target', f'{n:,}')
    c2.metric('Budget Used',         f'${cost:,.2f}')
    c3.metric('Expected Net Gain',   f'${gain:,.2f}')
    c4.metric('Return on Investment', f'{roi:.2f}x')

    # ── Target table ──────────────────────────────────────────────────────────
    if n > 0:
        display = opt_t[['Monthly Charges', 'churn_prob', 'c_i', 'delta_i']].copy()
        display.insert(0, 'Customer ID', opt_t.index + 1)
        display = display.rename(columns={
            'churn_prob'      : 'Churn Risk',
            'Monthly Charges' : 'Monthly Revenue ($)',
            'c_i'             : 'Incentive Cost ($)',
            'delta_i'         : 'Expected Net Gain ($)',
        })
        display['Churn Risk'] = (display['Churn Risk'] * 100).round(1).astype(str) + '%'
        display['Monthly Revenue ($)']   = display['Monthly Revenue ($)'].round(2)
        display['Incentive Cost ($)']    = display['Incentive Cost ($)'].round(2)
        display['Expected Net Gain ($)'] = display['Expected Net Gain ($)'].round(2)
        display = display.sort_values('Expected Net Gain ($)', ascending=False)

        st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)

        csv = display.to_csv(index=False).encode()
        st.download_button(
            'Download Target List (CSV)',
            data=csv,
            file_name='target_list.csv',
            mime='text/csv',
        )
    else:
        st.info('No customers can be profitably targeted at this budget level. Try increasing the budget or adjusting Advanced Settings.')
