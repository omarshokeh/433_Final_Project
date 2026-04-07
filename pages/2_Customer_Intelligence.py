import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Customer Overview', layout='wide')
st.title('Customer Overview')

if 'scored_df' not in st.session_state or 'opt_targeted' not in st.session_state:
    st.info('Run the analysis first on the Configure page.')
    st.stop()

scored_df = st.session_state['scored_df']
opt_t     = st.session_state['opt_targeted']
total     = len(scored_df)

# ── Section 1 — At a Glance ───────────────────────────────────────────────────
st.subheader('Your Customer Base at a Glance')

at_risk_n   = (scored_df['churn_prob'] >= 0.70).sum()
at_risk_pct = at_risk_n / total * 100
avg_rev     = scored_df['Monthly Charges'].mean()
rev_at_risk = scored_df['expected_loss'].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Customers',          f'{total:,}')
c2.metric('Currently at Churn Risk',  f'{at_risk_n:,}  ({at_risk_pct:.1f}%)')
c3.metric('Avg Monthly Revenue',      f'${avg_rev:.2f}')
c4.metric('Revenue at Risk',          f'${rev_at_risk:,.0f}')

# ── Section 2 — Churn Risk Distribution ──────────────────────────────────────
st.subheader('Churn Risk Distribution')

bands = {
    'Low Risk'      : (scored_df['churn_prob'] < 0.30),
    'Medium Risk'   : (scored_df['churn_prob'] >= 0.30) & (scored_df['churn_prob'] < 0.60),
    'High Risk'     : (scored_df['churn_prob'] >= 0.60) & (scored_df['churn_prob'] < 0.80),
    'Critical Risk' : (scored_df['churn_prob'] >= 0.80),
}
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

counts  = [mask.sum() for mask in bands.values()]
labels  = list(bands.keys())
texts   = [f'{c:,}  ({c/total*100:.1f}%)' for c in counts]

fig_bar = go.Figure(go.Bar(
    x=counts,
    y=labels,
    orientation='h',
    marker_color=colors,
    text=texts,
    textposition='outside',
))
fig_bar.update_layout(
    height=280,
    template='plotly_white',
    xaxis_title='Number of Customers',
    yaxis=dict(autorange='reversed'),
    showlegend=False,
    margin=dict(l=10, r=120, t=20, b=40),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Section 3 — Contract Type Breakdown ──────────────────────────────────────
st.subheader('Contract Type Breakdown')

contract_counts = scored_df['Contract'].value_counts()
fig_donut = go.Figure(go.Pie(
    labels=contract_counts.index,
    values=contract_counts.values,
    hole=0.45,
    marker_colors=['#e74c3c', '#3498db', '#2ecc71'],
))
fig_donut.update_layout(
    title_text='Customers by Contract Type',
    height=360,
    template='plotly_white',
)
st.plotly_chart(fig_donut, use_container_width=True)
st.caption('Month-to-month customers churn at significantly higher rates.')

# ── Section 4 — Top Revenue at Risk (not yet targeted) ───────────────────────
st.subheader('Top 10 Customers by Revenue at Risk (not yet targeted)')

targeted_idx = set(opt_t.index)
untargeted   = scored_df[~scored_df.index.isin(targeted_idx)]
top10        = untargeted.nlargest(10, 'expected_loss').copy()

display = top10[['Monthly Charges', 'churn_prob', 'expected_loss']].copy()
display.insert(0, 'Customer ID', top10.index + 1)
display = display.rename(columns={
    'churn_prob'      : 'Churn Risk (%)',
    'Monthly Charges' : 'Monthly Revenue ($)',
    'expected_loss'   : 'Expected Loss ($)',
})
display['Churn Risk (%)']      = (display['Churn Risk (%)'] * 100).round(1)
display['Monthly Revenue ($)'] = display['Monthly Revenue ($)'].round(2)
display['Expected Loss ($)']   = display['Expected Loss ($)'].round(2)

st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)
