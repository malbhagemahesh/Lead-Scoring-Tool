import sys
import os
import random
import argparse
from datetime import datetime, timedelta


try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import unittest





# Hide Streamlit Deploy and extra UI
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# -----------------------------
# Data generation
# -----------------------------

def generate_sample_leads(n=50, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)

    companies = [f"Company {i+1}" for i in range(n)]
    leads_data = []
    for i in range(n):
        has_email = rng.random() > 0.15
        has_phone = rng.random() > 0.30
        has_linkedin = rng.random() > 0.25
        employees = rng.choice([10, 25, 50, 100, 250, 500, 1000])
        revenue = employees * (40000 + rng.uniform(20000, 120000))
        recent_funding = rng.random() > 0.65
        lead = {
            'company_name': companies[i],
            'contact_name': f"Person {i+1}",
            'title': rng.choice(['CEO', 'VP Sales', 'CTO', 'Marketing Manager', 'COO']),
            'email': f"lead{i+1}@company.com" if has_email else None,
            'phone': f"+1 (555) {rng.randint(100,999)}-{rng.randint(1000,9999)}" if has_phone else None,
            'linkedin_url': f"linkedin.com/in/person{i+1}" if has_linkedin else None,
            'company_size': employees,
            'estimated_revenue': revenue,
            'recent_funding': recent_funding
        }
        leads_data.append(lead)
    return pd.DataFrame(leads_data)

# -----------------------------
# Scoring algorithm
# -----------------------------

def calculate_lead_score(row, weights, rng=None):
    """
    Calculate lead score using weighted factors.
    `rng` is an optional instance of random.Random for deterministic behavior in tests or console mode.
    """
    if rng is None:
        _rand = random.random
    else:
        _rand = rng.random

    score = 0.0
    factors = []

    # Company size (0-25)
    if row['company_size'] >= 1000:
        pts = 25
    elif row['company_size'] >= 500:
        pts = 20
    elif row['company_size'] >= 100:
        pts = 15
    else:
        pts = 10
    score += pts * weights.get('company_size', 1.0)
    factors.append(("Company Size", pts * weights.get('company_size', 1.0)))

    # Revenue (0-25)
    rev_m = row['estimated_revenue'] / 1_000_000
    if rev_m >= 50:
        pts = 25
    elif rev_m >= 20:
        pts = 20
    elif rev_m >= 5:
        pts = 15
    else:
        pts = 10
    score += pts * weights.get('revenue', 1.0)
    factors.append(("Revenue", pts * weights.get('revenue', 1.0)))

    # Data completeness (0-20)
    comp = 0
    if row.get('email'):
        comp += 8
    if row.get('phone'):
        comp += 7
    if row.get('linkedin_url'):
        comp += 5
    score += comp * weights.get('data', 1.0)
    factors.append(("Data Completeness", comp * weights.get('data', 1.0)))

    # Engagement readiness (0-15)
    pts = 0
    if row.get('recent_funding'):
        pts += 10
    # small random boost to simulate current signals (kept deterministic if rng provided)
    if _rand() > 0.5:
        pts += 5
    score += pts * weights.get('engagement', 1.0)
    factors.append(("Engagement", pts * weights.get('engagement', 1.0)))

    # Title relevance (0-15)
    if row.get('title') in ['CEO', 'CTO', 'COO', 'VP Sales', 'Chief Revenue Officer']:
        pts = 15
    else:
        pts = 8
    score += pts * weights.get('title', 1.0)
    factors.append(("Title", pts * weights.get('title', 1.0)))

    return min(100, round(score, 2)), factors


def calculate_confidence(row):
    """A simple confidence metric based on presence of contact fields."""
    c = 0
    if row.get('email'):
        c += 33
    if row.get('phone'):
        c += 33
    if row.get('linkedin_url'):
        c += 34
    return int(c)

# -----------------------------
# Console mode helpers
# -----------------------------

def save_plots_and_csv(df, out_dir='/mnt/data'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'leads_console_output.csv')
    html_path = os.path.join(out_dir, 'leads_summary.html')

    df.to_csv(csv_path, index=False)

    # Histogram of scores
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['lead_score'], nbinsx=20, name='Lead Scores'))
    fig_hist.update_layout(title='Lead Score Distribution', xaxis_title='Score', yaxis_title='Count')

    # Pie chart of categories
    category_counts = df['category'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts.values, hole=0.4)])
    fig_pie.update_layout(title='Lead Categories')

    # Write both figures into a single HTML
    with open(html_path, 'w') as f:
        f.write('<html><head><meta charset="utf-8" /></head><body>')
        f.write('<h1>Lead Scoring Summary</h1>')
        f.write('<h2>Histogram</h2>')
        f.write(fig_hist.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write('<h2>Categories</h2>')
        f.write(fig_pie.to_html(full_html=False, include_plotlyjs=False))
        f.write('</body></html>')

    return csv_path, html_path


def run_console_demo(n=50, seed=42, weights=None, out_dir='/mnt/data'):
    print("Streamlit not available — running console demo.")
    df = generate_sample_leads(n=n, seed=seed)

    if weights is None:
        weights = {'company_size': 1.0, 'revenue': 1.0, 'data': 1.0, 'engagement': 1.0, 'title': 1.0}

    rng = random.Random(seed)
    results = df.apply(lambda row: calculate_lead_score(row, weights, rng), axis=1)
    df['lead_score'] = results.apply(lambda x: x[0])
    df['factors'] = results.apply(lambda x: x[1])
    df['confidence'] = df.apply(calculate_confidence, axis=1)
    df['category'] = df['lead_score'].apply(lambda s: 'Hot' if s >= 70 else 'Warm' if s >= 40 else 'Cold')

    # Summary
    print(f"Generated {len(df)} leads. Hot: {len(df[df['category']=='Hot'])}, Warm: {len(df[df['category']=='Warm'])}, Cold: {len(df[df['category']=='Cold'])}")
    print('\nTop 10 leads:')
    print(df[['company_name', 'contact_name', 'title', 'lead_score', 'confidence', 'category']].sort_values('lead_score', ascending=False).head(10).to_string(index=False))

    csv_path, html_path = save_plots_and_csv(df, out_dir=out_dir)
    print(f"\nSaved CSV -> {csv_path}")
    print(f"Saved plots HTML -> {html_path}")
    return df, csv_path, html_path

# -----------------------------
# Unit tests
# -----------------------------

class LeadScoringTests(unittest.TestCase):
    def test_calculate_confidence_full(self):
        row = {'email': 'a@b.com', 'phone': '+1', 'linkedin_url': 'ln'}
        self.assertEqual(calculate_confidence(row), 100)

    def test_calculate_confidence_empty(self):
        row = {'email': None, 'phone': None, 'linkedin_url': None}
        self.assertEqual(calculate_confidence(row), 0)

    def test_calculate_lead_score_high(self):
        # Create a lead that should score high when weights are all 1 and engagement weight is 0 (to avoid randomness)
        row = {
            'company_size': 1000,
            'estimated_revenue': 60_000_000,  # 60M -> rev_m=60
            'email': 'a@b.com',
            'phone': '123',
            'linkedin_url': 'ln',
            'recent_funding': True,
            'title': 'CEO'
        }
        weights = {'company_size': 1.0, 'revenue': 1.0, 'data': 1.0, 'engagement': 0.0, 'title': 1.0}
        # Use deterministic rng but engagement weight is 0 so randomness doesn't matter
        score, factors = calculate_lead_score(row, weights, rng=random.Random(0))
        # Expected: company_size 25 + revenue 25 + data 20 + title 15 = 85
        self.assertEqual(score, 85)

# -----------------------------
# Streamlit app 
# -----------------------------

def run_streamlit_app():
    st.set_page_config(page_title="Lead Scoring Tool", page_icon="🎯", layout="wide")
    st.markdown("""
    <style>
    .main-header {font-size: 34px; font-weight: bold}
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Lead Scoring & Prioritization Tool")
    st.markdown("A multi-factor lead scoring demo with confidence indicators and adjustable weights.")

    st.sidebar.header("⚖️ Adjust Weights")
    weights = {
        'company_size': st.sidebar.slider("Company Size", 0.0, 1.0, 1.0, 0.1),
        'revenue': st.sidebar.slider("Revenue", 0.0, 1.0, 1.0, 0.1),
        'data': st.sidebar.slider("Data Completeness", 0.0, 1.0, 1.0, 0.1),
        'engagement': st.sidebar.slider("Engagement Readiness", 0.0, 1.0, 1.0, 0.1),
        'title': st.sidebar.slider("Title Relevance", 0.0, 1.0, 1.0, 0.1)
    }

    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_leads(50)

    df = st.session_state.df.copy()

    # Apply scoring (use non-deterministic randomness for engagement when using Streamlit)
    results = df.apply(lambda row: calculate_lead_score(row, weights, rng=None), axis=1)
    df['lead_score'] = results.apply(lambda x: x[0])
    df['factors'] = results.apply(lambda x: x[1])
    df['confidence'] = df.apply(calculate_confidence, axis=1)
    df['category'] = df['lead_score'].apply(lambda s: 'Hot' if s >= 70 else 'Warm' if s >= 40 else 'Cold')

    # Dashboard
    st.subheader("📊 Lead Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔥 Hot Leads", len(df[df['category'] == "Hot"]))
    col2.metric("⚡ Warm Leads", len(df[df['category'] == "Warm"]))
    col3.metric("❄️ Cold Leads", len(df[df['category'] == "Cold"]))

    st.markdown("---")
    st.subheader("📋 Leads Table (Top 15)")
    st.dataframe(df[['company_name', 'contact_name', 'title', 'lead_score', 'confidence', 'category']].head(15))

    st.markdown("---")
    st.subheader("🔍 Lead Details")
    for _, row in df.head(10).iterrows():
        with st.expander(f"{row['company_name']} - {row['lead_score']} pts ({row['category']})"):
            st.write(f"Contact: {row['contact_name']} - {row['title']}")
            st.write(f"Email: {row.get('email')} | Phone: {row.get('phone')} | LinkedIn: {row.get('linkedin_url')}")
            st.write(f"Company Size: {row['company_size']} | Revenue: ${row['estimated_revenue']:,.0f}")
            st.write(f"Confidence Score: {row['confidence']}%")
            st.progress(row['confidence'] / 100)
            st.write("**Score Breakdown:**")
            for f in row['factors']:
                st.write(f"- {f[0]}: +{f[1]:.1f} pts")

# -----------------------------
# Entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Lead Scoring Tool (console fallback)')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--no-save', action='store_true', help='Do not save CSV/HTML outputs in console mode')
    args = parser.parse_args()

    if STREAMLIT_AVAILABLE:
        # Launch Streamlit app
        run_streamlit_app()
    else:
        # Console demo
        out_dir = '/mnt/data'
        df, csv_path, html_path = run_console_demo(n=50, seed=42, weights=None, out_dir=out_dir)
        if args.no_save:
            print('Files were generated but --no-save was set (no files written).')

        if args.test:
            print('\nRunning unit tests...')
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(LeadScoringTests)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            if not result.wasSuccessful():
                print('Some tests failed.')
                sys.exit(1)
            else:
                print('All tests passed.')


if __name__ == '__main__':
    main()
