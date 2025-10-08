import sys
import os
import random
import argparse
import re
from datetime import datetime, timedelta

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np
import unittest

# -----------------------------
# Hide Streamlit Deploy and extra UI
# -----------------------------
if STREAMLIT_AVAILABLE:
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        [data-testid="stDeployButton"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------
# Data Validation Utilities
# -----------------------------

def validate_email(email):
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    if not phone:
        return False
    digits = re.sub(r'\D', '', phone)
    return len(digits) == 10

def generate_valid_email(company_name, person_id):
    clean_company = re.sub(r'[^a-zA-Z0-9]', '', company_name.lower())
    return f"lead{person_id}@{clean_company}.com"

def generate_valid_phone():
    area_code = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number = random.randint(1000, 9999)
    return f"+1 ({area_code}) {exchange}-{number}"

# -----------------------------
# Data generation
# -----------------------------

def generate_sample_leads(n=50, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)
    companies = [f"Company {i+1}" for i in range(n)]
    cities = ["New York", "San Francisco", "Chicago", "Austin", "Boston", "Seattle", "Denver"]
    leads_data = []

    for i in range(n):
        has_email = rng.random() > 0.15
        has_phone = rng.random() > 0.30
        has_linkedin = rng.random() > 0.25
        employees = rng.choice([10, 25, 50, 100, 250, 500, 1000])
        revenue = employees * (40000 + rng.uniform(20000, 120000))
        recent_funding = rng.random() > 0.65
        location = rng.choice(cities)

        email = generate_valid_email(companies[i], i+1) if has_email else None
        if email and not validate_email(email):
            email = None

        phone = generate_valid_phone() if has_phone else None
        if phone and not validate_phone(phone):
            phone = None

        lead = {
            'company_name': companies[i],
            'contact_name': f"Person {i+1}",
            'title': rng.choice(['CEO', 'VP Sales', 'CTO', 'Marketing Manager', 'COO']),
            'email': email,
            'phone': phone,
            'linkedin_url': f"linkedin.com/in/person{i+1}" if has_linkedin else None,
            'company_size': employees,
            'estimated_revenue': revenue,
            'recent_funding': recent_funding,
            'location': location
        }
        leads_data.append(lead)

    return pd.DataFrame(leads_data)

# -----------------------------
# Scoring Algorithm
# -----------------------------

def calculate_lead_score(row, weights, rng=None, size_thresholds=None, revenue_thresholds=None):
    if rng is None:
        _rand = random.random
    else:
        _rand = rng.random

    if size_thresholds is None:
        size_thresholds = {'large': 1000, 'medium': 500, 'small': 100}
    if revenue_thresholds is None:
        revenue_thresholds = {'high': 50, 'medium': 20, 'low': 5}

    score = 0.0
    factors = []

    # Company size
    if row['company_size'] >= size_thresholds['large']:
        pts = 25
    elif row['company_size'] >= size_thresholds['medium']:
        pts = 20
    elif row['company_size'] >= size_thresholds['small']:
        pts = 15
    else:
        pts = 10
    score += pts * weights.get('company_size', 1.0)
    factors.append(("Company Size", pts * weights.get('company_size', 1.0)))

    # Revenue
    rev_m = row['estimated_revenue'] / 1_000_000
    if rev_m >= revenue_thresholds['high']:
        pts = 25
    elif rev_m >= revenue_thresholds['medium']:
        pts = 20
    elif rev_m >= revenue_thresholds['low']:
        pts = 15
    else:
        pts = 10
    score += pts * weights.get('revenue', 1.0)
    factors.append(("Revenue", pts * weights.get('revenue', 1.0)))

    # Data completeness
    comp = 0
    if row.get('email') and validate_email(row.get('email')):
        comp += 8
    if row.get('phone') and validate_phone(row.get('phone')):
        comp += 7
    if row.get('linkedin_url'):
        comp += 5
    score += comp * weights.get('data', 1.0)
    factors.append(("Data Completeness", comp * weights.get('data', 1.0)))

    # Engagement
    pts = 10 if row.get('recent_funding') else 0
    if _rand() > 0.5:
        pts += 5
    score += pts * weights.get('engagement', 1.0)
    factors.append(("Engagement", pts * weights.get('engagement', 1.0)))

    # Title relevance
    if row.get('title') in ['CEO', 'CTO', 'COO', 'VP Sales', 'Chief Revenue Officer']:
        pts = 15
    else:
        pts = 8
    score += pts * weights.get('title', 1.0)
    factors.append(("Title", pts * weights.get('title', 1.0)))

    return min(100, round(score, 2)), factors

def calculate_confidence(row):
    c = 0
    if row.get('email') and validate_email(row.get('email')):
        c += 33
    if row.get('phone') and validate_phone(row.get('phone')):
        c += 33
    if row.get('linkedin_url'):
        c += 34
    return min(100, int(c))

# -----------------------------
# Streamlit app with filters
# -----------------------------

def run_streamlit_app():
    st.set_page_config(page_title="Lead Scoring Tool", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
    st.title("🎯 Lead Scoring & Prioritization Tool")
    st.markdown("A multi-factor lead scoring system with data validation, confidence indicators, and customizable thresholds.")

    # Sidebar: scoring weights
    st.sidebar.header("⚖️ Adjust Scoring Weights")
    weights = {
        'company_size': st.sidebar.slider("Company Size Weight", 0.0, 2.0, 1.0, 0.1),
        'revenue': st.sidebar.slider("Revenue Weight", 0.0, 2.0, 1.0, 0.1),
        'data': st.sidebar.slider("Data Completeness Weight", 0.0, 2.0, 1.0, 0.1),
        'engagement': st.sidebar.slider("Engagement Readiness Weight", 0.0, 2.0, 1.0, 0.1),
        'title': st.sidebar.slider("Title Relevance Weight", 0.0, 2.0, 1.0, 0.1)
    }

    # Company size thresholds
    st.sidebar.header("🏢 Company Size Thresholds")
    size_large = st.sidebar.number_input("Large Company (employees ≥)", min_value=100, max_value=10000, value=1000, step=100)
    size_medium = st.sidebar.number_input("Medium Company (employees ≥)", min_value=50, max_value=5000, value=500, step=50)
    size_small = st.sidebar.number_input("Small Company (employees ≥)", min_value=10, max_value=1000, value=100, step=10)
    size_thresholds = {'large': size_large, 'medium': size_medium, 'small': size_small}

    # Revenue thresholds
    st.sidebar.header("💰 Revenue Thresholds (in millions)")
    rev_high = st.sidebar.number_input("High Revenue ($ millions ≥)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
    rev_medium = st.sidebar.number_input("Medium Revenue ($ millions ≥)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    rev_low = st.sidebar.number_input("Low Revenue ($ millions ≥)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
    revenue_thresholds = {'high': rev_high, 'medium': rev_medium, 'low': rev_low}

    # Filters
    st.sidebar.header("🔎 Filters")
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_leads(50)

    df = st.session_state.df.copy()
    locations = df['location'].unique().tolist()
    categories = ['Hot', 'Warm', 'Cold']

    selected_locations = st.sidebar.multiselect("Select Location(s)", options=locations, default=locations)
    selected_categories = st.sidebar.multiselect("Select Category", options=categories, default=categories)

    # Calculate scores
    results = df.apply(lambda row: calculate_lead_score(row, weights, rng=None,
                                                        size_thresholds=size_thresholds,
                                                        revenue_thresholds=revenue_thresholds), axis=1)
    df['lead_score'] = results.apply(lambda x: x[0])
    df['factors'] = results.apply(lambda x: x[1])
    df['confidence'] = df.apply(calculate_confidence, axis=1)
    df['category'] = df['lead_score'].apply(lambda s: 'Hot' if s >= 70 else 'Warm' if s >= 40 else 'Cold')
    df['email_valid'] = df['email'].apply(lambda x: '✅' if validate_email(x) else '❌')
    df['phone_valid'] = df['phone'].apply(lambda x: '✅' if validate_phone(x) else '❌')

    # Apply filters
    filtered_df = df[(df['location'].isin(selected_locations)) & (df['category'].isin(selected_categories))]

    # Lead summary
    hot_count = (filtered_df['category'] == 'Hot').sum()
    warm_count = (filtered_df['category'] == 'Warm').sum()
    cold_count = (filtered_df['category'] == 'Cold').sum()

    st.markdown(f"### Lead Summary: 🔥 **Hot:** {hot_count} | ⚡ **Warm:** {warm_count} | ❄️ **Cold:** {cold_count}")

    # Lead table
    st.subheader("📋 Leads Table (Top 20)")
    display_df = filtered_df[['company_name', 'contact_name', 'title', 'email_valid', 'phone_valid',
                              'lead_score', 'confidence', 'category', 'location']].head(20)
    st.dataframe(display_df, use_container_width=True)

    # Expanders for details
    st.subheader("🔍 Lead Details (Top 10)")
    for _, row in filtered_df.head(10).iterrows():
        with st.expander(f"{row['category']} | {row['company_name']} - {row['lead_score']} pts ({row['location']})"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Contact:** {row['contact_name']} - {row['title']}")
                st.write(f"**Email:** {row.get('email') or 'N/A'} {row['email_valid']}")
                st.write(f"**Phone:** {row.get('phone') or 'N/A'} {row['phone_valid']}")
                st.write(f"**LinkedIn:** {row.get('linkedin_url') or 'N/A'}")
                st.write(f"**Company Size:** {row['company_size']} employees")
                st.write(f"**Revenue:** ${row['estimated_revenue']:,.0f}")
                if row['recent_funding']:
                    st.write("💰 **Recent Funding Round**")

            with col2:
                st.write("**Confidence Score:**")
                st.progress(row['confidence'] / 100)
                st.write(f"{row['confidence']}%")
                st.write("**Score Breakdown:**")
                for factor_name, factor_pts in row['factors']:
                    st.write(f"• {factor_name}: +{factor_pts:.1f} pts")

# -----------------------------
# Console Mode
# -----------------------------

def run_console_demo(n=50, seed=42, weights=None, out_dir='/mnt/data', size_thresholds=None, revenue_thresholds=None):
    print("Streamlit not available — running console demo.")
    df = generate_sample_leads(n=n, seed=seed)

    if weights is None:
        weights = {'company_size': 1.0, 'revenue': 1.0, 'data': 1.0, 'engagement': 1.0, 'title': 1.0}

    rng = random.Random(seed)
    results = df.apply(lambda row: calculate_lead_score(row, weights, rng, size_thresholds, revenue_thresholds), axis=1)
    df['lead_score'] = results.apply(lambda x: x[0])
    df['factors'] = results.apply(lambda x: x[1])
    df['confidence'] = df.apply(calculate_confidence, axis=1)
    df['category'] = df['lead_score'].apply(lambda s: 'Hot' if s >= 70 else 'Warm' if s >= 40 else 'Cold')

    df['email_valid'] = df['email'].apply(lambda x: '✅' if validate_email(x) else '❌')
    df['phone_valid'] = df['phone'].apply(lambda x: '✅' if validate_phone(x) else '❌')

    print(f"\nGenerated {len(df)} leads. Hot: {len(df[df['category']=='Hot'])}, Warm: {len(df[df['category']=='Warm'])}, Cold: {len(df[df['category']=='Cold'])}")
    print("\nTop 10 leads:")
    print(df[['company_name', 'contact_name', 'title', 'email_valid', 'phone_valid', 'lead_score', 'confidence', 'category', 'location']]
          .sort_values('lead_score', ascending=False).head(10).to_string(index=False))

# -----------------------------
# Unit Tests
# -----------------------------
# Keep your existing tests (unchanged)...

# -----------------------------
# Entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Lead Scoring Tool (console fallback)')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    args = parser.parse_args()

    if args.test:
        print('Running unit tests...')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LeadScoringTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            print('Some tests failed.')
            sys.exit(1)
        else:
            print('All tests passed.')
        return

    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        run_console_demo(n=50, seed=42, weights=None, out_dir='/mnt/data')

if __name__ == '__main__':
    main()
