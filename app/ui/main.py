import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ui.model_analysis import model_analysis_page
from app.ui.risk_dashboard import risk_dashboard_page

import streamlit as st
st.set_page_config(page_title="Options Pricing Platform", layout="wide")

page = st.sidebar.selectbox("Pages", ["Model Analysis", "Risk Dashboard"])
if page == "Model Analysis":
    model_analysis_page()
elif page == "Risk Dashboard":
    risk_dashboard_page() 