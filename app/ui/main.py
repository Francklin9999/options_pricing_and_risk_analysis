import sys
import os
from pathlib import Path

def find_project_root():
    try:
        current = Path(__file__).resolve()
    except NameError:
        current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'app').is_dir():
            return str(parent)
    raise RuntimeError("Could not find project root containing 'app' directory.")

project_root = find_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ui.model_analysis import model_analysis_page
from app.ui.risk_dashboard import risk_dashboard_page

import streamlit as st
st.set_page_config(page_title="Options Pricing Platform", layout="wide")

page = st.sidebar.selectbox("Pages", ["Model Analysis", "Risk Dashboard"])
if page == "Model Analysis":
    model_analysis_page()
elif page == "Risk Dashboard":
    risk_dashboard_page() 