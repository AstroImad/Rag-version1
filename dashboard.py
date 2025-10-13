import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Sample Streamlit Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page", ["Overview", "Data", "About"])

if page == "Overview":
    st.header("Overview")
    st.write("Welcome to your Streamlit dashboard!")
    st.metric("Users", 1200, "+50")
    st.metric("Revenue", "$34,000", "+$1,200")
    st.metric("Conversion Rate", "4.5%", "+0.2%")

elif page == "Data":
    st.header("Data Table")
    data = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40],
        "C": ["X", "Y", "Z", "W"]
    })
    st.dataframe(data)

elif page == "About":
    st.header("About")
    st.write("This is a simple dashboard built with Streamlit.")

st.sidebar.markdown("---")
st.sidebar.write("© 2024 Your Company")