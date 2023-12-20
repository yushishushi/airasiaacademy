import streamlit as st
import numpy as np
import pandas as pd

st.header("My first Streamlit App")

option = st.sidebar.selectbox(
    'Select a mini project',
     ['line chart','map','T n C'])

st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
show = st.checkbox('I agree the terms and conditions')

if show:
    st.write(pd.DataFrame({
        'Students': ['John', 'Lofa', 'Siti', 'Amy'],
        'Attendance Status': ['yes', 'yes', 'yes', 'no']
    }))
