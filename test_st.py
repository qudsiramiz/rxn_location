import streamlit as st
import pandas as pd

df = pd.DataFrame({"B_{IMF}": [1], "Bᵢₘ_f": [2], "$B_{IMF}$": [3]})
st.data_editor(df)
