pip install sklearn
pip install xgboost

import numpy as np
import pandas as pd

import pickle
import streamlit as st
import time

model = pickle.load(open('model.h5', 'rb'))

st.header("Sales Predictor")

# <=== Columns ===>
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8 = st.columns(2)

# <=== Encoded Values for Features ===>
item_types = {'Fruits and Vegetables':0, 'Snack Foods':1, 'Household':2,'Frozen Foods':3, 'Dairy':4, 'Canned':5, 
              'Baking Goods':6, 'Health and Hygiene':7, 'Soft Drinks':8, 'Meat':9, 'Breads':10, 'Hard Drinks':11, 
              'Others':12, 'Starchy Foods':13, 'Breakfast':14, 'Seafood':15}

fat_content = {'LF': 0, 'Low Fat': 1, 'Regular': 2, 'low fat':3, 'reg':4}
outlet_ids = {'OUT013': 0, 'OUT018':1, 'OUT019':2, 'OUT027':3, 'OUT035':4, 'OUT046':5, 'OUT049':6}
outlet_size = {"Small": 0, "Medium": 1, "High": 2}
outlet_location = {"Tier-1": 0, "Tier-2": 1, 'Tier-3': 2}
outlet_type = {"Grocery Store":0, "Supermarket Type1":1, "Supermarket Type2":2, "Supermarket Type3": 3}

# === Column Names ===
columns = ['Item Type', 'Item Weight', 'Item MRP','Fat Content', 'Outlet Identifier', 'Outlet Size',
           'Outlet Location Type', 'Outlet Type']

st.sidebar.title("Welcome to Big Mart Sales Predictor")
st.sidebar.markdown("This web application predicts the overall sales based upon these feature values.")

st.sidebar.dataframe(columns)

values = np.zeros(shape=(1, 8))

with col1: 
    item = st.selectbox(f"{columns[0]}", options=item_types.keys())
    values[0, 0] = item_types[item]

with col2: 
    values[0, 1] = st.number_input(label=f'{columns[1]}', min_value=0.0, max_value=21.35)

with col3: 
    values[0, 2] = st.number_input(label=f"{columns[2]}", min_value=0.0, max_value=277.0)

with col4:
    fat = st.selectbox(f"{columns[3]}", options=fat_content.keys())
    values[0, 3] = fat_content[fat]

with col5:
    id = st.selectbox(f"{columns[4]}", options=outlet_ids.keys())
    values[0, 4] = outlet_ids[id]

with col6:
    size = st.selectbox(f"{columns[5]}", options=outlet_size.keys())
    values[0, 5] = outlet_size[size]

with col7: 
    location = st.selectbox(f"{columns[6]}", options=outlet_location.keys())
    values[0, 6] = outlet_location[location]

with col8: 
    outlet = st.selectbox(f"{columns[7]}", options=outlet_type.keys())
    values[0, 7] = outlet_type[outlet]

def predict():
    st.divider()
    pred_value = model.predict(values)[0]
    st.progress(value=0, text="Running...")
    time.sleep(1)
    st.subheader(f"Predicted Sales: {np.round(pred_value)}$")

st.button(label='See Prediction', on_click=predict)
