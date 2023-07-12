import numpy as np
import pandas as pd
import pickle
import streamlit as st
import time

model = pickle.load(open('Model.h5', 'rb'))

st.header("Big Mart Sales Prediction System")

# <=== Columns ===>
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8, col9 = st.columns(3)

# <=== Encoded Values for Features ===>
fat_encode = {'Low Fat': 0, 'Regular': 1}

outlet_id = {'OUT027': 0, 'OUT013': 1, 'OUT049': 2, 'OUT046': 3, 'OUT035': 4, 
             'OUT045': 5, 'OUT018': 6, 'OUT017': 7, 'OUT010': 8, 'OUT019': 9
            }

outlet_size = {'Small': 0, 'Medium': 1, 'High': 2}

outlet_loc_type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}

outlet_type = {'Supermarket Type1': 0, 'Grocery Store': 3, 'Supermarket Type3': 2, 'Supermarket Type2': 1}

item_type = {
    'Fruits and Vegetables': 0, 'Snack Foods': 1, 'Household': 2,
    'Frozen Foods': 3, 'Dairy': 4, 'Canned': 5, 'Baking Goods': 6,
    'Health and Hygiene': 7, 'Soft Drinks': 8, 'Meat': 9, 
    'Breads': 10, 'Hard Drinks': 11, 'Others': 12,
    'Starchy Foods': 13, 'Breakfast': 14,'Seafood': 15
 }

st.sidebar.title("Welcome to Big Mart Sales Predictor")
st.sidebar.markdown("This web application predicts the overall sales based upon these feature values.")

st.sidebar.markdown("""
`Item_Weight` ---- Weight of product
                    
`Item_Fat_Content` ---- Whether the product is low fat or not
                    
`Item_Visibility` ---- Percentage the total display area of all products in a store allocated 
to the particular product
                    
`Item_Type` ---- The category to which the product belongs
                    
`Item_MRP` ---- Maximum Retail Price (list price) of the product
                    
`Outlet_Identifier` ---- Unique store ID
                    
`Outlet_Establishment_Year` ---- The year in which the store was established
                    
`Outlet_Size` ---- The size of the store in terms of ground area covered
                    
`Outlet_Location_Type` ---- The type of city in which the store is located
                    
`Outlet_Type` ---- Whether the outlet is just a grocery store or some sort of supermarket
""")

values = np.zeros(shape=(1, 10))

with col1: 
    values[0, 0]= st.slider(label="Item Weight", min_value=0.00, max_value=25.00)

with col2: 
    values[0, 1] = fat_encode[st.selectbox(label='Item Fat Content', options=fat_encode.keys())]

with col3: 
    values[0, 2] = st.slider(label="Item Visibility", min_value=0.00, max_value=0.20)

with col4:
    values[0, 3] = item_type[st.selectbox(label="Item Type", options=item_type.keys())]

with col5:
    values[0, 4] = st.slider("Item MRP", min_value=0.00, max_value=270.00)

with col6:
    values[0, 5] = outlet_id[st.selectbox(label="Outlet Identifier", options=outlet_id.keys())]

with col7: 
    values[0, 6] = st.slider(label='Outlet Establishment Year', min_value=1970, max_value=2010)

with col8: 
    values[0, 7] = outlet_size[st.selectbox(label="Outlet Size", options=outlet_size.keys())]

with col9: 
    values[0, 8] = outlet_loc_type[st.selectbox(label="Outlet Location Type", options=outlet_loc_type.keys())]

values[0, 9] = outlet_type[st.selectbox(label="Outlet Type", options=outlet_type.keys())]

st.divider()

def predict():
    pred_value = model.predict(values)[0]
    st.progress(value=100)
    st.subheader(f"Predicted Sales: {np.round(pred_value)}$")

st.button(label='See Sales', on_click=predict)
