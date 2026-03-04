# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:26:28 2022

@author: Peng Guo
"""

# -*- coding: utf-8 -*-
from pandas import DataFrame
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_squared_error

st.markdown('Peng Guo')
st.title('Quantify crustal thickness using the machine learning method')
st.text('Based on the Extremely Randomized Trees algorithm proposed by Geurts et al.2006')

dataFile = pd.read_csv('CrustThickness.csv')
Features = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P2O5', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sr', 'Y', 'Rb', 'Ba', 'Hf', 'Nb', 'Ta', 'Th']
x = DataFrame(dataFile, columns=Features)
y = dataFile.Crustal_Thickness

st.subheader('Training data')
st.dataframe(dataFile)

# Load pretrained model and cross-validation results
regr = joblib.load('model.pkl')
pt = joblib.load('scaler.pkl')
x_pt = pt.transform(x)
y_predict = np.load('y_cv_predict.npy')
y_true = np.load('y_true.npy')

r2_test = r2_score(y_true, y_predict)
RMSE = mean_squared_error(y_true, y_predict)**0.5

st.subheader('Modeling result')
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_true, y_predict, 25, color='r')
ax.plot([0, 90], [0, 90], linestyle='--', lw=2, color='b', alpha=.8)
ax.plot([10, 90], [0, 80], linestyle='--', lw=2, color='g', alpha=.5)
ax.plot([0, 80], [10, 90], linestyle='--', lw=2, color='g', alpha=.5)
ax.text(10, 75, r'$R^2 = {:.3f}$'.format(r2_test), fontsize=15)
ax.text(10, 70, r'RMSE = {:.1f}'.format(RMSE), fontsize=15)
ax.set_title('Crustal thickness')
ax.set_xlabel('Observed', fontsize=12)
ax.set_ylabel('Predicted', fontsize=12)
ax.axis([0, 90, 0, 90])
st.pyplot(fig)

# Predict your own data
st.subheader('Predict your own data')

# Download input template
template_df = pd.DataFrame(columns=Features)
st.download_button(
    label="📥 Download input template (CSV)",
    data=template_df.to_csv(index=False),
    file_name='input_template.csv',
    mime='text/csv'
)

uploaded_file = st.file_uploader(
    "Upload a CSV file with the following features: " + ", ".join(Features) + ". No NaN values allowed.",
    type=['csv']
)

if uploaded_file is not None:
    Data = pd.read_csv(uploaded_file)

    # Preview uploaded data
    st.write("**Preview of uploaded data:**")
    st.dataframe(Data.head())

    # Check for missing columns
    missing_features = [f for f in Features if f not in Data.columns]
    if missing_features:
        st.error(f"Missing columns: {missing_features}. Please check your file.")
    # Check for NaN values
    elif Data[Features].isnull().any().any():
        nan_cols = Data[Features].columns[Data[Features].isnull().any()].tolist()
        st.error(f"NaN values detected in columns: {nan_cols}. Please clean your data.")
    else:
        X = DataFrame(Data, columns=Features)
        X_pt = pt.transform(X)
        Predicting_results = regr.predict(X_pt)

        # Merge input data with prediction results
        Result_df = Data.copy()
        Result_df['Predicted_Crustal_Thickness_km'] = Predicting_results

        st.write("**Prediction results:**")
        st.dataframe(Result_df)

        # Plot spatial distribution if lat/lon columns exist
        lat_col = next((c for c in Data.columns if c.lower() in ['lat', 'latitude']), None)
        lon_col = next((c for c in Data.columns if c.lower() in ['lon', 'long', 'longitude']), None)
        if lat_col and lon_col:
            st.write("**Spatial distribution of predicted crustal thickness:**")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sc = ax2.scatter(Data[lon_col], Data[lat_col], c=Predicting_results, cmap='RdYlBu_r', s=30)
            plt.colorbar(sc, ax=ax2, label='Crustal Thickness (km)')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('Predicted Crustal Thickness')
            st.pyplot(fig2)

        # Download prediction results
        st.download_button(
            label="📤 Download prediction results (CSV)",
            data=Result_df.to_csv(index=False),
            file_name='Predicted_crustal_thickness.csv',
            mime='text/csv'
        )
     



