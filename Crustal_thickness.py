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

# 加载预训练模型和交叉验证结果
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

st.subheader('Predict your own data')
uploaded_file = st.file_uploader("Upload a csv file; The file should include contents of SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Sr, Y, Rb, Ba, Hf, Nb, Ta and Th, without NaN value")
if uploaded_file is not None:
    Data = pd.read_csv(uploaded_file)
    X = DataFrame(Data, columns=Features)
    X_pt = pt.transform(X)
    Predicting_results = regr.predict(X_pt)
    Predicting_thickness = pd.DataFrame(Predicting_results, columns=['Crustal thickness/km'])
    st.dataframe(Predicting_thickness)

    def convert_df(df):
        return df.to_csv()
    st.download_button(
        label="Download predicting results as CSV",
        data=convert_df(Predicting_thickness),
        file_name='Predicting_thickness.csv',
        mime='text/csv')
     


