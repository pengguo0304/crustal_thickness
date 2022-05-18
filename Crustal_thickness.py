# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:26:28 2022

@author: SUSTC
"""

# In[1]
import streamlit as st
import numpy as np
import pandas as pd
st.markdown('Peng Guo')
st.title('Quantify crustal thickness using the machine learning method')
st.subheader('Algorithm')
st.text('Extremely Randomized Tress proposed by Geurts et al.2006')

dataFile = 'CrustThickness_5Ma_Tibet_Normalized.csv' #导入训练数据
data = np.loadtxt(dataFile, dtype=float, delimiter=',',comments='S')
x,y = np.split(data, (32,), axis=1)  #分割特征和分类结果
x = x[:,:32] 

columns = ('SiO2','TiO2','Al2O3','FeO','MnO','MgO','Ca0','Na2O','K2O','P2O5','La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sr', 'Y', 'Rb', 'Ba', 'Hf', 'Nb', 'Ta', 'Th','Crustal thickness')
data = pd.DataFrame(data,columns=columns)
if st.checkbox('Show Training data'):
    st.subheader('Training data')
    st.dataframe(data)

# In[2]
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

pt = StandardScaler()
pt.fit(x)
x_pt = pt.transform(x)

kf10 = KFold(n_splits=10,shuffle=True) 
regr = ExtraTreesRegressor(n_estimators=500,max_features='auto')
y_predict = np.zeros((y.size,)) #创建预测值数组
i=0
for train_index,test_index in kf10.split(x_pt):
    x_train,x_test = x_pt[train_index],x_pt[test_index]
    y_train,y_test = y[train_index],y[test_index]
    regr.fit(x_train,y_train.ravel())
    y_predict[test_index]=regr.predict(x_test)
#np.savetxt('y_predict.dat',y_predict,fmt='%.2f')
r2_test = r2_score(y,y_predict)
RMSE = mean_squared_error(y,y_predict)**0.5 #均方根误差

# In[3]
st.subheader('Modeling result')
y_predict = y_predict.ravel()
fig,ax = plt.subplots(figsize=(6,6))
ax.scatter(y,y_predict,25,color='r')
ax.plot([0, 90], [0, 90], linestyle='--', lw=2, color='b',alpha=.8)
ax.plot([10, 90], [0, 80], linestyle='--', lw=2, color='g',alpha=.5)
ax.plot([0, 80], [10, 90], linestyle='--', lw=2, color='g',alpha=.5)
ax.text(10, 75, r'$R^2 = {:.3f}$'.format(r2_test),fontsize=15)
ax.text(10, 70, r'RMSE = {:.1f}'.format(RMSE),fontsize=15)
ax.set_title('Crustal thickness')
ax.set_xlabel('Observed',fontsize=12)
ax.set_ylabel('Predicted',fontsize=12)
ax.axis([0,90,0,90])
st.pyplot(fig)

# In[4]
st.subheader('Predict your own data')
uploaded_file = st.file_uploader("Choose a csv format file")
if uploaded_file is not None:
     dataframe = np.loadtxt(uploaded_file, dtype=float, delimiter=',',comments='S')
     regr.fit(x_pt,y.ravel())
     X_pt = pt.transform(dataframe)
     Predicting_results = regr.predict(X_pt)
     Predicting_thickness = pd.DataFrame(Predicting_results, columns = ['Crustal thickness/km'])
     #np.savetxt('Predicting_results.csv',Predicting_thickness,fmt='%.2f')
     st.dataframe(Predicting_thickness)
     
     def convert_df(df):
          return df.to_csv()
     Predicting_thickness = convert_df(Predicting_thickness)
     st.download_button(
          label="Download predicting results as CSV",
          data=Predicting_thickness,
          file_name='Predicting_thickness.csv',
          mime='text/csv')
     
