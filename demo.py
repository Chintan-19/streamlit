import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 

st.set_option('deprecation.showPyplotGlobalUse', False)
data = pd.read_csv("data//data.csv")
x = np.array(data['YearExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))


st.title("Salary Predictor")
st.image("data//sal.jpg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute","About us"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(data)
    
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    val = st.slider("Filter data using years",0,5)
    data = data.loc[data["YearExperience"]>= val]
    if graph == "Non-Interactive":
        
        plt.figure(figsize = (10,5))
        plt.scatter(data["YearExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout =go.Layout(
            xaxis = dict(range=[0,10]),
            yaxis = dict(range =[0,210000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearExperience"], y=data["Salary"], mode='markers'),layout = layout)
        st.plotly_chart(fig)
    
if nav == "Prediction":
    st.header("Know your Salary")
    val = st.number_input("Enter you exp",0.00,5.00,step = 0.2)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    if st.button("submit"):
        to_add = {"YearExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//data.csv",mode='a',header = False,index= False)
        st.success("Submitted")

if nav == "About us":
    st.header("Welcome to our Salary Predictor App")        
    audio_file = open("data//Recording.m4a", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/m4a')
    audio_file.close()