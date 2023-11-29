import streamlit as st
import pickle

# Load the pickled model
with open('modelprofit.pkl', 'rb') as file:
    modelprofit = pickle.load(file)

# Create the Streamlit web app
st.header("Streamlit demo")

st.sidebar.header("This is a web app")

product = st.sidebar.slider("Select X to get yhat", 0, 80, 5)

st.write("PRODUCT is:", product)

profit = modelprofit.predict([[product]])

st.write("b0 is", round(modelprofit.intercept_, 3))
st.write("b1 is", round(modelprofit.coef_[0], 3))
st.write("PROFIT is", profit)
