"""
The script develops the user application. This application accepts user inputs and 
feeds them to the model. The resulting model predictions are displayed in the application.    
"""

# from pyexpat import model
# from webbrowser import BaseBrowser
import pandas as pd
import streamlit as st
import pickle as pkl
import os


df = pd.read_csv("data_given/online_shoppers_intention.csv")
# df = pd.read_csv("data/raw/raw_data.csv")
# df = pd.read_csv("data/processed/train_data.csv")

st.title('Online Shoppers Intention Prediction')

st.write("This app is based on 17 inputs that predict wheather an online visitor will shop or not? Using this app, a bank can identify specific customer segments; that will make deposits.")
st.write("Please use the following form to get started!")
st.markdown('<p class="big-font">(NOTE: For convinience, usual values are pre-selected in the form.)</p>', unsafe_allow_html=True)


# selecting administrative pages
admin_list = sorted(df.Administrative.unique())
st.subheader("Select number of visits on administrative pages")
selected_admin = st.selectbox("", admin_list)    



# selecting administrative duration
admin_dur = sorted(df.Administrative_Duration.unique())
st.subheader("Select visit duration of administrative pages")
selected_admin_dur = st.selectbox("", admin_dur)    


# selecting informational pages
info_list = sorted(df.Informational.unique())
st.subheader("Select number of visits on informational pages")
selected_info = st.selectbox("", info_list)    


# selecting informational duration
info_dur = sorted(df.Informational_Duration.unique())
st.subheader("Select duration of visit on informational pages")
selected_info_dur = st.selectbox("", info_dur)    


# selecting ProductRelated
prod_pages = sorted(df.ProductRelated.unique())
st.subheader("Select number visits on product related pages")
selected_prod = st.selectbox("", prod_pages)


# selecting ProductRelated Duration
prod_duration = sorted(df.ProductRelated_Duration.unique())
st.subheader("Select visit duration on product related pages")
selected_prod_dur = st.selectbox("", prod_duration)


# selecting BounceRates
bounce_rate = sorted(df.BounceRates.unique())
st.subheader("Select bounce rate of page")
selected_bounce = st.selectbox("", bounce_rate)


# selecting ExitRates
exit_rate = sorted(df.ExitRates.unique())
st.subheader("Select exit rate of page")
selected_exit = st.selectbox("", exit_rate)


# selecting PageValues
page_values = sorted(df.PageValues.unique())
st.subheader("Select page values")
selected_page_value = st.selectbox("", page_values)


# # selecting PageValues
# page_values = sorted(df.PageValues.unique())
# st.subheader("Select exit rate of page")
# selected_page_value = st.selectbox("", page_values)


# selecting SpecialDay
special_day = sorted(df.SpecialDay.unique())
st.subheader("Select special day")
selected_special_day = st.selectbox("", special_day)


# selecting Month
month = sorted(df.Month.unique())
st.subheader("Select month")
selected_month = st.selectbox("", month)


# Encode month
def encode_month(selected_item):
    
    file = open('dict/month_dict.pkl', 'rb')
    dict_loc = pkl.load(file) 
    
    return dict_loc.get(selected_item, 'No info available')

selected_month = encode_month(selected_month)


# selecting os
selected_os = sorted(df.OperatingSystems.unique())
st.subheader("Select Operating System")
selected_os = st.selectbox("", selected_os)


# selecting Browser
selected_browser = sorted(df.Browser.unique())
st.subheader("Select month")
selected_browser = st.selectbox("", selected_browser)


# selecting Region
selected_region = sorted(df.Region.unique())
st.subheader("Select region")
selected_region = st.selectbox("", selected_region)


# selecting TrafficType
selected_traffic = sorted(df.TrafficType.unique())
st.subheader("Select traffic type")
selected_traffic = st.selectbox("", selected_traffic)


# selecting VisitorType
selected_visitor = sorted(df.VisitorType.unique())
st.subheader("Select visitor type")
selected_visitor = st.selectbox("", selected_visitor)


# Encode VisitorType
def encode_visitor(selected_item):
    
    file = open('dict/visitor_dict.pkl', 'rb')
    dict_loc = pkl.load(file) 
    
    return dict_loc.get(selected_item, 'No info available')

selected_visitor = encode_visitor(selected_visitor)



# selecting Weekend
selected_weekend = sorted(df.Weekend.unique())
st.subheader("Is the day weekend?")
selected_weekend = st.selectbox("", selected_weekend)


# Encode month
def encode_weekend(selected_item):
    
    file = open('dict/weekend_dict.pkl', 'rb')
    dict_loc = pkl.load(file) 
    
    return dict_loc.get(selected_item, 'No info available')

selected_weekend = encode_weekend(selected_weekend)


# import joblib
pickle_in = open("saved_models/model.pkl","rb")
model = pkl.load(pickle_in)

data = {
    "Administrative": selected_admin, 
    "Administrative_Duration": selected_admin_dur, 
    "Informational": selected_info, 
    "Informational_Duration": selected_info_dur, 
    "ProductRelated": selected_prod, 
    "ProductRelated_Duration": selected_prod_dur, 
    "BounceRates": selected_bounce, 
    "ExitRates": selected_exit,
    "PageValues": selected_page_value, 
    "SpecialDay": selected_special_day, 
    "Month": selected_month, 
    "OperatingSystems": selected_os, 
    "Browser": selected_browser, 
    "Region": selected_region, 
    "TrafficType": selected_traffic, 
    "VisitorType": selected_visitor, 
    "Weekend": selected_weekend
}

df_1 = pd.DataFrame(data, index = [0])
prediction = model(df_1)
# prediction = model.predict([[selected_admin, selected_admin_dur, selected_info, selected_info_dur, selected_prod, selected_prod_dur, selected_bounce, selected_exit, selected_page_value, selected_special_day, selected_month, selected_os, selected_browser, selected_region, selected_traffic, selected_visitor, selected_weekend]])   # index is causing problem


# Adding Predict Button
predict_button = st.button('Predict')


if predict_button:
    if(prediction == 1):
        st.success('This customer will Shop')
    else:
        st.success('This customer segment will NOT Shop')    

st.write('\n')
about = st.expander('More about app')
about.write("https://github.com/vikramnayyar/online-shoppers-intention/blob/main/README.md")
# about = st.expander('Contact Developer')
about.write("Contact: vikramnayyar@live.com")