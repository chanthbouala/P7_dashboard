import streamlit as st
import pandas as pd
import base64
import numpy as np
import pickle5 as pickle
import shap
import requests
import datetime
import json as js
import math

#FASTAPI_URI = 'https://p7-fastapi-backend.herokuapp.com/'
FASTAPI_URI = 'http://127.0.0.1:8000/'

df_data = pd.read_csv('df_application_test.zip', compression='zip', header=0, sep=',', quotechar='"')
df_selection = pd.read_csv('selected_feats.csv')
selection = df_selection["selected_feats"].tolist()

with open('shap_explainer_2.pickle', 'rb') as handle:
    explainer = pickle.load(handle)
with open('shap_values_2.pickle', 'rb') as handle:
    shap_values_ttl = pickle.load(handle)

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="loan_approved_hero_image.jpg"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    def replace_none_in_dict(items):
        replacement = None
        return {k: v if ((type(v) is not str and not np.isnan(v)) or (type(v) is str and v == v)) else replacement for k, v in items}

    def request_prediction(model_uri, data):
        headers = {"Content-Type": "application/json"}
        #response = requests.request(method='POST', headers=headers, url=model_uri, json=payload)
        json_str = js.dumps(data)
        payload = js.loads(json_str, object_pairs_hook=replace_none_in_dict)
        response = requests.post(model_uri + 'predict', json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(
                "Request failed with status {}, {}".format(response.status_code, response.text))

        return response.json()

    def preprocess(AMT_CREDIT, 
                   AMT_INCOME_TOTAL, 
                   AMT_ANNUITY, 
                   AMT_GOODS_PRICE,
                   CODE_GENDER, 
                   DAYS_BIRTH, 
                   NAME_FAMILY_STATUS, 
                   NAME_EDUCATION_TYPE, 
                   ORGANIZATION_TYPE, 
                   DAYS_EMPLOYED, 
                   ACTIVE_AMT_CREDIT_SUM_DEBT_MAX, 
                   DAYS_ID_PUBLISH, 
                   REGION_POPULATION_RELATIVE, 
                   FLAG_OWN_CAR, OWN_CAR_AGE, 
                   FLAG_DOCUMENT_3, 
                   CLOSED_DAYS_CREDIT_MAX, 
                   INSTAL_AMT_PAYMENT_SUM, 
                   APPROVED_CNT_PAYMENT_MEAN, 
                   PREV_CNT_PAYMENT_MEAN, 
                   PREV_APP_CREDIT_PERC_MIN, 
                   INSTAL_DPD_MEAN, 
                   INSTAL_DAYS_ENTRY_PAYMENT_MAX, 
                   POS_MONTHS_BALANCE_SIZE
                  ):

        # Pre-processing user input
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
        ANNUITY_INCOME_PERC = AMT_ANNUITY/AMT_INCOME_TOTAL
        user_input_dict = {
            "ACTIVE_AMT_CREDIT_SUM_DEBT_MAX": ACTIVE_AMT_CREDIT_SUM_DEBT_MAX,
            "AMT_ANNUITY": AMT_ANNUITY,
            "AMT_CREDIT": AMT_CREDIT,
            "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
            "ANNUITY_INCOME_PERC": ANNUITY_INCOME_PERC,
            "APPROVED_CNT_PAYMENT_MEAN": APPROVED_CNT_PAYMENT_MEAN, 
            "CLOSED_DAYS_CREDIT_MAX": CLOSED_DAYS_CREDIT_MAX,
            "CODE_GENDER": CODE_GENDER,
            "DAYS_BIRTH": DAYS_BIRTH,
            "DAYS_EMPLOYED": DAYS_EMPLOYED,
            "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
            "FLAG_DOCUMENT_3": FLAG_DOCUMENT_3,
            "FLAG_OWN_CAR": FLAG_OWN_CAR,
            "INSTAL_AMT_PAYMENT_SUM": INSTAL_AMT_PAYMENT_SUM,
            "INSTAL_DAYS_ENTRY_PAYMENT_MAX": INSTAL_DAYS_ENTRY_PAYMENT_MAX,
            "INSTAL_DPD_MEAN": INSTAL_DPD_MEAN,
            "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
            "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
            "ORGANIZATION_TYPE": ORGANIZATION_TYPE,
            "OWN_CAR_AGE": OWN_CAR_AGE,
            "PAYMENT_RATE": PAYMENT_RATE,
            "POS_MONTHS_BALANCE_SIZE": POS_MONTHS_BALANCE_SIZE,
            "PREV_APP_CREDIT_PERC_MIN": PREV_APP_CREDIT_PERC_MIN,
            "PREV_CNT_PAYMENT_MEAN": PREV_CNT_PAYMENT_MEAN,
            "REGION_POPULATION_RELATIVE": REGION_POPULATION_RELATIVE,
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL
        }

        return dict(user_input_dict)

    ######################
    #sidebar layout
    ######################

    st.sidebar.title("Loan Applicant Info")
    st.sidebar.image("ab.png", width=100)
    st.sidebar.write("Please choose ID of the applicant, and if you want, adjust the parameters that describe the applicant")

    #input features
    IDs = df_data["SK_ID_CURR"].tolist()
    
    SK_ID_CURR = st.sidebar.selectbox("Select client ID: ", (IDs))
    dict_data = df_data.loc[df_data["SK_ID_CURR"] == SK_ID_CURR].to_dict(orient='records')
    AMT_CREDIT = st.sidebar.number_input("(AMT_CREDIT) Enter the credit amount of the loan (dollars):", min_value=1.0, value=dict_data[0]['AMT_CREDIT'])
    AMT_INCOME_TOTAL = st.sidebar.number_input("(AMT_INCOME_TOTAL) Enter the annual income of the client (dollars):", min_value=1.0, value=dict_data[0]['AMT_INCOME_TOTAL'])
    
    if not np.isnan(dict_data[0]['AMT_ANNUITY']):
        AMT_ANNUITY = st.sidebar.number_input("(AMT_ANNUITY) Enter the loan annuity (dollars):", min_value=1.0, value=dict_data[0]['AMT_ANNUITY'])
    else:
        AMT_ANNUITY_disable = not st.sidebar.checkbox("(AMT_ANNUITY) Missing value, tick the box to modify input:")
        if AMT_ANNUITY_disable:
            AMT_ANNUITY = np.nan
        else:
            AMT_ANNUITY = st.sidebar.number_input("Enter the loan annuity (dollars):", 
                                                  min_value=1.0, 
                                                  disabled=AMT_ANNUITY_disable, 
                                                  value=df_data["AMT_ANNUITY"].median()
                                                 )
        
        
    AMT_GOODS_PRICE = st.sidebar.number_input(
        "(AMT_GOODS_PRICE) For consumer loans, enter the price of the goods for which the loan is given:", 
        min_value=1.0, 
        value=dict_data[0]['AMT_GOODS_PRICE']
    )
    
    if dict_data[0]["CODE_GENDER"] == True:
        CODE_GENDER = st.sidebar.radio("Select client gender: ", ('Female', 'Male'), index=0)
    else:
        CODE_GENDER = st.sidebar.radio("Select client gender: ", ('Female', 'Male'), index=1)
    
    if CODE_GENDER == "Female":
        CODE_GENDER = True
    else:
        CODE_GENDER = False

    date_of_birth = datetime.date.today() + datetime.timedelta(days=dict_data[0]["DAYS_BIRTH"])
    DAYS_BIRTH = -(datetime.date.today() - (st.sidebar.date_input(
        "(DAYS_BIRTH) Enter the birth date of the client:", 
        min_value=datetime.date(1900, 1, 1), 
        max_value=datetime.date.today(), 
        value=date_of_birth))
                  ).days
    
    def get_string_index(strings, substr):
        if substr != None:
            for idx, string in enumerate(strings):
                if substr in string:
                    break
            return idx
        else:
            return substr
    
    family_status = [
        'Civil marriage',
        'Married',
        'Separated',
        'Single / not married',
        'Unknown',
        'Widow'
    ]
    NAME_FAMILY_STATUS = st.sidebar.selectbox(
        "(NAME_FAMILY_STATUS) Select the family status of the client: ", 
        family_status, 
        index=get_string_index(family_status, dict_data[0]["NAME_FAMILY_STATUS"])
    )
    
    education_type = [
        'Academic degree', 
        'Higher education', 
        'Incomplete higher', 
        'Lower secondary', 
        'Secondary / secondary special'
    ]
    
    NAME_EDUCATION_TYPE = st.sidebar.selectbox(
        "(NAME_EDUCATION_TYPE) Select the client's education: ", 
        education_type, 
        index=get_string_index(education_type, dict_data[0]["NAME_EDUCATION_TYPE"])
    )
    
    organization_type = [
         'Advertising',
         'Agriculture',
         'Bank',
         'Business Entity Type 1',
         'Business Entity Type 2',
         'Business Entity Type 3',
         'Cleaning',
         'Construction',
         'Culture',
         'Electricity',
         'Emergency',
         'Government',
         'Hotel',
         'Housing',
         'Industry: type 1',
         'Industry: type 2',
         'Industry: type 3',
         'Industry: type 4',
         'Industry: type 5',
         'Industry: type 6',
         'Industry: type 7',
         'Industry: type 8',
         'Industry: type 9',
         'Industry: type 10',
         'Industry: type 11',
         'Industry: type 12',
         'Industry: type 13',
         'Insurance',
         'Kindergarten',
         'Legal Services',
         'Medicine',
         'Military',
         'Mobile',
         'Other',
         'Police',
         'Postal',
         'Realtor',
         'Religion',
         'Restaurant',
         'School',
         'Security',
         'Security Ministries',
         'Self-employed',
         'Services',
         'Telecom',
         'Trade: type 1',
         'Trade: type 2',
         'Trade: type 3',
         'Trade: type 4',
         'Trade: type 5',
         'Trade: type 6',
         'Trade: type 7',
         'Transport: type 1',
         'Transport: type 2',
         'Transport: type 3',
         'Transport: type 4',
         'University',
         'XNA'
        ]
        
    ORGANIZATION_TYPE = st.sidebar.selectbox(
        "(ORGANIZATION_TYPE) Select the type of organization where the client works: ", 
        organization_type, 
        index=get_string_index(organization_type, dict_data[0]["ORGANIZATION_TYPE"])
    )
    
    if not np.isnan(dict_data[0]["DAYS_EMPLOYED"]):
        DAYS_EMPLOYED = st.sidebar.number_input(
            "(DAYS_EMPLOYED) Enter how many days before the application the person started current employment (days):",
            min_value=-20000.0,
            max_value=0.0,
            value=dict_data[0]["DAYS_EMPLOYED"]
        )
    else:
        DAYS_EMPLOYED_disable = not st.sidebar.checkbox("(DAYS_EMPLOYED) Missing value, tick the box to modify input:")
        if DAYS_EMPLOYED_disable:
            DAYS_EMPLOYED = np.nan
        else:
            DAYS_EMPLOYED = st.sidebar.number_input(
                "(DAYS_EMPLOYED) Enter how many days before the application the person started current employment (days):",
                min_value=-20000.0,
                max_value=0.0,
                disabled=DAYS_EMPLOYED_disable,
                value=df_data["DAYS_EMPLOYED"].median()
            )
        
    if not np.isnan(dict_data[0]["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]):
        ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = st.sidebar.number_input(
            "(ACTIVE_AMT_CREDIT_SUM_DEBT_MAX) Enter the maximum current debt on Credit Bureau credit (dollars):",

            value=dict_data[0]["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]
        )
    else:
        ACTIVE_AMT_CREDIT_SUM_DEBT_MAX_disable = not st.sidebar.checkbox("(ACTIVE_AMT_CREDIT_SUM_DEBT_MAX) Missing value, tick the box to modify input:")
        if ACTIVE_AMT_CREDIT_SUM_DEBT_MAX_disable:
            ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = np.nan
        else:
            ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = st.sidebar.number_input(
                "(ACTIVE_AMT_CREDIT_SUM_DEBT_MAX) Enter the maximum current debt on Credit Bureau credit (dollars):",
                disabled=ACTIVE_AMT_CREDIT_SUM_DEBT_MAX_disable,
                value=df_data["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"].median()
            )
        
    
    DAYS_ID_PUBLISH = st.sidebar.number_input(
        "(DAYS_ID_PUBLISH) How many days before the application did client change the identity document with which he applied for the loan, time only relative to the application (days):",
        min_value=-10000,
        max_value=0,
        value=dict_data[0]["DAYS_ID_PUBLISH"]
    )
    
    REGION_POPULATION_RELATIVE = st.sidebar.slider(
        "(REGION_POPULATION_RELATIVE) Enter the normalized population of region where client lives (higher number means the client lives in more populated region): ", 
        min_value=0.0, 
        max_value=0.1, 
        step=0.001,
        value=dict_data[0]["REGION_POPULATION_RELATIVE"]
    )
    
    if dict_data[0]["FLAG_OWN_CAR"] == True:
        FLAG_OWN_CAR = st.sidebar.radio("(FLAG_OWN_CAR) Does the client own a car?", ("Yes", "No"), index=0)
    else:
        FLAG_OWN_CAR = st.sidebar.radio("(FLAG_OWN_CAR) Does the client own a car?", ("Yes", "No"), index=1)
        
    if FLAG_OWN_CAR == "Yes":
        FLAG_OWN_CAR = True
    else:
        FLAG_OWN_CAR = False
    
    if FLAG_OWN_CAR and (not np.isnan(dict_data[0]["OWN_CAR_AGE"])):
        OWN_CAR_AGE = st.sidebar.number_input(
            "(OWN_CAR_AGE) Age of the client's car (years):",
            min_value=0.0,
            value=dict_data[0]["OWN_CAR_AGE"],
            disabled=not FLAG_OWN_CAR
        )
    elif (FLAG_OWN_CAR) and (np.isnan(dict_data[0]["OWN_CAR_AGE"])):
        OWN_CAR_AGE_disable = not st.sidebar.checkbox("(OWN_CAR_AGE) Missing value, tick the box to modify input:")
        if FLAG_OWN_CAR and (OWN_CAR_AGE_disable):
            OWN_CAR_AGE = np.nan
        elif FLAG_OWN_CAR and (not OWN_CAR_AGE_disable):
            OWN_CAR_AGE = st.sidebar.number_input(
                "(OWN_CAR_AGE) Age of the client's car (years):",
                min_value=0.0,
                disabled=OWN_CAR_AGE_disable,
                value=df_data["OWN_CAR_AGE"].median()
            )
    elif (not FLAG_OWN_CAR) and (np.isnan(dict_data[0]["OWN_CAR_AGE"])):
        OWN_CAR_AGE = np.nan
        
    if dict_data[0]["FLAG_DOCUMENT_3"] == True:
        FLAG_DOCUMENT_3 = st.sidebar.radio("Did client provide document 3? ", ('Yes', 'No'), index=0)
    else:
        FLAG_DOCUMENT_3 = st.sidebar.radio("Did client provide document 3? ", ('Yes', 'No'), index=1
                                          )
    if FLAG_DOCUMENT_3 == "Yes":
        FLAG_DOCUMENT_3 = True
    else:
        FLAG_DOCUMENT_3 = False
    
    if not np.isnan(dict_data[0]["CLOSED_DAYS_CREDIT_MAX"]):
        CLOSED_DAYS_CREDIT_MAX = st.sidebar.number_input(
            "(CLOSED_DAYS_CREDIT_MAX) When the status of the Credit Bureau (CB) reported credits si 'closed', how many days (MAX) before current application did client apply for Credit Bureau credit? time only relative to the application (days):",
            min_value=-5000.0,
            max_value=0.0,
            value=dict_data[0]["CLOSED_DAYS_CREDIT_MAX"]
        )
    else:
        CLOSED_DAYS_CREDIT_MAX_disable = not st.sidebar.checkbox("(CLOSED_DAYS_CREDIT_MAX) Missing value, tick the box to modify input:")
        if CLOSED_DAYS_CREDIT_MAX_disable:
            CLOSED_DAYS_CREDIT_MAX = np.nan
        else:
            CLOSED_DAYS_CREDIT_MAX = st.sidebar.number_input(
                "(CLOSED_DAYS_CREDIT_MAX) When the status of the Credit Bureau (CB) reported credits si 'closed', how many days (MAX) before current application did client apply for Credit Bureau credit? time only relative to the application (days):",
                max_value=0.0,
                disabled=CLOSED_DAYS_CREDIT_MAX_disable,
                value=df_data["CLOSED_DAYS_CREDIT_MAX"].median()
            )
        
        
    if not np.isnan(dict_data[0]["INSTAL_AMT_PAYMENT_SUM"]):
        INSTAL_AMT_PAYMENT_SUM = st.sidebar.number_input(
            "(INSTAL_AMT_PAYMENT_SUM) Enter the total sum of previous loan installments (dollars):",
            min_value=0.0,
            value=dict_data[0]["INSTAL_AMT_PAYMENT_SUM"]
        )
    else:
        INSTAL_AMT_PAYMENT_SUM_disable = not st.sidebar.checkbox("(INSTAL_AMT_PAYMENT_SUM) Missing value, tick the box to modify input:")
        if INSTAL_AMT_PAYMENT_SUM_disable:
            INSTAL_AMT_PAYMENT_SUM = np.nan
        else:
            INSTAL_AMT_PAYMENT_SUM = st.sidebar.number_input(
                "(INSTAL_AMT_PAYMENT_SUM) Enter the total sum of previous loan installments (dollars):",
                min_value=0.0,
                disabled=INSTAL_AMT_PAYMENT_SUM_disable,
                value=df_data["INSTAL_AMT_PAYMENT_SUM"].median()
            )
        
        
    if not np.isnan(dict_data[0]["APPROVED_CNT_PAYMENT_MEAN"]):
        APPROVED_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "(APPROVED_CNT_PAYMENT_MEAN) Enter the MEAN term of previous ACCEPTED credit applications (years):",
            min_value=0.0,
            value=dict_data[0]["APPROVED_CNT_PAYMENT_MEAN"]
        )
    else:
        APPROVED_CNT_PAYMENT_MEAN_disable = not st.sidebar.checkbox("(APPROVED_CNT_PAYMENT_MEAN) Missing value, tick the box to modify input:")
        if APPROVED_CNT_PAYMENT_MEAN_disable:
            APPROVED_CNT_PAYMENT_MEAN = np.nan
        else:
            APPROVED_CNT_PAYMENT_MEAN = st.sidebar.number_input(
                "(APPROVED_CNT_PAYMENT_MEAN) Enter the MEAN term of previous ACCEPTED credit applications (years):",
                min_value=0.0,
                disabled=APPROVED_CNT_PAYMENT_MEAN_disable,
                value=df_data["APPROVED_CNT_PAYMENT_MEAN"].median()
            )
    
    if not np.isnan(dict_data[0]["PREV_CNT_PAYMENT_MEAN"]):
        PREV_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "(PREV_CNT_PAYMENT_MEAN) Enter the MEAN term of ALL (accepted or refused) previous credit applications (years):",
            min_value=0.0,
            value=dict_data[0]["PREV_CNT_PAYMENT_MEAN"]
        )
    else:
        PREV_CNT_PAYMENT_MEAN_disable = not st.sidebar.checkbox("(PREV_CNT_PAYMENT_MEAN) Missing value, tick the box to modify input:")
        if PREV_CNT_PAYMENT_MEAN_disable:
            PREV_CNT_PAYMENT_MEAN = np.nan
        else:
            PREV_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "(PREV_CNT_PAYMENT_MEAN) Enter the MEAN term of ALL (accepted or refused) previous credit applications (years):",
            min_value=0.0,
            disabled=PREV_CNT_PAYMENT_MEAN_disable,
            value=df_data["PREV_CNT_PAYMENT_MEAN"].median()
            )
    
    if not np.isnan(dict_data[0]["PREV_APP_CREDIT_PERC_MIN"]):
        PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider(
            "(PREV_APP_CREDIT_PERC_MIN) Enter minimum of the ratio between how much credit did client asked for on the previous application and how much he actually was offered (%):",
            min_value=0.0, 
            max_value=200.0, 
            step=0.1, 
            value=dict_data[0]["PREV_APP_CREDIT_PERC_MIN"], 
        )
    else:
        PREV_APP_CREDIT_PERC_MIN_disable = not st.sidebar.checkbox("(PREV_APP_CREDIT_PERC_MIN) Missing value, tick the box to modify input:")
        if PREV_APP_CREDIT_PERC_MIN_disable:
            PREV_APP_CREDIT_PERC_MIN = np.nan
        else:
            PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider(
            "(PREV_APP_CREDIT_PERC_MIN) Enter minimum of the ratio between how much credit did client asked for on the previous application and how much he actually was offered (%):",
            min_value=0.0, 
            max_value=200.0, 
            step=0.1,
            disabled=PREV_APP_CREDIT_PERC_MIN_disable,
            value=df_data["PREV_APP_CREDIT_PERC_MIN"].median()
            )
        
    if not np.isnan(dict_data[0]["INSTAL_DPD_MEAN"]):
        INSTAL_DPD_MEAN = st.sidebar.number_input(
            "(INSTAL_DPD_MEAN) What is the MEAN days past due of the previous credit? (days):",
            min_value=0.0, 
            value=dict_data[0]["INSTAL_DPD_MEAN"],
        )
    else:
        INSTAL_DPD_MEAN_disable = not st.sidebar.checkbox("(INSTAL_DPD_MEAN) Missing value, tick the box to modify input:")
        if INSTAL_DPD_MEAN_disable:
            INSTAL_DPD_MEAN = np.nan
        else:
            INSTAL_DPD_MEAN = st.sidebar.number_input(
            "(INSTAL_DPD_MEAN) What is the MEAN days past due of the previous credit? (days):",
            min_value=0.0, 
            disabled=INSTAL_DPD_MEAN_disable,
            value=df_data["INSTAL_DPD_MEAN"].median()
            )
    
    if not np.isnan(dict_data[0]["INSTAL_DAYS_ENTRY_PAYMENT_MAX"]):
        INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.number_input("(INSTAL_DAYS_ENTRY_PAYMENT_MAX) What is the maximum number of days between when the installments of previous credit was actually paid and the application date of current loan (days):",
                                                                max_value=0.0, 
                                                                value=dict_data[0]["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], 
                                                               )
    else:
        INSTAL_DAYS_ENTRY_PAYMENT_MAX_disable = not st.sidebar.checkbox("(INSTAL_DAYS_ENTRY_PAYMENT_MAX) Missing value, tick the box to modify input:")
        if INSTAL_DAYS_ENTRY_PAYMENT_MAX_disable:
            INSTAL_DAYS_ENTRY_PAYMENT_MAX = np.nan
        else:
            INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.number_input("(INSTAL_DAYS_ENTRY_PAYMENT_MAX) What is the maximum number of days between when the installments of previous credit was actually paid and the application date of current loan (days):",
                                                                    max_value=0.0, 
                                                                    disabled=INSTAL_DAYS_ENTRY_PAYMENT_MAX_disable,
                                                                    value=df_data["INSTAL_DAYS_ENTRY_PAYMENT_MAX"].median()
                                                                   )
    
    if not np.isnan(dict_data[0]["POS_MONTHS_BALANCE_SIZE"]):
        POS_MONTHS_BALANCE_SIZE = st.sidebar.number_input(
            "(POS_MONTHS_BALANCE_SIZE) How may monthly cash balances were observed for ALL the previous loans (months):", 
            min_value=0.0,
            value=dict_data[0]["POS_MONTHS_BALANCE_SIZE"],
        )
    else:
        POS_MONTHS_BALANCE_SIZE_disable = not st.sidebar.checkbox("(POS_MONTHS_BALANCE_SIZE) Missing value, tick the box to modify input:")
        if POS_MONTHS_BALANCE_SIZE_disable:
            POS_MONTHS_BALANCE_SIZE = np.nan
        else:
            POS_MONTHS_BALANCE_SIZE = st.sidebar.number_input(
            "(POS_MONTHS_BALANCE_SIZE) How may monthly cash balances were observed for ALL the previous loans (months):", 
            min_value=0.0,
            disabled=POS_MONTHS_BALANCE_SIZE_disable,
            value=df_data["POS_MONTHS_BALANCE_SIZE"].median()
            )
    
    ######################
    #main page layout
    ######################

    st.title("Loan Default Prediction")
    st.subheader("Are you sure your loan applicant is going to pay the loan back?💸 "
                     "This machine learning app will help you to make a prediction to help you with your decision!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("loan_approved_hero_image.jpg")

    with col2:
        st.write("""When a client apply to borrow money, credit analysis is performed. The necessary investigations to measure 
    the probability that the applicant reimburse or not the loan on time are quite time-consuming. It gets even more complicated as the number of applications that are reviewed by loan officers increases.
    Human approval requires extensive hours to review each application. This sometimes causes human error and bias for it is not easy 
    to digest the heavy workload.""")
    st.write("Please find below the data about the client:")
    feat_filter = st.checkbox("Tick the box to filter out the unimportant data:")
    if feat_filter:
        st.write(df_data.loc[df_data["SK_ID_CURR"] == SK_ID_CURR][selection])
    else:
        st.write(df_data.loc[df_data["SK_ID_CURR"] == SK_ID_CURR].drop("Unnamed: 0", axis=1))
    st.subheader("In order to calculate failure probability, you need to execute the following steps:")
    st.markdown("""
    1. Select the client ID on the left side bar.
    2. Press the "Predict" button (all the way down of the left side bar) and wait for the result.
    """)

    st.subheader("Below is your prediction result:")
    
    #predict button
    btn_predict = st.sidebar.button("Predict")

    if btn_predict:
        user_input = preprocess(
            AMT_CREDIT, 
            AMT_INCOME_TOTAL,
            AMT_ANNUITY,
            AMT_GOODS_PRICE,
            CODE_GENDER,
            DAYS_BIRTH, 
            NAME_FAMILY_STATUS,
            NAME_EDUCATION_TYPE,
            ORGANIZATION_TYPE,
            DAYS_EMPLOYED,
            ACTIVE_AMT_CREDIT_SUM_DEBT_MAX,
            DAYS_ID_PUBLISH, 
            REGION_POPULATION_RELATIVE,
            FLAG_OWN_CAR,
            OWN_CAR_AGE, 
            FLAG_DOCUMENT_3, 
            CLOSED_DAYS_CREDIT_MAX,
            INSTAL_AMT_PAYMENT_SUM,
            APPROVED_CNT_PAYMENT_MEAN,
            PREV_CNT_PAYMENT_MEAN,
            PREV_APP_CREDIT_PERC_MIN,
            INSTAL_DPD_MEAN,
            INSTAL_DAYS_ENTRY_PAYMENT_MAX,
            POS_MONTHS_BALANCE_SIZE
        )

        pred = None
        risk_score = None
        pred = request_prediction(FASTAPI_URI, user_input)["Probability"][0]
        
        threshold = 0.51
        if pred > threshold:
            st.error('DECLINED! The applicant has a high risk of not paying the loan back.')
            if pred > threshold and pred < 0.6:
                risk_score = "A"
            elif pred >= 0.6 and pred < 0.7:
                risk_score = "B"
            elif pred >= 0.7 and pred < 0.8:
                risk_score = "C"
            elif pred >= 0.8 and pred < 0.9:
                risk_score = "D"
            elif pred >= 0.9 and pred <= 1:
                risk_score = "E"
            st.metric('The credit default risk is:', risk_score)
            st.write("A: The applicant is close to getting the loan")
            st.write("B: The applicant can get the loan with minor adjustements in his application")
            st.write("C: The applicant can get the loan with major adjustements in his application")
            st.write("D: The applicant is far from getting the loan")
            st.write("E: The applicant will almost never get the loan")

        else:
            st.success('APPROVED! The applicant has a high probability of paying the loan back.')
        
        #prepare test set for shap explainability
        st.subheader('Result Interpretability - Applicant Level')
        shap.initjs()
            
        df_user_input = pd.DataFrame([user_input])

        df_user_input["NAME_EDUCATION_TYPE"] = df_user_input["NAME_EDUCATION_TYPE"].astype('category')
        df_user_input["ORGANIZATION_TYPE"] = df_user_input["ORGANIZATION_TYPE"].astype('category')
        df_user_input["NAME_FAMILY_STATUS"] = df_user_input["NAME_FAMILY_STATUS"].astype('category')
        df_user_input["OWN_CAR_AGE"] = df_user_input["OWN_CAR_AGE"].astype('float')
       
        #shap_values = explainer.shap_values(df_user_input)
        #fig = shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names=df_user_input.columns, matplotlib=True, text_rotation=90)
        #fig = shap.plots.waterfall(exp[df_data.reset_index().loc[df_data.reset_index()["SK_ID_CURR"] == SK_ID_CURR].index[0]])
        fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                                     shap_values_ttl[1][df_data.reset_index().loc[
                                                         df_data.reset_index()["SK_ID_CURR"] == SK_ID_CURR].index[0]],
                                                     feature_names=selection, 
                                                     max_display=40)
        st.write("""
        In this chart blue and red mean the contribution of each variable to the outcome of the simulation, i.e. blue increases the chance of getting the loan whereas red decreases the chance of getting the loan.
        """)                                             
        st.pyplot(fig)
        
        st.subheader('Model Interpretability - Overall')
        st.write(""" 
        In this chart blue and red mean the feature value, i.e. annual income blue is a smaller value e.g. 40K USD, and red is a higher value e.g. 100K USD. The x-axis represent the contribution of the variable to the outcome of the simulation, i.e. the negative contributions (left part of the graph) increase the chance of getting the loan while the positive contributions (right part of the graph) decrease the chance of getting the loan. The width of the bars represents the number of observations on a certain feature value, for example with the INSTAL_DPD_MEAN feature we can see that most of the applicants are within the lower or blue area. The features are ordered according to their importance to the outcome of the simulation, i.e. PAYMENT_RATE is the most important feature globally and PREV_CNT_PAYMENT_MEAN is the least important globally. What we learn from this chart is that features such as CODE_GENDER or DAYS_EMPLOYED are the most likely to strongly drive the outcome prediction.
        """)
        
        fig_ttl = shap.summary_plot(
            shap_values_ttl[1],
            features=df_data[selection], 
            feature_names=df_data[selection].columns
        )
        st.pyplot(fig_ttl)

    
if __name__ == '__main__':
    main()