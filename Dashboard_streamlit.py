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

FASTAPI_URI = 'https://p7-fastapi-backend.herokuapp.com/'

df_data = pd.read_csv('df_application_test.zip', compression='zip', header=0, sep=',', quotechar='"')
df_selection = pd.read_csv('selected_feats.csv')
selection = df_selection["selected_feats"].tolist()


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
        user_input_dict = {'AMT_CREDIT': AMT_CREDIT,
                         'AMT_ANNUITY': AMT_ANNUITY,
                         'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
                         'CODE_GENDER': CODE_GENDER,
                         'DAYS_BIRTH': DAYS_BIRTH,
                         'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
                         'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
                         'ORGANIZATION_TYPE': ORGANIZATION_TYPE,
                         'DAYS_EMPLOYED': DAYS_EMPLOYED,
                         'ACTIVE_AMT_CREDIT_SUM_DEBT_MAX': ACTIVE_AMT_CREDIT_SUM_DEBT_MAX,
                         'PAYMENT_RATE': PAYMENT_RATE,
                         'ANNUITY_INCOME_PERC': ANNUITY_INCOME_PERC,
                         'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH,
                         'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
                         'FLAG_OWN_CAR': FLAG_OWN_CAR,
                         'OWN_CAR_AGE': OWN_CAR_AGE,
                         'FLAG_DOCUMENT_3': FLAG_DOCUMENT_3,
                         'CLOSED_DAYS_CREDIT_MAX': CLOSED_DAYS_CREDIT_MAX,
                         'INSTAL_AMT_PAYMENT_SUM': INSTAL_AMT_PAYMENT_SUM,
                         'APPROVED_CNT_PAYMENT_MEAN': APPROVED_CNT_PAYMENT_MEAN,
                         'PREV_CNT_PAYMENT_MEAN': PREV_CNT_PAYMENT_MEAN,
                         'PREV_APP_CREDIT_PERC_MIN': PREV_APP_CREDIT_PERC_MIN,
                         'INSTAL_DPD_MEAN': INSTAL_DPD_MEAN,
                         'INSTAL_DAYS_ENTRY_PAYMENT_MAX': INSTAL_DAYS_ENTRY_PAYMENT_MAX,
                         'POS_MONTHS_BALANCE_SIZE': POS_MONTHS_BALANCE_SIZE
                        }

        return dict(user_input_dict)

    ######################
    #main page layout
    ######################

    st.title("Loan Default Prediction")
    st.subheader("Are you sure your loan applicant is surely going to pay the loan back?💸 "
                     "This machine learning app will help you to make a prediction to help you with your decision!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("loan_approved_hero_image.jpg")

    with col2:
        st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
    the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.
    These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
    Human approval requires extensive hour effort to review each application, however, the company will always seek
    cost optimization and improve human productivity. This sometimes causes human error and bias, as it’s not practical
    to digest a large number of applicants considering all the factors involved.""")

    st.subheader("To predict default/ failure to pay back status, you need to follow the steps below:")
    st.markdown("""
    1. Enter/choose the parameters that best descibe your applicant on the left side bar;
    2. Press the "Predict" button and wait for the result.
    """)

    st.subheader("Below you could find prediction result: ")

    ######################
    #sidebar layout
    ######################

    st.sidebar.title("Loan Applicant Info")
    st.sidebar.image("ab.png", width=100)
    st.sidebar.write("Please choose parameters that describe the applicant")

    #input features
    IDs = df_data["SK_ID_CURR"].tolist()
    
    SK_ID_CURR = st.sidebar.selectbox("Select client ID: ", (IDs))
    dict_data = df_data.loc[df_data["SK_ID_CURR"] == SK_ID_CURR].to_dict(orient='records')
    AMT_CREDIT = st.sidebar.number_input("Enter the credit amount of the loan (dollars):", min_value=1.0, value=dict_data[0]['AMT_CREDIT'])
    AMT_INCOME_TOTAL = st.sidebar.number_input("Enter the annual income of the client (dollars):", min_value=1.0, value=dict_data[0]['AMT_INCOME_TOTAL'])
    
    if dict_data[0]['AMT_ANNUITY'] != None:
        AMT_ANNUITY = st.sidebar.number_input("Enter the loan annuity (dollars):", min_value=1.0, value=dict_data[0]['AMT_ANNUITY'])
    else:
        AMT_ANNUITY = st.sidebar.number_input("Enter the loan annuity (dollars):", min_value=1.0, disabled=True)
        AMT_ANNUITY = None
        
    AMT_GOODS_PRICE = st.sidebar.number_input(
        "For consumer loans, enter the price of the goods for which the loan is given:", 
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
        "Enter the birth date of the client:", 
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
        "Select the family status of the client: ", 
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
        "Select the client's education: ", 
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
        "Select the type of organization where the client works: ", 
        organization_type, 
        index=get_string_index(organization_type, dict_data[0]["ORGANIZATION_TYPE"])
    )
    
    if dict_data[0]["DAYS_EMPLOYED"] != None:
        DAYS_EMPLOYED = st.sidebar.number_input(
            "Enter how many days before the application the person started current employment (days):",
            min_value=-20000.0,
            max_value=0.0,
            value=dict_data[0]["DAYS_EMPLOYED"]
        )
    else:
        DAYS_EMPLOYED = st.sidebar.number_input(
            "Enter how many days before the application the person started current employment (days):",
            min_value=-20000.0,
            max_value=0.0,
            disabled=True
        )
        DAYS_EMPLOYED = None
        
    if dict_data[0]["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"] != None:
        ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = st.sidebar.number_input(
            "Enter the maximum current debt on Credit Bureau credit (dollars):",
            min_value=-5000000.0, 
            max_value=5000000.0,
            value=dict_data[0]["ACTIVE_AMT_CREDIT_SUM_DEBT_MAX"]
        )
    else:
        ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = st.sidebar.number_input(
            "Enter the maximum current debt on Credit Bureau credit (dollars):",
            min_value=-5000000.0, 
            max_value=5000000.0,
            disabled=True
        )
        ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = None
    
    DAYS_ID_PUBLISH = st.sidebar.number_input(
        "How many days before the application did client change the identity document with which he applied for the loan, time only relative to the application (days):",
        min_value=-10000,
        max_value=0,
        value=dict_data[0]["DAYS_ID_PUBLISH"]
    )
    
    REGION_POPULATION_RELATIVE = st.sidebar.slider(
        "Enter the normalized population of region where client lives (higher number means the client lives in more populated region): ", 
        min_value=0.0, 
        max_value=0.1, 
        step=0.001, 
        value=dict_data[0]["REGION_POPULATION_RELATIVE"]
    )
    
    if dict_data[0]["FLAG_OWN_CAR"] == True:
        FLAG_OWN_CAR = st.sidebar.radio("Does the client own a car?", ("Yes", "No"), index=0)
    else:
        FLAG_OWN_CAR = st.sidebar.radio("Does the client own a car?", ("Yes", "No"), index=1)
        
    if FLAG_OWN_CAR == "Yes":
        FLAG_OWN_CAR = True
    else:
        FLAG_OWN_CAR = False
        
    if dict_data[0]["OWN_CAR_AGE"] != None:
        OWN_CAR_AGE = st.sidebar.number_input(
            "Age of the client's car (years):",
            min_value=0.0,
            value=dict_data[0]["OWN_CAR_AGE"],
            disabled=not FLAG_OWN_CAR
        )
    else:
        OWN_CAR_AGE = st.sidebar.number_input(
            "Age of the client's car (years):",
            min_value=0.0,
            disabled=True
        )
        OWN_CAR_AGE = None
    
    if dict_data[0]["FLAG_DOCUMENT_3"] == True:
        FLAG_DOCUMENT_3 = st.sidebar.radio("Did client provide document 3? ", ('Yes', 'No'), index=0)
    else:
        FLAG_DOCUMENT_3 = st.sidebar.radio("Did client provide document 3? ", ('Yes', 'No'), index=1
                                          )
    if FLAG_DOCUMENT_3 == "Yes":
        FLAG_DOCUMENT_3 = True
    else:
        FLAG_DOCUMENT_3 = False
    
    if dict_data[0]["CLOSED_DAYS_CREDIT_MAX"] != None:
        CLOSED_DAYS_CREDIT_MAX = st.sidebar.number_input(
            "When the status of the Credit Bureau (CB) reported credits si 'closed', how many days (MAX) before current application did client apply for Credit Bureau credit? time only relative to the application (days):",
            min_value=-5000.0,
            max_value=0.0,
            value=dict_data[0]["CLOSED_DAYS_CREDIT_MAX"]
        )
    else:
        CLOSED_DAYS_CREDIT_MAX = st.sidebar.number_input(
            "When the status of the Credit Bureau (CB) reported credits si 'closed', how many days (MAX) before current application did client apply for Credit Bureau credit? time only relative to the application (days):",
            min_value=-5000.0,
            max_value=0.0,
            disabled=True
        )
        CLOSED_DAYS_CREDIT_MAX = None
        
    if dict_data[0]["INSTAL_AMT_PAYMENT_SUM"] != None:
        INSTAL_AMT_PAYMENT_SUM = st.sidebar.number_input(
            "Enter the total sum of previous loan installments (dollars):",
            min_value=0.0,
            max_value=5000000.0,
            value=dict_data[0]["INSTAL_AMT_PAYMENT_SUM"]
        )
    else:
        INSTAL_AMT_PAYMENT_SUM = st.sidebar.number_input(
            "Enter the total sum of previous loan installments (dollars):",
            min_value=0.0,
            max_value=5000000.0,
            disabled=True
        )
        INSTAL_AMT_PAYMENT_SUM = None
        
    if dict_data[0]["APPROVED_CNT_PAYMENT_MEAN"] != None:
        APPROVED_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "Enter the MEAN term of previous ACCEPTED credit applications (years):",
            min_value=0.0,
            max_value=5000000.0,
            value=dict_data[0]["APPROVED_CNT_PAYMENT_MEAN"]
        )
    else:
        APPROVED_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "Enter the MEAN term of previous ACCEPTED credit applications (years):",
            min_value=0.0,
            max_value=5000000.0,
            disabled=True,
        )
        APPROVED_CNT_PAYMENT_MEAN = None
    
    if dict_data[0]["PREV_CNT_PAYMENT_MEAN"] != None:
        PREV_CNT_PAYMENT_MEAN = st.sidebar.number_input(
            "Enter the MEAN term of ALL (accepted or refused) previous credit applications (years):",
            min_value=0.0,
            max_value=5000000.0,
            value=dict_data[0]["PREV_CNT_PAYMENT_MEAN"]
        )
    else:
        PREV_CNT_PAYMENT_MEAN = st.sidebar.number_input(
        "Enter the MEAN term of ALL (accepted or refused) previous credit applications (years):",
        min_value=0.0,
        max_value=5000000.0, 
        disabled=True
        )
        PREV_CNT_PAYMENT_MEAN = None
    
    if dict_data[0]["PREV_APP_CREDIT_PERC_MIN"] != None:
        PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider(
            "Enter minimum of the ratio between how much credit did client asked for on the previous application and how much he actually was offered (%):",
            min_value=0.0, 
            max_value=1000.0, 
            step=1.0, 
            value=dict_data[0]["PREV_APP_CREDIT_PERC_MIN"], 
        )
    else:
        PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider(
        "Enter minimum of the ratio between how much credit did client asked for on the previous application and how much he actually was offered (%):",
        min_value=0.0, 
        max_value=1000.0, 
        step=1.0,
        disabled=True
        )
        PREV_APP_CREDIT_PERC_MIN = None
        
    if dict_data[0]["INSTAL_DPD_MEAN"] != None:
        INSTAL_DPD_MEAN = st.sidebar.number_input(
            "What is the MEAN days past due of the previous credit? (days):",
            min_value=0.0, 
            max_value=10000.0, 
            value=dict_data[0]["INSTAL_DPD_MEAN"],
        )
    else:
        INSTAL_DPD_MEAN = st.sidebar.number_input(
        "What is the MEAN days past due of the previous credit? (days):",
        min_value=0.0, 
        max_value=10000.0,
        disabled=True
        )
        INSTAL_DPD_MEAN = None
    
    if dict_data[0]["INSTAL_DAYS_ENTRY_PAYMENT_MAX"] != None:
        INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.number_input("What is the maximum number of days between when the installments of previous credit was actually paid and the application date of current loan (days):",
                                                                min_value=-5000.0, 
                                                                max_value=0.0, 
                                                                value=dict_data[0]["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], 
                                                               )
    else:
        INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.number_input("What is the maximum number of days between when the installments of previous credit was actually paid and the application date of current loan (days):",
                                                                min_value=-5000.0, 
                                                                max_value=0.0, 
                                                                disabled=True
                                                               )
        INSTAL_DAYS_ENTRY_PAYMENT_MAX = None
    
    if dict_data[0]["POS_MONTHS_BALANCE_SIZE"] != None:
        POS_MONTHS_BALANCE_SIZE = st.sidebar.number_input(
            "How may monthly cash balances were observed for ALL the previous loans (months):", 
            min_value=0.0,
            value=dict_data[0]["POS_MONTHS_BALANCE_SIZE"],
        )
    else:
        POS_MONTHS_BALANCE_SIZE = st.sidebar.number_input(
        "How may monthly cash balances were observed for ALL the previous loans (months):", 
        min_value=0.0,
        disabled=True
        )
        POS_MONTHS_BALANCE_SIZE = None

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
        pred = request_prediction(FASTAPI_URI, user_input)["Probability"][0]
        st.write(
            'The credit default risk is {:.2f}'.format(pred))
        
        threshold = 0.5
        if pred > threshold:
            st.error('DECLINED! The applicant has a high risk of not paying the loan back.')
        else:
            st.success('APPROVED! The applicant has a high probability of paying the loan back.')
        
        #prepare test set for shap explainability
        st.subheader('Result Interpretability - Applicant Level')
        shap.initjs()
        with open('/shap_explainer.pickle', 'rb') as handle:
            explainer = pickle.load(handle)
        df_user_input = pd.DataFrame([user_input])

        df_user_input["NAME_EDUCATION_TYPE"] = df_user_input["NAME_EDUCATION_TYPE"].astype('category')
        df_user_input["ORGANIZATION_TYPE"] = df_user_input["ORGANIZATION_TYPE"].astype('category')
        df_user_input["NAME_FAMILY_STATUS"] = df_user_input["NAME_FAMILY_STATUS"].astype('category')
        df_user_input["OWN_CAR_AGE"] = df_user_input["OWN_CAR_AGE"].astype('float')
       
        shap_values = explainer.shap_values(df_user_input)
        fig = shap.summary_plot(
            shap_values,
            features=df_user_input.iloc[[0]], 
            feature_names=df_user_input.iloc[[0]].columns, 
            plot_type='bar')
        st.pyplot(fig)
        
        st.subheader('Model Interpretability - Overall')
        with open('/shap_values.pickle', 'rb') as handle:
            shap_values_ttl = pickle.load(handle)
            
        if pred > threshold:
            fig_ttl = shap.summary_plot(
                shap_values_ttl[1],
                features=df_data[selection], 
                feature_names=df_data[selection].columns
            )
        else:
            fig_ttl = shap.summary_plot(
                shap_values_ttl[0],
                features=df_data[selection], 
                feature_names=df_data[selection].columns
            )
        st.pyplot(fig_ttl)
        st.write(""" In this chart blue and red mean the feature value, e.g. annual income blue is a smaller value e.g. 40K USD,
        and red is a higher value e.g. 100K USD. The width of the bars represents the number of observations on a certain feature value,
        for example with the annual_inc feature we can see that most of the applicants are within the lower-income or blue area. And on axis x negative SHAP
        values represent applicants that are likely to churn and the positive values on the right side represent applicants that are likely to pay the loan back.
        What we are learning from this chart is that features such as annual_inc and sub_grade are the most impactful features driving the outcome prediction.
        The higher the salary is, or the lower the subgrade is, the more likely the applicant to pay the loan back and vice versa, which makes total sense in our case.
        """)
    
if __name__ == '__main__':
    main()