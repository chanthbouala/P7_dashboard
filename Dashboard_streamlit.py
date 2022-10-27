import streamlit as st
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pickle
import shap
import streamlit.components.v1 as components
import requests
import datetime
import json as js

def main():
    FASTAPI_URI = 'http://127.0.0.1:8000/predict'

    def request_prediction(model_uri, data):
        headers = {"Content-Type": "application/json"}
        #response = requests.request(method='POST', headers=headers, url=model_uri, json=payload)
        response = requests.post(model_uri, json=data, headers=headers)

        if response.status_code != 200:
            raise Exception(
                "Request failed with status {}, {}".format(response.status_code, response.text))

        return response.json()
    
    def preprocess(AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE, CODE_GENDER, DAYS_BIRTH, NAME_FAMILY_STATUS, NAME_EDUCATION_TYPE, ORGANIZATION_TYPE, DAYS_EMPLOYED, ACTIVE_AMT_CREDIT_SUM_DEBT_MAX, DAYS_ID_PUBLISH, REGION_POPULATION_RELATIVE, FLAG_OWN_CAR, OWN_CAR_AGE, FLAG_DOCUMENT_3, CLOSED_DAYS_CREDIT_MAX, INSTAL_AMT_PAYMENT_SUM, APPROVED_CNT_PAYMENT_MEAN, PREV_CNT_PAYMENT_MEAN, PREV_APP_CREDIT_PERC_MIN, INSTAL_DPD_MEAN, INSTAL_DAYS_ENTRY_PAYMENT_MAX, POS_MONTHS_BALANCE_SIZE):

        # Pre-processing user input
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
        ANNUITY_INCOME_PERC = AMT_ANNUITY/AMT_INCOME_TOTAL
        user_input_dict = {'AMT_CREDIT': AMT_CREDIT,
                         'AMT_ANNUITY': AMT_ANNUITY,
                         'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
                         'CODE_GENDER': CODE_GENDER,
                         'DAYS_BIRTH': -DAYS_BIRTH,
                         'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
                         'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
                         'ORGANIZATION_TYPE': ORGANIZATION_TYPE,
                         'DAYS_EMPLOYED': -DAYS_EMPLOYED,
                         'ACTIVE_AMT_CREDIT_SUM_DEBT_MAX': ACTIVE_AMT_CREDIT_SUM_DEBT_MAX,
                         'PAYMENT_RATE': PAYMENT_RATE,
                         'ANNUITY_INCOME_PERC': ANNUITY_INCOME_PERC,
                         'DAYS_ID_PUBLISH': -DAYS_ID_PUBLISH,
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


    st.set_page_config(
        page_title="Loan Prediction App",
        page_icon="loan_approved_hero_image.jpg"
    )

    st.set_option('deprecation.showPyplotGlobalUse', False)

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
    AMT_CREDIT = st.sidebar.number_input("Enter the credit amount of the loan (dollars):", min_value=1, value=100000)
    AMT_INCOME_TOTAL = st.sidebar.number_input("Enter the annual income of the client (dollars):", min_value=1, value=60000)
    AMT_ANNUITY = st.sidebar.number_input("Enter the loan annuity (dollars):", min_value=0, value=20000)
    AMT_GOODS_PRICE = st.sidebar.number_input("For consumer loans, enter the price of the goods for which the loan is given:", min_value=0, value=450000)
    CODE_GENDER = st.sidebar.radio("Select client gender: ", ('Female', 'Male'))
    if CODE_GENDER == "Female":
        CODE_GENDER = True
    else:
        CODE_GENDER = False
    DAYS_BIRTH = (datetime.date.today() - (st.sidebar.date_input("Enter the birth date of the client:", min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today()))).days
    NAME_FAMILY_STATUS = st.sidebar.selectbox("Select the family status of the client: ", (
        'Civil marriage',
        'Married',
        'Separated',
        'Single / not married',
        'Unknown',
        'Widow'
    ), index=3)
    NAME_EDUCATION_TYPE = st.sidebar.selectbox("Select the client's education: ", ('Academic degree', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Secondary / secondary special'), index=1)
    ORGANIZATION_TYPE = st.sidebar.selectbox("Select the type of organization where the client works: ", (
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
                                            ), index=2)
    DAYS_EMPLOYED = st.sidebar.number_input("Enter how many days before the application the person started current employment (days):", min_value=0, max_value=20000, value=365)
    ACTIVE_AMT_CREDIT_SUM_DEBT_MAX = st.sidebar.number_input("Enter the maximum current debt on Credit Bureau credit (dollars):", min_value=-5000000, max_value=5000000, value=200000)
    DAYS_ID_PUBLISH = st.sidebar.number_input("How many days before the application did client change the identity document with which he applied for the loan, time only relative to the application (days):", min_value=0, max_value=10000, value=3254)
    REGION_POPULATION_RELATIVE = st.sidebar.slider("Enter the normalized population of region where client lives (higher number means the client lives in more populated region): ", min_value=0.0, max_value=0.1, step=0.001, value=0.018850)
    FLAG_OWN_CAR = st.sidebar.radio("Does the client own a car?", ("Yes", "No"))
    if FLAG_OWN_CAR == "Yes":
        FLAG_OWN_CAR = True
    else:
        FLAG_OWN_CAR = False
    OWN_CAR_AGE = st.sidebar.number_input("Age of the client's car (years):", min_value=0, value=9, disabled=not FLAG_OWN_CAR)
    FLAG_DOCUMENT_3 = st.sidebar.radio("Did client provide document 3? ", ('Yes', 'No'))
    if FLAG_DOCUMENT_3 == "Yes":
        FLAG_DOCUMENT_3 = True
    else:
        FLAG_DOCUMENT_3 = False
    
    CLOSED_DAYS_CREDIT_MAX = st.sidebar.number_input("When the status of the Credit Bureau (CB) reported credits si 'closed', how many days (MAX) before current application did client apply for Credit Bureau credit? time only relative to the application (days):", min_value=0, max_value=5000, value=729)
    
    # if previous application to loan was accepted
    prev_loan = st.sidebar.radio("Have you ever contracted a loan before?", ("Yes", "No"))
    if prev_loan == "Yes":
        prev_loan = False
    else:
        prev_loan = True

    INSTAL_AMT_PAYMENT_SUM = st.sidebar.number_input("Enter the total sum of previous loan installments (dollars):", min_value=0, max_value=5000000, disabled=prev_loan, value=50000)
    APPROVED_CNT_PAYMENT_MEAN = st.sidebar.number_input("Enter the MEAN term of previous ACCEPTED credit applications (years):", min_value=0, max_value=5000000, disabled=prev_loan, value=13)
    PREV_CNT_PAYMENT_MEAN = st.sidebar.number_input("Enter the MEAN term of ALL (accepted or refused) previous credit applications (years):", min_value=0, max_value=5000000, disabled=prev_loan, value=13)
    PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider("Enter minimum of the ratio between how much credit did client asked for on the previous application and how much he actually was offered (%):", min_value=0, max_value=1000, step=1, value=90, disabled=prev_loan)
    INSTAL_DPD_MEAN = st.sidebar.number_input("What is the MEAN days past due of the previous credit? (days):",min_value=0, max_value=10000, value=0, disabled=prev_loan)
    INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.number_input("What is the maximum number of days between when the installments of previous credit was actually paid and the application date of current loan (days):", min_value=0, max_value=5000, value=88, disabled=prev_loan)
    POS_MONTHS_BALANCE_SIZE = st.sidebar.number_input("How may monthly cash balances were observed for ALL the previous loans (months):", min_value=0, value=22, disabled=prev_loan)

    #predict button
    btn_predict = st.sidebar.button("Predict")

    if btn_predict:
        user_input = preprocess(AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE, CODE_GENDER, DAYS_BIRTH, NAME_FAMILY_STATUS, NAME_EDUCATION_TYPE, ORGANIZATION_TYPE, DAYS_EMPLOYED, ACTIVE_AMT_CREDIT_SUM_DEBT_MAX, DAYS_ID_PUBLISH, REGION_POPULATION_RELATIVE, FLAG_OWN_CAR, OWN_CAR_AGE, FLAG_DOCUMENT_3, CLOSED_DAYS_CREDIT_MAX, INSTAL_AMT_PAYMENT_SUM, APPROVED_CNT_PAYMENT_MEAN, PREV_CNT_PAYMENT_MEAN, PREV_APP_CREDIT_PERC_MIN, INSTAL_DPD_MEAN, INSTAL_DAYS_ENTRY_PAYMENT_MAX, POS_MONTHS_BALANCE_SIZE)
        
        pred = None
        pred = request_prediction(FASTAPI_URI, user_input)["Probability"][0]
        st.write(
            'La prédiction est de {:.2f}'.format(pred))
        
        if pred > 0.5:
            st.error('Warning! The applicant has a high risk of not paying the loan back!')
        else:
            st.success('It is green! The applicant has a high probability of paying the loan back!')
        '''
        #prepare test set for shap explainability
        loans = st.cache(pd.read_csv)("mycsvfile.csv")
        X = loans.drop(columns=['loan_status','home_ownership__ANY','home_ownership__MORTGAGE','home_ownership__NONE','home_ownership__OTHER','home_ownership__OWN',
                       'home_ownership__RENT','addr_state__AK','addr_state__AL','addr_state__AR','addr_state__AZ','addr_state__CA','addr_state__CO','addr_state__CT',
                       'addr_state__DC','addr_state__DE','addr_state__FL','addr_state__GA','addr_state__HI','addr_state__ID','addr_state__IL','addr_state__IN',
                       'addr_state__KS','addr_state__KY','addr_state__LA','addr_state__MA','addr_state__MD','addr_state__ME','addr_state__MI','addr_state__MN',
                       'addr_state__MO','addr_state__MS','addr_state__MT','addr_state__NC','addr_state__ND','addr_state__NE','addr_state__NH','addr_state__NJ',
                       'addr_state__NM','addr_state__NV','addr_state__NY','addr_state__OH','addr_state__OK','addr_state__OR','addr_state__PA','addr_state__RI',
                       'addr_state__SC','addr_state__SD','addr_state__TN','addr_state__TX','addr_state__UT','addr_state__VA','addr_state__VT', 'addr_state__WA',
                       'addr_state__WI','addr_state__WV','addr_state__WY'])
        y = loans[['loan_status']]
        y_ravel = y.values.ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.25, random_state=42, stratify=y)

        st.subheader('Result Interpretability - Applicant Level')
        shap.initjs()
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(user_input)
        fig = shap.plots.bar(shap_values[0])
        st.pyplot(fig)

        st.subheader('Model Interpretability - Overall')
        shap_values_ttl = explainer(X_test)
        fig_ttl = shap.plots.beeswarm(shap_values_ttl)
        st.pyplot(fig_ttl)
        st.write(""" In this chart blue and red mean the feature value, e.g. annual income blue is a smaller value e.g. 40K USD,
        and red is a higher value e.g. 100K USD. The width of the bars represents the number of observations on a certain feature value,
        for example with the annual_inc feature we can see that most of the applicants are within the lower-income or blue area. And on axis x negative SHAP
        values represent applicants that are likely to churn and the positive values on the right side represent applicants that are likely to pay the loan back.
        What we are learning from this chart is that features such as annual_inc and sub_grade are the most impactful features driving the outcome prediction.
        The higher the salary is, or the lower the subgrade is, the more likely the applicant to pay the loan back and vice versa, which makes total sense in our case.
        """)
        '''
    
if __name__ == '__main__':
    main()