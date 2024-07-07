import streamlit as st
import pandas as pd
import os
from streamlit_option_menu import option_menu
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')

# groq_api_key=st.secrets["groq_api_key"]
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")

st.set_page_config(page_title="LISA : LLM Informed Statistical Analysis",
                    page_icon=":books:",layout = "wide")

st.header("Welcome to LISA: LLM Informed Statistical Analysis", divider='rainbow')
st.markdown("LISA is an innovative platform designed to automate your data analysis process using advanced Large Language Models (LLM) for insightful inferences. Whether you're a data enthusiast, researcher, or business analyst, LISA simplifies complex data tasks, providing clear and comprehensible explanations for your data.")
st.markdown("LISA combines the efficiency of automated data processing with the intelligence of modern language models to deliver a seamless and insightful data analysis experience. Empower your data with LISA!")
st.divider()


# # Display the image using st.image
# st.image('D:/Modular Coding/modular_coding/Designer (2).png', width=700, caption='LISA: LLM Informed Statistical Analysis', use_column_width=True)


with st.sidebar:
    uploaded_file=st.file_uploader("Upload a CSV file",type=['csv'])
    
    
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df.head())
    st.divider()

    option = st.selectbox("Select an option:", ["Show dataset dimensions","Display data description","Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data"])
    
    if option == "Show dataset dimensions":
        shape_of_the_data = df.shape
        systemmessageprompt = SystemMessagePromptTemplate.from_template( "You are Bot who is expert knowing all the functions in pandas library " "Explain about only the output obtained")
        humanmessageprompt = HumanMessagePromptTemplate.from_template(
            'The shape of the dataset is: {shape_of_the_data}')
        
        chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
        formattedchatprompt = chatprompt.format_messages(shape_of_the_data=shape_of_the_data)
        response = llm.invoke(formattedchatprompt)
        response=response.content
        st.write(response)
        
    elif option=="Display data description":
        column_description=df.columns
        
        systemmessageprompt = SystemMessagePromptTemplate.from_template( 
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English.")
        humanmessageprompt = HumanMessagePromptTemplate.from_template(
        'The columns in the dataset are: {columns}')
        
        chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
        formattedchatprompt = chatprompt.format_messages(columns=column_description)
        response = llm.invoke(formattedchatprompt)
        response=response.content
        st.write(response)
        
    elif option=="Verify data integrity":
        def check(df):
            l=[]
            columns=df.columns
            for col in columns:
                dtypes=df[col].dtypes
                nunique=df[col].nunique()
                duplicated=df.duplicated().sum()
                sum_null=df[col].isnull().sum()
                l.append([col,dtypes,nunique,duplicated,sum_null])
            df_check=pd.DataFrame(l)
            df_check.columns=['columns','Data Types','No of Unique Values','No of Duplicated Rows','No of Null Values']
            return df_check 
        
        df_check = check(df)
        st.dataframe(df_check)

        systemmessageprompt = SystemMessagePromptTemplate.from_template( 
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English.")
        humanmessageprompt = HumanMessagePromptTemplate.from_template(
        'The columns in the dataset are: {df_check}')
        
        chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
        formattedchatprompt = chatprompt.format_messages(df_check=df_check)
        response = llm.invoke(formattedchatprompt)
        response=response.content
        st.write(response)
        
    elif option=="Summarize numerical data statistics":
        descibe_numerical = df.describe().T
        st.dataframe(descibe_numerical)
    
        systemmessageprompt = SystemMessagePromptTemplate.from_template( 
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English.")
        humanmessageprompt = HumanMessagePromptTemplate.from_template(
        'The columns in the dataset are: {columns}')
        
        chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
        formattedchatprompt = chatprompt.format_messages(columns=descibe_numerical)
        response = llm.invoke(formattedchatprompt)
        response=response.content
        st.write(response)
        
    elif option=="Summarize categorical data":
        categorical_df = df.select_dtypes(include=['object'])
        if categorical_df.empty:
            st.write("No categorical columns found.")
        else:
            des2 = categorical_df.describe()
            st.dataframe(des2)
            
        systemmessageprompt = SystemMessagePromptTemplate.from_template( 
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English.")
        humanmessageprompt = HumanMessagePromptTemplate.from_template(
        'The columns in the dataset are: {columns}')
        
        chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
        formattedchatprompt = chatprompt.format_messages(columns=des2)
        response = llm.invoke(formattedchatprompt)
        response=response.content
        st.write(response)
