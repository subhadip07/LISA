import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
import os
from streamlit_option_menu import option_menu
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
import os
import pyrebase

from functions import check
# from functions import interactive_data_cleaning

from dotenv import load_dotenv
load_dotenv()

# Firebase configuration and initialization
firebaseConfig = {
    'apiKey': "AIzaSyCW3RIp-Cz1bkK9LdXb7FEoWZ4QdwyoTY8",
    'authDomain': "lisa-streamlit.firebaseapp.com",
    'projectId': "lisa-streamlit",
    'databaseURL': "https://lisa-streamlit-default-rtdb.asia-southeast1.firebasedatabase.app",
    'storageBucket': "lisa-streamlit.appspot.com",
    'messagingSenderId': "338108058761",
    'appId': "1:338108058761:web:d5b286d1273df123f08a58",
    'measurementId': "G-ZLRFZ610C3"
  }

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

st.set_page_config(page_title="LISA : LLM Informed Statistical Analysis ",page_icon=":books:",layout = "wide")
tab1, tab2 = st.tabs(["Home", "ChatBot"])

# Initialize session state for login status
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

#sidebar    
with st.sidebar:
    st.title("LISA")
    
    if not st.session_state.is_logged_in:
        choice = st.selectbox('login/Signup', ['Login', 'Sign up'])
        email = st.text_input('Please enter your Email:')
        password = st.text_input('Please enter your password:', type='password')
        
        if choice == 'Login':
            if st.button('Login'):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.is_logged_in = True
                    st.success("Logged in successfully!")
                    st.experimental_rerun()
                except:
                    st.error("Invalid credentials. Please try again.")
        
        elif choice == 'Sign up':
            handle = st.text_input('Please input your Email', value='Default')
            if st.button('Create my Account'):
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    st.success('Your account is created successfully!')
                    st.info('Please login using the login option.')
                except:
                    st.error("Unable to create account. Please try again.")
    
    else:
        if st.button('Logout'):
            st.session_state.is_logged_in = False
            st.experimental_rerun()











    with st.sidebar.expander(":Red[Get Your Api Key Here]"):
        st.markdown("## How to use\n"
            "1. Enter your [Groq API key](https://console.groq.com/keys) below🔑\n" 
            "2. Upload a CSV file📄\n"
            "3. Let LISA do it's work!!!💬\n"
                )
    
    groq_api_key = st.text_input("Enter your Groq API key:", type="password",
            placeholder="Paste your Groq API key here (gsk_...)",
            help="You can get your API key from https://console.groq.com/keys")
    
    st.text("The below parameters like temperature and top-p play a crucial role in controlling the randomness and creativity of the generated text. Adjust these parameters according to your requirements.")    
    with st.sidebar.expander("Model Parameters"):
        model_name = st.selectbox("Select Model:", ["llama3-8b-8192","llama3-70b-8192","mixtral-8x7b-32768","gemma-7b-it","gemma2-9b-it"])
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        top_p = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=1.0, step=0.25)

    st.divider()

    # Initialize LLM only if API key is provided
    llm = None
    if groq_api_key:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model_name,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            st.sidebar.error(f"Error initializing model: {str(e)}")

with tab1:
    st.header("Welcome to LISA: LLM Informed Statistical Analysis 🎈", divider='rainbow')
    st.markdown("LISA is an innovative platform designed to automate your data analysis process using advanced Large Language Models (LLM) for insightful inferences. Whether you're a data enthusiast, researcher, or business analyst, LISA simplifies complex data tasks, providing clear and comprehensible explanations for your data.")
    st.markdown("LISA combines the efficiency of automated data processing with the intelligence of modern language models to deliver a seamless and insightful data analysis experience. Empower your data with LISA!")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        AgGrid(df,theme="balham")
        st.divider()

        option = st.selectbox("Select an option:", ["Show dataset dimensions","Display data description","Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data"])
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to use the analysis features.")
        elif llm is None:
            st.error("Failed to initialize the model. Please check your API key.")
        else:
            
            if option == "Show dataset dimensions":
                shape_of_the_data = df.shape
                systemmessageprompt = SystemMessagePromptTemplate.from_template( 
                "You are StatBot, an expert statistical analyst. "
                "Explain the output in simple English.")
                humanmessageprompt = HumanMessagePromptTemplate.from_template(
                'The columns in the dataset are: {columns}')
                
                chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
                formattedchatprompt = chatprompt.format_messages(columns=shape_of_the_data)
                response = llm.invoke(formattedchatprompt)
                response = response.content
                st.write(response)
                
            elif option == "Display data description":
                column_description = df.columns
                
                systemmessageprompt = SystemMessagePromptTemplate.from_template( 
                "You are StatBot, an expert statistical analyst. "
                "Explain the output in simple English.")
                humanmessageprompt = HumanMessagePromptTemplate.from_template(
                'The columns in the dataset are: {columns}')
                
                chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
                formattedchatprompt = chatprompt.format_messages(columns=column_description)
                response = llm.invoke(formattedchatprompt)
                response = response.content
                st.write(response)
                
            elif option == "Verify data integrity":
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
                response = response.content
                st.write(response)
                
            elif option == "Summarize numerical data statistics":
                describe_numerical = df.describe().T
                st.dataframe(describe_numerical)
            
                systemmessageprompt = SystemMessagePromptTemplate.from_template( 
                "You are StatBot, an expert statistical analyst. "
                "Explain the output in simple English.")
                humanmessageprompt = HumanMessagePromptTemplate.from_template(
                'The columns in the dataset are: {columns}')
                
                chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
                formattedchatprompt = chatprompt.format_messages(columns=describe_numerical)
                response = llm.invoke(formattedchatprompt)
                response = response.content
                st.write(response)
                
            elif option == "Summarize categorical data":
                categorical_df = df.select_dtypes(include=['object'])
                if categorical_df.empty:
                    st.write("No categorical columns found.")
                else:
                    describe_categorical = categorical_df.describe()
                    st.dataframe(describe_categorical)
                    
                systemmessageprompt = SystemMessagePromptTemplate.from_template( 
                "You are StatBot, an expert statistical analyst. "
                "Explain the output in simple English.")
                humanmessageprompt = HumanMessagePromptTemplate.from_template(
                'The columns in the dataset are: {columns}')
                
                chatprompt = ChatPromptTemplate.from_messages([systemmessageprompt, humanmessageprompt])
                formattedchatprompt = chatprompt.format_messages(columns=describe_categorical)
                response = llm.invoke(formattedchatprompt)
                response = response.content
                st.write(response)

with tab2:
    st.markdown("""Our integrated chatbot is available to assist you, providing real-time answers to your data-related queries and enhancing your overall experience with personalized support.""")
    st.markdown("""---""")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def get_response(query, chat_history):
        template = """
        You are a helpful assistant. Answer the following the user asks:
        
        Chat history:{chat_history}
        user question:{user_question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({
            "chat_history": chat_history,"user_question": query})

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
                
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to use the chatbot.")
    elif llm is None:
        st.error("Failed to initialize the model. Please check your API key.")
    else:
        user_query = st.chat_input("Type your message here")
        
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            with st.chat_message("AI"):
                ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
                
            st.session_state.chat_history.append(AIMessage(ai_response))

# with tab3:
#     interactive_data_cleaning()
    