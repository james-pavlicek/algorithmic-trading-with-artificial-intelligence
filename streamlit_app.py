# Welcome to the final part of Utilizing Artificial Intelligence for Algorithmic Trading.
# For this part we will be combining all the methods in this project into one web app to test on out of sample data.
# This app is built with streamlit and grabs the most recent 10k file from the selected stock,
# then it preforms model analysis and created a positive or negative prediction based on the Naive Bayes Meta Model (best performer).
# Lastly we will simulate a trade for the time period after the 10k was released and track the performance. 

# This code has been built for a large scale research project for Texas State University.
# The research for the project has been conducted by James Pavlicek, Jack Burt, and Andrew Hocher.
# If you have any questions or inquiries feel free to reach out to me at https://www.jamespavlicek.com/ 
# This code is free use and anyone can use it.
# To start I have a summary below of what this project will do and the code will follow below.

#0. Import all necessary python packages.
#1. Initializing Streamlit App.
#2. Displaying Company Info.
#3. Sourcing most recent 10k file.
#4. Parsing live 10k text.
#5. Data Cleaning.
#6. Predictions with Machine Learning Models.
#7. Predictions with Sentiment Models.
#8. 10k Textual Summary.
#9. Model Prediction Output.
#10. Naive Bayes Meta Model Prediction.
#11. Trade Performance.
#12. Documentation and Closing Remarks.

#-----------------------------------------------------------#
#------------------STEP 0: Import Packages------------------#
#-----------------------------------------------------------#

import streamlit as st
import pandas as pd
from joblib import load
import re
import numpy as np 
import nltk 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer  
import contractions  
from bs4 import BeautifulSoup 
from nltk.stem.wordnet import WordNetLemmatizer 
from datetime import datetime, timedelta
import yfinance as yf
import requests
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer
from dotenv import load_dotenv
from openai import OpenAI
import openai
import os 
import tiktoken
import time
import gc 
import warnings
warnings.filterwarnings("ignore")


#-----------------------------------------------------------#
#---------STEP 1: Initializing Streamlit App----------------#
#-----------------------------------------------------------#

# Load models, sp500 .csv, and nltk datasets into cache to help with memory issues 
@st.cache_data
def download_nltk_data():
    """Download necessary NLTK datasets."""
    nltk.download("wordnet")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download("words")

download_nltk_data()

@st.cache_data
def load_data(filename):
    """Load and return data from a CSV file."""
    return pd.read_csv(filename)

sp500_company_info = load_data("sp500_companies.csv")

st.header("""Algorithmic Trading with Artificial Intelligence""")

companies = sp500_company_info["Security"].tolist()
companies.insert(0, "")

#-------Import .Joblib files--------#
@st.cache_resource
def load_model(model_name):
    return load(model_name)

clf_loaded = load_model('random_forest_model_countvectorizer.joblib')
clf_loaded_2 = load_model('random_forest_model_tf_idf.joblib')
xgboost_loaded_1 = load_model('XGBoost_model_countvectorizer.joblib')
xgboost_loaded_2 = load_model('XGBoost_model_tf_idf.joblib')
naive_bayes_loaded_1 = load_model('naive_bayes_model_countvectorizer.joblib')
naive_bayes_loaded_2 = load_model('naive_bayes_model_tf_idf.joblib')
meta_naive_bayes_model = load_model('meta_naive_bayes_model_final.joblib')

# Define session variables
if 'option' not in st.session_state or 'stock_selected' not in st.session_state:
    st.session_state['option'] = ""
    st.session_state['stock_selected'] = False

# Build the stock selection dropdown
st.session_state['option'] = st.selectbox(
    "What company would you like to view? (~2 minute processing time)",
    companies, 
    index=0,
    placeholder="Select a stock...",
    on_change=lambda: setattr(st.session_state, 'stock_selected', True))

# Check if an stock is selected and the flag is True
if st.session_state['option'] and st.session_state['stock_selected']:
    
    # Start of the Progress bar
    option = st.session_state["option"]
    my_bar = st.progress(0, text=f'Stock Picked: {option}')
    time.sleep(.2)
    

    #-----------------------------------------------------------#
    #------------STEP 2: Displaying Company Info----------------#
    #-----------------------------------------------------------#
    
    my_bar.progress(10, text="Finding Company Info")
    
    company_info = sp500_company_info[sp500_company_info["Security"] == option].iloc[0]
    st.markdown("")
    st.subheader("Company Description")
    
    # Build rows to put data into "boxes"
    row1 = st.columns(3)
    row2 = st.columns(3)

    row1[0].markdown(f"**Company**")
    row1[0].caption(company_info['Security'])
    row1[1].markdown(f"**Ticker**")
    row1[1].caption(company_info['TICKER'])
    row1[2].markdown(f"**Sector** ")
    row1[2].caption(company_info['GICS-Sector'])
    row2[0].markdown(f"**Sub-Industry**")
    row2[0].caption(company_info['GICS Sub-Industry'])
    row2[1].markdown(f"**Headquarters Location** ")
    row2[1].caption(company_info['Headquarters Location'])
    row2[2].markdown(f"**Founded**")
    row2[2].caption(company_info['Founded'])
    st.markdown("")
    
    my_bar.progress(20, text="Displaying Company Info")
    

    #-----------------------------------------------------------#
    #----------STEP 3: Sourcing most recent 10k file------------#
    #-----------------------------------------------------------#

    cik = company_info['CIK']
    ticker = company_info['TICKER']
    cik = int(cik)
    cik = str(cik).zfill(10)
        
    # create request header
    headers = {'User-Agent': 'jamespavlicek@txstate.edu'}
    
    # Finding accessionNumber and other information for most recent 10k document
    filingMetadata = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json', headers=headers)
    
    allForms = pd.DataFrame.from_dict(filingMetadata.json()['filings']['recent'])
    
    filteredForms = allForms[allForms["form"] == "10-K"]
    most_recent_tenk = filteredForms.head(1)
    
    file_date = most_recent_tenk["filingDate"].iloc[0]
    
    formid = most_recent_tenk["accessionNumber"].iloc[0]
    primaryDocument = most_recent_tenk["primaryDocument"].iloc[0]
    adjusted_formid = formid.replace("-", "")
    
    URL_text = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adjusted_formid}/{formid}.txt"
    URL_text_to_display = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adjusted_formid}/{primaryDocument}"

    my_bar.progress(30, text="Finding Most Recent 10k Document. (~ 1 minute)")
    print(URL_text)


    #-----------------------------------------------------------#
    #------------STEP 4: Parsing live 10k text------------------#
    #-----------------------------------------------------------#

    response = requests.get(URL_text, headers=headers)
    
    # Parse the response
    soup = BeautifulSoup(response.content, 'lxml')
    
    # This function will get te largest text string between the pattern_start and pattern_end
    # Adapted from step 1 parser for use on live data
    def extract_largest_text_between_phrases(text, start_pattern, end_pattern):
        start_matches = list(re.finditer(start_pattern, text))
        end_matches = list(re.finditer(end_pattern, text))
    
        largest_text = ""
        largest_text_length = 0
    
        for start in start_matches:
            end = next((e for e in end_matches if e.start() > start.end()), None)
            if end:
                extracted_text = text[start.end():end.start()]
                if len(extracted_text) > largest_text_length:
                    largest_text = extracted_text
                    largest_text_length = len(extracted_text)
                
                # Only store the text if it is longer than 25000 characters
                if len(largest_text) > 25000:
                    largest_text = largest_text[:25000]
    
        return largest_text if largest_text else "No suitable text found between the phrases."
    
    for filing_document in soup.find_all('document'):
        document_type = filing_document.type.find(string=True, recursive=False).strip()
        if document_type == "10-K":
            TenKtext = filing_document.find('text').extract().text
    
    # Identify pattern start and end phrases. Then run the extract_largest_text_between_phrases() function
    pattern_start = re.compile(r"item\s*7\s*[:\.\-]?\s*(management's|management|discussion)", re.IGNORECASE | re.DOTALL)
    pattern_end = re.compile(r'item\s*7a\s*[:\-]?\s*', re.IGNORECASE)
    text = extract_largest_text_between_phrases(TenKtext, pattern_start, pattern_end)
    
    # If the text is less that 1000 words return ""Could not effectively parse 10k document."" and restart the app
    if len(text) < 1000:
        st.markdown("Could not effectively parse 10k document.")
        my_bar.progress(100, text='Please try another stock.')
        time.sleep(3)  
        st.session_state['option'] = ""  
        st.session_state['stock_selected'] = False 
        st.stop() 
        
    my_bar.progress(40, text="Reading Company 10k Document.")    
    df = pd.DataFrame({"Largest Text": [text]})
    df_2 = df.copy()

    #-----------------------------------------------------------#
    #-----------------STEP 5: Data Cleaning---------------------#
    #-----------------------------------------------------------#
    #Tag Removal
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    df["Largest Text"] = df["Largest Text"].apply(lambda x: strip_html(x))

    #Replace Contractions
    def replace_contractions(text):
        return contractions.fix(text)
    df["Largest Text"] = df["Largest Text"].apply(lambda x: replace_contractions(x))
    df_3 = df.copy()
    
    #Remove Number/ Special Characters
    def remove_special_characters(text, remove_digits=True):
        special = r"[^a-zA-Z0-9\s]" if not remove_digits else r"[^a-zA-z\s]"
        text = re.sub(special, "", text)
        return text
    df["Largest Text"] = df["Largest Text"].apply(lambda x: remove_special_characters(x))

    #Convert to lowercase
    def to_lowercase(text):
        lower = text.lower()
        return lower
    df["Largest Text"] = df["Largest Text"].apply(lambda x: to_lowercase(x))

    #Tokenization
    df['Largest Text'] = df['Largest Text'].astype(str)
    for index, row in df.iterrows():
        df.at[index, "Largest Text"] = nltk.word_tokenize(row["Largest Text"])

    #Stopwords
    stopwords = stopwords.words("english")
    customlist = ["not", "could", "did", "does", "had", "has", "have", "is", "ma", "might", "must", "need", "shall", "should", "was", "were", "will", "would"]
    stopwords = list(set(stopwords) - set(customlist))

    def remove_stopwords(words):
        new_words = []
        for word in words:
            if word not in stopwords:
                new_words.append(word)
        return new_words
    df["Largest Text"] = df["Largest Text"].apply(lambda x: remove_stopwords(x))

    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    def lemmatize_list(words):
        new_words = []
        for word in words:
            new_words.append(lemmatizer.lemmatize(word, pos="v"))
        return new_words
    df["Largest Text"] = df["Largest Text"].apply(lambda x: lemmatize_list(x))

    #Word list to text string
    def join_words(words):
        return " ".join(words)
    df["Largest Text"] = df["Largest Text"].apply(lambda x: join_words(x))

    my_bar.progress(50, text="Cleaning Text.")

    # Vectorization (Convert text data to numbers).
    bow_vec = CountVectorizer(max_features=300)
    data_features1 = bow_vec.fit_transform(df["Largest Text"])
    data_features1 = data_features1.toarray()
    shape_size = data_features1.shape
    X1 = data_features1

    vectorizer = TfidfVectorizer(max_features=300)
    data_features2 = vectorizer.fit_transform(df["Largest Text"])
    data_features2 = data_features2.toarray()
    X2 = data_features2

    my_bar.progress(60, text="Vectorizing Text.")


    #-----------------------------------------------------------#
    #----STEP 6: Predictions with Machine Learning Models-------#
    #-----------------------------------------------------------#

    predictions = clf_loaded.predict(X1)
    predictions_2 = clf_loaded_2.predict(X2)
    predictions_3 = xgboost_loaded_1.predict(X1)
    predictions_4 = xgboost_loaded_2.predict(X2)
    predictions_5 = naive_bayes_loaded_1.predict(X1)
    predictions_6 = naive_bayes_loaded_2.predict(X2)

    my_bar.progress(70, text="Formulating Predictions with Machine Learning Models.")
    #-----------------------------------------------------------#
    #--------STEP 7: Predictions with Sentiment Models----------#
    #-----------------------------------------------------------#
    
    # FinBERT has been removed due to competitive intensity in a virtual python environment    
    # FinBERT did not impact the Naive Bayes Meta Model

    #-------Text Blob Model--------#
    for index, row in df.iterrows():
        text = row['Largest Text']
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        df.at[index, 'TextBlob_Sentiment_Score'] = sentiment

        if sentiment > 0: 
            df.at[index, 'TextBlob_Sentiment_Type'] = int(1)
        else:
            df.at[index, 'TextBlob_Sentiment_Type'] = int(0)

    predictions_7 = df['TextBlob_Sentiment_Type'][0]

     #-------Word Sentiment Model--------#
    data = pd.read_csv("Loughran-McDonald_MasterDictionary.csv")
    
    testing_df = data

    negative_word_list = []
    positive_word_list = []
    
    for index, row in testing_df.iterrows():
        if row['Negative'] > 0:
            negative_word_list.append(row['Word'])
        if row['Positive'] > 0:
            positive_word_list.append(row['Word'])
    
    negative_word_list = [word.lower() for word in negative_word_list]
    positive_word_list = [word.lower() for word in positive_word_list]
    
    # Count all the negative and positive words in the text string
    def count_neg_and_pos_words(text, negative_word_list, positive_word_list):
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
    
        count_of_negative_words = 0 
        count_of_positive_words = 0 
        total_words = len(words)
        for word in words:
            word_lower = word.lower() 
            if word_lower in negative_word_list:
                count_of_negative_words += 1
            elif word_lower in positive_word_list:
                count_of_positive_words += 1
        
        return count_of_positive_words, count_of_negative_words, total_words
    
    # Assuming count_neg_and_pos_words is defined elsewhere and works as expected
    for index, row in df.iterrows():
        text = row['Largest Text'] 
        results = count_neg_and_pos_words(text, negative_word_list, positive_word_list)
    
        df.at[index, 'Positive_Word_Count_Loughran_McDonald'] = results[0]
        df.at[index, 'Negative_Word_Count_Loughran_McDonald'] = results[1]
        df.at[index, 'Total Word Count_Loughran_McDonald'] = results[2]
    
        # Calculate the Pos-Neg Score, strength_ratio and update the DataFrame
        Pos_neg_score = results[0] - results[1]
        df.at[index, "Word_Pos_minus_Neg_Score_Loughran_McDonald"] = Pos_neg_score 
        pos_word_ratio = results[0] / results[2]
        neg_word_ratio = results[1] / results[2]
        strength_ratio = pos_word_ratio - neg_word_ratio

        if strength_ratio > .03:
            df.at[index, 'Word_Sentiment_Type_Loughran_McDonald'] = 1
        else:
            df.at[index, 'Word_Sentiment_Type_Loughran_McDonald'] = 0

    predictions_8 = df['Word_Sentiment_Type_Loughran_McDonald'][0]
    my_bar.progress(80, text="Formulating Predictions with Text Sentiment Models.")

    #----------Chatgpt-----------------#
    open_ai_key = st.secrets["open_ai_key"]
    os.environ['OPENAI_API_KEY'] = open_ai_key 
    
    client = OpenAI(api_key=open_ai_key)
    
    load_dotenv()
    
    df['Chat_GPT_Sentiment'] = np.nan
    df['Chat_GPT_Sentiment_binary'] = np.nan
    
    openai.api_key = open_ai_key
    
    def send(
        prompt=None,
        text_data=None,
        chat_model="gpt-3.5-turbo-0125",
        model_token_limit=16000,
        max_tokens=10000,
    ):
        """
        Send the prompt at the start of the conversation and then send chunks of text_data to ChatGPT via the OpenAI API.
        If the text_data is too long, it splits it into chunks and sends each chunk separately.
    
        Args:
        - prompt (str, optional): The prompt to guide the model's response.
        - text_data (str, optional): Additional text data to be included.
        - max_tokens (int, optional): Maximum tokens for each API call. Default is 2500.
    
        Returns:
        - list or str: A list of model's responses for each chunk or an error message.
        """
    
        # Check if the necessary arguments are provided
        if not prompt:
            return "Error: Prompt is missing. Please provide a prompt."
        if not text_data:
            return "Error: Text data is missing. Please provide some text data."
    
        # Initialize the tokenizer
        tokenizer = tiktoken.encoding_for_model(chat_model)
    
        # Encode the text_data into token integers
        token_integers = tokenizer.encode(text_data)
    
        # Split the token integers into chunks based on max_tokens
        chunk_size = max_tokens - len(tokenizer.encode(prompt))
        chunks = [
            token_integers[i : i + chunk_size]
            for i in range(0, len(token_integers), chunk_size)]
    
        # Decode token chunks back to strings
        chunks = [tokenizer.decode(chunk) for chunk in chunks]
    
        responses = []
        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "To provide the context for the above prompt, I will send you text in parts. When I am finished, I will tell you 'ALL PARTS SENT'. Do not answer until you have received all the parts.",
            },]
    
        for chunk in chunks:
            messages.append({"role": "user", "content": chunk})
            while (
                sum(len(tokenizer.encode(msg["content"])) for msg in messages)
                > model_token_limit
            ):
                messages.pop(1)  # Remove the oldest chunk
    
            response = client.chat.completions.create(model=chat_model, messages=messages)
            chatgpt_response = response.choices[0].message
            responses.append(chatgpt_response)
    
        # Add the final "ALL PARTS SENT" message
        messages.append({"role": "user", "content": "ALL PARTS SENT"})
        response = client.chat.completions.create(model=chat_model, messages=messages)
        final_response = response.choices[0].message
        responses.append(final_response)
    
        return responses
    
    def text(text):
        """
        Cleans the provided text by removing URLs, email addresses, non-letter characters, and extra whitespace.
    
        Args:
        - text (str): The input text to be cleaned.
    
        Returns:
        - str: The cleaned text.
        """
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove everything that's not a letter (a-z, A-Z)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
    
        return text
    
    # Use the function
    def summarize_text(tenk_text):
        file_content = text(tenk_text)
        # Define your prompt
        prompt_text = "You are a financial analyst and I am given you a portion of the item 7 of a SEC 10k report. Please summarize this text in around 5 sentences or 100 words. This text will later be used for sentiment analysis so make sure not to remove it in the summary. I will keep sending parts of the document and keep summarizing each part until I stop sending."
        
        # Send the file content to ChatGPT
        responses = send(prompt=prompt_text, text_data=file_content)
        
        # Print the responses
        text_summary = []
        for response in responses:    
            extracted_content = response.content
            text_summary.append(extracted_content)
        return text_summary
    
    def analyze_sentiment(text):
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst that conducts sentiment analysis on SEC 10-k documents. You respond in either 'Positive' or 'Negative'."},
            {"role": "user", "content": f"I have a list of summaries from the item 7 of a SEC 10k report. The summaries provided are in the same order as the original 10k's item 7. Do not take into account any financial document I gave you previously, only look at the string of text that will be provided later in this prompt. What is the sentiment of the following text? (keep your answer to one word, Positive or Negative.) {text}"}
        ])
    
        message_text = completion.choices[0].message.content
        return message_text


    #-----------------------------------------------------------#
    #--------------STEP 8: 10k Textual Summary------------------#
    #-----------------------------------------------------------#

    def provide_summary(text):
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst who takes summaries of SEC 10k documents and provides summaries of them.."},
            {"role": "user", "content": f"I have a list of summaries from the item 7 of a SEC 10k report. The summaries provided are in the same order as the original 10k's item 7. Do not take into account any financial document I gave you previously, only look at the string of text that will be provided later in this prompt. Can you please write a 5 sentence summary of all of this text provided. Make sure to not include any number values. If you want to talk about financials just say increase or decrease. Write about how managers feel, product performance, and events. Here is the text: {text}"}
            ])
        tenk_summary_from_chatgpt = completion.choices[0].message.content
        return tenk_summary_from_chatgpt
    
    for index, row in df_2.iterrows():
        time.sleep(1)
        tenk_text = row["Largest Text"]
        text_summary = summarize_text(tenk_text)
        my_bar.progress(90, text="Summarizing Managment's Discussion and Analysis of Financial Condition.")
        message_text = analyze_sentiment(text_summary)
        print(f'The {index} 10k is : {message_text}')
        df.at[index, 'Chat_GPT_Sentiment'] = message_text
    
        if message_text == "Positive":
            df.at[index, 'Chat_GPT_Sentiment_binary'] = 1
        else:
            df.at[index, 'Chat_GPT_Sentiment_binary'] = 0

    predictions_10 = df['Chat_GPT_Sentiment_binary'][0]


    #-----------------------------------------------------------#
    #------------STEP 9: Model Prediction Output----------------#
    #-----------------------------------------------------------#

    #convert machine learning predictions to integer
    first_prediction = predictions[0]
    first_prediction = int(first_prediction)
    second_prediction = predictions_2[0]
    second_prediction = int(second_prediction)
    third_prediction = predictions_3[0]
    third_prediction = int(third_prediction)
    fourth_prediction = predictions_4[0]
    fourth_prediction = int(fourth_prediction)
    fifth_prediction = predictions_5[0]
    fifth_prediction = int(fifth_prediction)
    sixth_prediction = predictions_6[0]
    sixth_prediction = int(sixth_prediction)

    #convert all to string for display
    first_prediction = str(first_prediction)
    second_prediction = str(second_prediction)
    third_prediction = str(third_prediction)
    fourth_prediction = str(fourth_prediction)
    fifth_prediction = str(fifth_prediction)
    sixth_prediction = str(sixth_prediction)
    seventh_prediction = str(int(predictions_7))
    eighth_prediction = str(int(predictions_8))
    ninth_prediction = str(int(predictions_10))



    prediction_list = [first_prediction, second_prediction, third_prediction, fourth_prediction, fifth_prediction, sixth_prediction, seventh_prediction, eighth_prediction, ninth_prediction]
    
    colors = []
    neg_or_pos_prediction = []
    for prediction in prediction_list:
        if prediction == "1":
            string = "Positive"
            color = "green"
        else:
            string = "Negative"
            color = "red"
        neg_or_pos_prediction.append(string)
        colors.append(color)

    # Show outputs of all models
    st.subheader("Model Performance")

    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    
    row1[0].markdown(f"**Random Forest CV**")
    row1[0].caption(f':{colors[0]}[{neg_or_pos_prediction[0]}]')
    row1[1].markdown(f"**Random Forest TF-IDF**")
    row1[1].caption(f':{colors[1]}[{neg_or_pos_prediction[1]}]')
    row1[2].markdown(f"**XGBoost CV**")
    row1[2].caption(f':{colors[2]}[{neg_or_pos_prediction[2]}]')
    row2[0].markdown(f"**XGBoost TF-IDF**")
    row2[0].caption(f':{colors[3]}[{neg_or_pos_prediction[3]}]')
    row2[1].markdown(f"**Naive Bayes CV**")
    row2[1].caption(f':{colors[4]}[{neg_or_pos_prediction[4]}]')
    row2[2].markdown(f"**Naive Bayes TF-IDF**")
    row2[2].caption(f':{colors[5]}[{neg_or_pos_prediction[5]}]')                
    row3[0].markdown(f"**Textblob**")
    row3[0].caption(f':{colors[6]}[{neg_or_pos_prediction[6]}]')
    row3[1].markdown(f"**Word Sentiment**")
    row3[1].caption(f':{colors[7]}[{neg_or_pos_prediction[7]}]')
    row3[2].markdown(f"**Chat GPT**")
    row3[2].caption(f':{colors[8]}[{neg_or_pos_prediction[8]}]')
    st.markdown("")


    #-----------------------------------------------------------#
    #------STEP 10: Naive Bayes Meta Model Prediction-----------#
    #-----------------------------------------------------------#

    my_bar.progress(95, text="Computing Final prediction with Naive Bayes Meta Model.")

    columns = ["Random_Forest_CV_pred", "XGBoost_CV_TF_IDF_pred", "Naive_Bayes_CV_pred"]
    row_1 = [int(first_prediction), int(third_prediction), int(fifth_prediction)]

    df = pd.DataFrame([row_1], columns=columns)
    meta_prediction = meta_naive_bayes_model.predict(df)
    print(meta_prediction)
    st.markdown(meta_prediction)
    if meta_prediction[0] > 0:
        string = "Positive"
        color = "green"
    else:
        string = "Negative"
        color = "red"

    st.markdown(f"Meta-Model Prediction: :{color}[{string}]")
    tenk_summary_from_chatgpt_text = provide_summary(text_summary)
    st.subheader("Document Summary")
    st.markdown(tenk_summary_from_chatgpt_text)
    st.markdown("")
    st.subheader("Meta Model Trading vs Stock Market")


    #-----------------------------------------------------------#
    #----------------STEP 11: Trade Performance-----------------#
    #-----------------------------------------------------------#
    my_bar.progress(98, text='Simulating Trade Performance')
    
    ticker_save_for_later = ticker
    buy_or_short = meta_prediction[0]
    tickers = [ticker, "VTI"]
    
    # set start date file_date. Trade for 7 days.
    start_date = datetime.strptime(file_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=7)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch the stock price data for the ticker & VTI (Stock Market Benchmark)
    investment_values = {}
    percent_changes = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date_str, end=end_date_str)
        if not data.empty:
            
            # simulates a short trade with a trade amount of $1000
            if buy_or_short == 0 and ticker != "VTI":
                print("hi")
                initial_price = data['Adj Close'].iloc[0]
                shares = 1000 / initial_price
                data['Returns'] = data['Adj Close'].pct_change()
                data['Inverted Returns'] = 1 - data['Returns']
                data['Short Scenario'] = initial_price * (data['Inverted Returns'].cumprod() * shares)
                data['Short Scenario'].iloc[0] = 1000 
                investment_values[ticker] = data['Short Scenario']
                final_value = data['Short Scenario'].iloc[-1]
                percent_changes[ticker] = ((final_value - 1000) / 1000) * 100

            # simulates a buy trade with a trade amount of $1000
            else:
                initial_price = data['Adj Close'].iloc[0]
                shares = 1000 / initial_price
                data['Investment Value'] = data['Adj Close'] * shares
                investment_values[ticker] = data['Investment Value']
                final_value = data['Investment Value'].iloc[-1]
                percent_changes[ticker] = ((final_value - 1000) / 1000) * 100
    
    # Combine all the data to compile into a graph
    if investment_values:
        combined_investment_values = pd.concat(investment_values.values(), axis=1, keys=investment_values.keys())
    
        df_reset = combined_investment_values.reset_index()
        df_reset.rename(columns={df_reset.columns[0]: 'Date'}, inplace=True)
        df_reset['Date'] = pd.to_datetime(df_reset['Date'])
        df_chart = df_reset.melt('Date', var_name='Stock', value_name='Value')
        
        # Change ticker to "Meta Model" and VTI to "Stock Market"
        replacement_dict = {'VTI': 'Stock Market', str(ticker_save_for_later): 'Meta Model'}
        df_chart['Stock'] = df_chart['Stock'].replace(replacement_dict)
        
        # Change the graph to index at 1000 and not 0
        max_value = df_chart['Value'].max()
        min_value = df_chart['Value'].min()
        if max_value < 1000 or min_value > 1000:
            raise ValueError("1000 is not within the range of data values.")
        centered_range = max(max_value - 1000, 1000 - min_value)
        y_min = 1000 - centered_range
        y_max = 1000 + centered_range
        
        # Building Graph
        vega_spec = {
            'mark': 'line',
            'encoding': {
                'x': {'field': 'Date', 'type': 'temporal', 'title': 'Date'},
                'y': {
                    'field': 'Value',
                    'type': 'quantitative',
                    'title': 'Stock Price',
                    'scale': {'domain': [y_min, y_max]}
                },
                'color': {
                    'field': 'Stock',
                    'type': 'nominal',
                    'legend': {
                        'title': 'Stock',
                        'orient': 'top',  
                        'direction': 'horizontal' 
                    }
                }
            },
        }

        st.vega_lite_chart(df_chart, vega_spec, use_container_width=True)

    # Showing Trade Preformance in the app
    first_value = list(percent_changes.values())[0]
    vti_percent_change = percent_changes.get("VTI", 0)
    first_value = first_value * .01
    vti_percent_change = vti_percent_change * .01
    row1 = st.columns(2)
    
    time.sleep(.5)
    if first_value > 0 :
        ticker_color = "green"
    else:
        ticker_color = "red"

    if vti_percent_change > 0 :
        vti_color = "green"
    else:
        vti_color = "red"
    
    formatted_percentage_ticker = f"{first_value:.2%}"
    formatted_percentage_vti =f"{vti_percent_change:.2%}"

    row1[0].markdown(f"**Meta Model Return**")
    row1[0].caption(f":{ticker_color}[{formatted_percentage_ticker}]")
    row1[1].markdown(f"**Stock Market Return**")
    row1[1].caption(f":{vti_color}[{formatted_percentage_vti}]")
    st.markdown(" ")

    #-----------------------------------------------------------#
    #--------STEP 12: Documentation and Closing Remarks---------#
    #-----------------------------------------------------------#
    
    st.subheader("Documentation")
    st.markdown(f'10k Document Source: {URL_text_to_display}')
    pdf_of_poster = "https://www.bit.ly/trading-ai-poster"
    st.markdown(f'Link to the Full Analysis: : {pdf_of_poster}')

    st.divider()          
    st.write("Built by James Pavlicek. More info at jamespavlicek.com")
    st.write("Research Conducted by James Pavlicek, Jack Burt, and Andrew Hocher at Texas State University.")
    st.write("Version 0.1.0")
    
    time.sleep(.5)
    my_bar.progress(100, text='Done. Scroll to see your results!')
    
    time.sleep(5)
    my_bar.progress(100, text='')
    
    del text
    del df
    del message_text
    
    gc.collect()
