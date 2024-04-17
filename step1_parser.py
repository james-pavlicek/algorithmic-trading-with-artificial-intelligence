# Welcome to the 1st part of Utilizing Artificial Intelligence for Algorithmic Trading.
# For this part we will be parsing through our initial dataset and then exporting for use in training our models.
# The dataset is 10k reports from the SEC and has been compiled by Bill McDonald at Notre Dame University. 
# Dataset: https://drive.google.com/drive/folders/1tZP9A0hrAj8ptNP3VE9weYZ3WDn9jHic

# This code has been built for a large scale research project for Texas State University.
# The research for the project has been conducted by James Pavlicek, Jack Burt, and Andrew Hocher.
# If you have any questions or inquiries feel free to reach out to me at https://www.jamespavlicek.com/ 
# This code is free use and anyone can use it.
# To start I have a summary below of what this project will do and the code will follow below.

#0. Import all necessary python packages.
#1. Building functions for extracting target text from the 10k documents.
#2. Find, Parse, and Return for specified QTR & YR
#3. Convert Company CIK to Ticker and Get 7 day Returns
#4. Append Company Return Info to Each Parsed 10k row
#5. Combine all QTR Csv into 1 Yearly
#6. Run all Functions for Year Range (2000-2023)
#7. Put all Info onto one CSV for Model Training


#-----------------------------------------------------------#
#------------------STEP 0: Import Packages------------------#
#-----------------------------------------------------------#

import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import glob
import yfinance as yf
import csv
from datetime import datetime, timedelta


#-----------------------------------------------------------#
#------STEP 1: Build Functions for Extracting Text----------#
#-----------------------------------------------------------#

# Combine all text extracting functions into one function
def extract_information_from_10k_v2(file_path):

    # Get company name from document header
    def extract_company_name(text):
        start_phrase = "company conformed name:"
        end_phrase = "central index key:"
        start_idx = text.find(start_phrase)
        if start_idx == -1:
            return "Company name not found"
        start_idx += len(start_phrase)
        end_idx = text.find(end_phrase, start_idx)
        if end_idx == -1:
            return "End phrase for company name not found"
        return text[start_idx:end_idx].strip()
    
    # Get filed as date from document header
    def extract_filed_as_of_date(text):
        start_phrase = "filed as of date:"
        start_idx = text.find(start_phrase)
        if start_idx == -1:
            return "Filed as of date not found"
        start_idx += len(start_phrase)
        end_idx = text.find(' ', start_idx)
        return text[start_idx:end_idx].strip()

    # Get CIK from document header
    def extract_central_index_key(text):
        start_phrase = "central index key:"
        start_idx = text.find(start_phrase)
        if start_idx == -1:
            return "Central index key not found"
        start_idx += len(start_phrase)
        end_idx = text.find(' ', start_idx)
        return text[start_idx:end_idx].strip()
    
    # Get company industry from document header
    def extract_standard_industrial_classification(text):
        start_phrase = "standard industrial classification:"
        start_idx = text.find(start_phrase)
        if start_idx == -1:
            return "Standard industrial classification not found"
        start_idx += len(start_phrase)
        end_idx = text.find(']', start_idx) + 1
        if len(text[start_idx:end_idx].strip()) > 100:
            return "Standard industrial classification not found"
        return text[start_idx:end_idx].strip()
    
    # Get the largest string of text with the starting point "Item 7" and ending point "Item 7a."
    def extract_largest_text_between_phrases(text, start_phrase, end_phrase):
        start_indices = [i for i in range(len(text)) if text.startswith(start_phrase, i)]
        end_indices = [i for i in range(len(text)) if text.startswith(end_phrase, i)]

        largest_text = ""
        largest_text_length = 0
        for start_idx in start_indices:
            end_idx = next((e for e in end_indices if e > start_idx), None)
            if end_idx is not None:
                extracted_text = text[start_idx:end_idx + len(end_phrase)]
                if len(extracted_text) > largest_text_length:
                    largest_text = extracted_text
                    largest_text_length = len(extracted_text)

        return largest_text if largest_text else "No suitable text found between the phrases."


    # Open file from computer. Clean and store text.
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', ' ').replace('\r', ' ').lower()

    # Extract target information with the functions defined above
    company_name = extract_company_name(text)
    filed_as_of_date = extract_filed_as_of_date(text)
    central_index_key = extract_central_index_key(text)
    standard_industrial_classification = extract_standard_industrial_classification(text)
    largest_text = extract_largest_text_between_phrases(text, "item 7. management", "item 7a. ")

    return company_name, filed_as_of_date, central_index_key, standard_industrial_classification, largest_text

# Count Words. Will be used later to verify parsed data. 
def count_words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    total_words = len(words)
    return total_words


#-----------------------------------------------------------#
#--STEP 2: Find, Parse, and Return for specified QTR & YR---#
#-----------------------------------------------------------#

# This function will combine the results from all the parsing functions above and store them into one dataframe.
# The dataframe will then be exported to a csv.
def super_function(year, folder):

    # This is where your files are located. Use your own path. I have left mine here for you to understand the structure.
    directory_path = f"/Users/jamespavlicek/Desktop/QMST/4320/Project/10_X_Data/{year}/{folder}"  

    # Initialize the DataFrame to store the extracted information.
    df = pd.DataFrame(columns=['Filename', 'Company Name', 'Filed As Of Date', 'Central Index Key', 'Standard Industrial Classification' , 'Largest Text'])
    
    # Loop through the all files in the directory_path.
    counter = 0
    for filename in os.listdir(directory_path):
        if "_10k_" in filename.lower() or "_10-K_" in filename:
            file_path = os.path.join(directory_path, filename)
            company_name, filed_as_of_date, central_index_key, standard_industrial_classification, largest_text = extract_information_from_10k_v2(file_path)
            if largest_text == "No suitable text found between the phrases.":
                continue
            
            elif count_words(largest_text) < 1000:
                continue

            else:
                # Create a DataFrame from the dictionary.
                new_row = pd.DataFrame([{
                    'Filename': filename,
                    'Company Name': company_name,
                    'Filed As Of Date': filed_as_of_date,
                    'Central Index Key': central_index_key,
                    'Standard Industrial Classification': standard_industrial_classification,
                    'Largest Text': largest_text
                }])

                # Add the new row to the existing dataframe.
                df = pd.concat([df, new_row], ignore_index=True)
                counter = counter + 1
                print(f"Count:{counter}, Year: {year}, Qtr: {folder}")

    # This is where the .csv will be stored. Use your own path. I have left mine here for you to understand the structure. 
    csv_file_path = f"/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Parsed/{year}_{folder}_10_K.csv"  # replace with your desired path and filename

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    print(f"DataFrame successfully saved to {csv_file_path}")


#-----------------------------------------------------------#
#---STEP 3: Convert CIK to Ticker and Get 7 day Returns-----#
#-----------------------------------------------------------#

#This is a stored file of all central index keys and their tickers to match. Tickers will be needed to find company returns with yahoo finance.
file_path = 'ticker_and_central_index_key.csv'


ticker_to_central_index_key = []

# Open the csv and store into dictionary
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        ticker_to_central_index_key.append(row)

ticker_to_cik_dict = {}

for item in ticker_to_central_index_key:
    cik = item['CIK']
    ticker = item['ticker']
    ticker_to_cik_dict[ticker] = cik

# Function to get the ticker from passing the cik as a variable.
def get_ticker_by_cik(cik, ticker_to_cik_dict):
    str(cik)
    for ticker, current_cik in ticker_to_cik_dict.items():
        if current_cik == cik:
            return ticker
    return "No ticker found for the given CIK."

# Get the 7 day return of a stock, in percent change and Positive or Negative. 
def get_percent_change_return_type(ticker,start_date):
        if ticker == "No ticker found for the given CIK." :
            percent_change = None
            return_type = None
        else: 
            start_date = str(start_date)
            start_date_object = datetime.strptime(start_date, "%Y%m%d")
            formatted_start_date = start_date_object.strftime("%Y-%m-%d")

            end_date_object = start_date_object + timedelta(days=7)
            formatted_end_date = end_date_object.strftime("%Y-%m-%d")
            
            stock_data = yf.download(ticker, start=formatted_start_date, end=formatted_end_date)
            
            if stock_data.empty:
                return None, None            

            if stock_data["Volume"][0] < 100000:
                return None, None

            percent_change = (stock_data["Adj Close"][-1] - stock_data["Adj Close"][0]) / stock_data["Adj Close"][0]
            return_type = ""

            if percent_change >= 0:
                return_type = "Positive"
            else:
                return_type = "Negative"

        return percent_change, return_type


#-----------------------------------------------------------#
#---STEP 4: Append Return Info to Each Parsed 10k row-------#
#-----------------------------------------------------------#

# Get the stock data from get_percent_change_return_type() and append it to the rows made in super_function().
def get_stock_data_onto_csv(filepath, year, folder):

    df = pd.read_csv(filepath)
    df['Ticker'] = None
    df['Percent Change (1 Week)'] = None
    df['Return Type'] = None

    counter = 0
    for index,row in df.iterrows(): 
        cik = row["Central Index Key"]
        cik = str(cik)
        start_date = row["Filed As Of Date"]
        ticker = get_ticker_by_cik(cik,ticker_to_cik_dict)

        percent_change, return_type = get_percent_change_return_type(ticker,start_date)

        df.at[index, 'Ticker'] = ticker.upper()
        df.at[index, 'Percent Change (1 Week)'] = percent_change
        df.at[index, 'Return Type'] = return_type

        counter = counter + 1 
  
    df_cleaned = df.dropna()
    df_cleaned.info()

    # This is where the .csv will be stored. Use your own path. I have left mine here for you to understand the structure. 
    df_cleaned.to_csv(f"/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Parsed_With_Returns/{year}_{folder}_10_K_with_returns.csv", index=False)

    print(f"{year}'s {folder} CSV file with returns has been successfully written.")


#-----------------------------------------------------------#
#------STEP 5: Combine all QTR Csvs into 1 Yearly Csv-------#
#-----------------------------------------------------------#
    
# This function combines all the reports that have the same year.
def combine_reports_by_year(year, directory):

    file_pattern = f'{year}_QTR*_10_K_with_returns.csv'
    path_pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(path_pattern)
    combined_csv = pd.concat([pd.read_csv(f) for f in file_list])

    # Save the combined data to a new CSV file. Use your own path.
    combined_csv.to_csv(f'/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Entire_Year_Final/{year}_final_combined_with_returns.csv', index=False)
    print(f"{year}'s CSV files have been successfully combined.")

#-----------------------------------------------------------#
#---STEP 6: Run all Functions for Year Range (2000-2023)----#
#-----------------------------------------------------------#

# The main function takes all the functions above and stores each csv into their specified path.
# Change the years to the range you want to collect. (Remember to add +1 to ending year.)
# THIS TAKES MULTIPLE HOURS TO RUN SO MAKE SURE YOU HAVE EVERYTHING 100% CORRECT BEFORE YOU START.
def main():
    folders = ["QTR1", "QTR2", "QTR3", "QTR4"]
    years = [str(year) for year in range(2000, 2024)]
    directory = '/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Parsed_With_Returns/'
    
    for year in years:
        for folder in folders:
            print(year, folder)
            super_function(year, folder)
            get_stock_data_onto_csv(f"/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Parsed/{year}_{folder}_10_K.csv", year, folder)
        
        combine_reports_by_year(year, directory)
    print("All CSV files have been successfully combined.")

if __name__ == "__main__":
    main()


#-----------------------------------------------------------#
#---STEP 7: Put all Info onto one CSV for Model Training----#
#-----------------------------------------------------------#

# Specify the directory where your files are located. Use your own path.
directory = '/Users/jamespavlicek/Desktop/QMST/4320/Project/10_K_Entire_Year_Final/'
years = [str(year) for year in range(2000, 2024)]

combined_csv = pd.DataFrame()

for year in years:
    file_pattern = f'{year}_final_combined_with_returns.csv'
    path_pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(path_pattern)

    if file_list:
        year_data = pd.concat([pd.read_csv(f) for f in file_list])
        combined_csv = pd.concat([combined_csv, year_data])
        print(f"{year}'s CSV files have been successfully combined.")
    else:
        print(f"No files found for {year}.")

# After the loop, save the combined data to a new CSV file. This is the final dataset that will be passed onto EDA and model training. See you in step 2!
combined_csv.to_csv('/Users/jamespavlicek/Desktop/QMST/4320/Project Final/Final_dataset.csv', index=False)

print("All CSV files have been successfully combined.")