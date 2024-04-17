# Welcome to the 4th part of Utilizing Artificial Intelligence for Algorithmic Trading.
# For this part we will be building our meta models and graphing our final results.

# This code has been built for a large scale research project for Texas State University.
# The research for the project has been conducted by James Pavlicek, Jack Burt, and Andrew Hocher.
# If you have any questions or inquiries feel free to reach out to me at https://www.jamespavlicek.com/ 
# This code is free use and anyone can use it.
# To start I have a summary below of what this project will do and the code will follow below.

#0. Import all necessary python packages.
#1. Set up Seed iterations.
#2. Logistic Regression Meta Model.
#3. Naive Bayes Meta Model.
#4. Majority Vote Meta Model Dataframe Generation.
#5. Logistic Regression & Naive Bayes Meta Model Testing and Trading Dataframe.
#6. Benchmark Model Trading Testing.
#7. Portfolio Performance.
#8. Results Cleaning.
#9. Graphing Results.

# WARNING: THIS CODE IS COMPUTATIONALLY INTENSIVE AND TAKES 1+ DAY TO RUN!


#-----------------------------------------------------------#
#------------------STEP 0: Import Packages------------------#
#-----------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from joblib import load
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import random
from datetime import timedelta
import yfinance as yf


#-----------------------------------------------------------#
#---------------STEP 1: Set up Seed iterations--------------#
#-----------------------------------------------------------#

list_of_seeds = list(range(1, 1001))
collection_of_returns = []
collection_of_standard_deviations = []
collection_of_weekly_values = []

for i in list_of_seeds:
    print(i)

    # Use your own path. I did a little bit of pre-processing in excel, you basically want your company info and 0s or 1s for model predictions.
    data = pd.read_csv("/Users/jamespavlicek/Desktop/QMST/4320/Project Final/QMST_4320_model_results_2.csv")
    df = data.copy()
    
    # Data Preprocessing and Train / Test splits
    highest_return_row = df.nlargest(1, 'Percent Change (1 Week)')
    index_to_drop = df.nlargest(1, 'Return Type_actual').index
    df = df.drop(index_to_drop)

    lowest_return_row = df.nsmallest(1, 'Percent Change (1 Week)')
    index_to_drop = df.nsmallest(1, 'Return Type_actual').index

    df = df.drop(index_to_drop)

    # First Split is for meta model training, then meta model out of sample testing
    train_data, test_data = train_test_split(df, test_size=0.5)
    train_1, train_2 = train_test_split(train_data, test_size=0.5)
    df = train_1

    predicted_columns = ['Random_Forest_CV_pred', 'Random_Forest_TF_IDF_pred', 'XGBoost_CV_pred',
        'XGBoost_CV_TF_IDF_pred', 'Naive_Bayes_CV_pred',
        'Naive_Bayes_TF_IDF_pred', 'TextBlob_Sentiment_Type_adjusted_pred',
        'Word_Sentiment_Type_Loughran_McDonald_adjusted_pred',
        'FinBERT_Sentiment_Type_adjusted_pred', 'Chat_GPT_pred']
    

    #-----------------------------------------------------------#
    #--------STEP 2: Logistic Regression Meta Model-------------#
    #-----------------------------------------------------------#

    # List of all predictor column names
    columns = ['Random_Forest_CV_pred', 'Random_Forest_TF_IDF_pred', 'XGBoost_CV_pred',
            'XGBoost_CV_TF_IDF_pred', 'Naive_Bayes_CV_pred',
            'Naive_Bayes_TF_IDF_pred', 'TextBlob_Sentiment_Type_adjusted_pred',
            'Word_Sentiment_Type_Loughran_McDonald_adjusted_pred',
            'FinBERT_Sentiment_Type_adjusted_pred', 'Chat_GPT_pred']

    results = []

    # Loop over every combination of columns to find the best combination of models
    for r in range(1, len(columns) + 1):
        for combo in combinations(columns, r):
            accuracy_results = []
            for seed in range(1, 51):
                X = df[list(combo)]
                y = df['Return Type_actual']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                
                model = LogisticRegression(random_state=seed)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracy_results.append(accuracy)
            
            average_accuracy = np.mean(accuracy_results)
            results.append({'Columns': combo, 'Average_Accuracy': average_accuracy})

    # Grab the number 1 result
    sorted_results = sorted(results, key=lambda x: x['Average_Accuracy'], reverse=True)
    top_model = sorted_results[:1]
    first_dict = top_model[0]
    columns = first_dict['Columns']
    columns_list_logistic = list(columns)

    #Train Meta Model with best preforming combination of columns 
    df = train_2
    random_seeds = list(range(1, 51))
    accuracy_results = []

    for seed in random_seeds:
        y = df['Return Type_actual']
        X = df[columns_list_logistic]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        meta_learner = LogisticRegression(random_state=seed)
        meta_learner.fit(X_train, y_train)
        
        predictions = meta_learner.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_results.append(accuracy)

    average_logistic_Regression_accuracy = np.mean(accuracy_results)

    # Save the model to use later, use your own path.
    dump(meta_learner, '/Users/jamespavlicek/Desktop/QMST/4320/Project Final/meta_logistic_regression_model.joblib')


    #-----------------------------------------------------------#
    #------------STEP 3: Naive Bayes Meta Model-----------------#
    #-----------------------------------------------------------#
    
    df = train_1

    # List of all predictor column names
    columns = ['Random_Forest_CV_pred', 'Random_Forest_TF_IDF_pred', 'XGBoost_CV_pred',
            'XGBoost_CV_TF_IDF_pred', 'Naive_Bayes_CV_pred',
            'Naive_Bayes_TF_IDF_pred', 'TextBlob_Sentiment_Type_adjusted_pred',
            'Word_Sentiment_Type_Loughran_McDonald_adjusted_pred',
            'FinBERT_Sentiment_Type_adjusted_pred', 'Chat_GPT_pred']

    results_NB = []

    # Loop over every combination of columns
    for r in range(1, len(columns) + 1):
        for combo in combinations(columns, r):
            accuracy_results_NB = []
            for seed in range(1, 51):
                X = df[list(combo)]
                y = df['Return Type_actual']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                
                model = GaussianNB()
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracy_results_NB.append(accuracy)
            
            average_accuracy = np.mean(accuracy_results_NB)
            
            results_NB.append({'Columns': combo, 'Average_Accuracy': average_accuracy})

    # Grab the number 1 result
    sorted_results = sorted(results_NB, key=lambda x: x['Average_Accuracy'], reverse=True)
    top_model = sorted_results[:1]
    first_dict = top_model[0]
    columns = first_dict['Columns']
    columns_list_nb = list(columns)

    #Train Meta Model with best preforming combination of columns 
    df = train_2

    random_seeds = list(range(1, 51))
    accuracy_results = []

    for seed in random_seeds:
        y = df['Return Type_actual']
        X = df[columns_list_nb]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        meta_learner_2 = GaussianNB()
        meta_learner_2.fit(X_train, y_train)
        
        predictions = meta_learner_2.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        accuracy_results.append(accuracy)

    average_naive_bayes_accuracy = np.mean(accuracy_results)

    # Save the model to use later, use your own path.
    dump(meta_learner_2, '/Users/jamespavlicek/Desktop/QMST/4320/Project/meta_naive_bayes_model_final.joblib')


    #-----------------------------------------------------------#
    #--STEP 4: Majority Vote Meta Model Dataframe Generation----#
    #-----------------------------------------------------------#

    df = test_data

    accuracy_thresholds = [6,7,8,9,10]
    number_of_rows = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    filtered_dfs = {}

    # Make a dataframe for each row that matches the accuracy_thresholds criteria.
    # For example, if accuracy_thresholds = 8, add all rows were 8/10 of the model predicted the same thing.
    for accuracy_threshold in accuracy_thresholds:
        
        df['sum_of_predictions'] = df[predicted_columns].sum(axis=1)
        df['majority_vote'] = np.nan
        total_models = len(predicted_columns)
        filtered_condition = (df['sum_of_predictions'] >= accuracy_threshold) | (df['sum_of_predictions'] <= (total_models - accuracy_threshold))
        filtered_df = df[filtered_condition]
        
        def apply_majority_vote(row):
            if row['sum_of_predictions'] >= accuracy_threshold:
                return 1
            else:
                return 0
        
        df['majority_vote'] = df.apply(lambda row: apply_majority_vote(row), axis=1)
        filtered_df['majority_vote'] = filtered_df.apply(lambda row: apply_majority_vote(row), axis=1)
        number_of_rows.append(len(filtered_df))
        
        # And that 'majority_vote' is used for predictions in filtered_df
        actual = filtered_df['Return Type_actual'].astype(bool)
        predicted = filtered_df['majority_vote'].astype(bool)
        
        # Calculate Accuracy /Sensitivity / Specificity
        accuracy = accuracy_score(actual, predicted)
        accuracy_list.append(accuracy)
        sensitivity = recall_score(actual, predicted)
        sensitivity_list.append(sensitivity)   
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)

        #Store the dataframes
        filtered_dfs[accuracy_threshold] = filtered_df


    majority_vote_results_df = pd.DataFrame({
        'Accuracy Threshold': accuracy_thresholds,
        'Number of Rows': number_of_rows,
        'Accuracy': accuracy_list,
        'Sensitivity': sensitivity_list,
        'Specificity': specificity_list
    })


    #-----------------------------------------------------------------------------------#
    #-STEP 5:Logistic Regression & Naive Bayes Meta Model Testing and Trading Dataframe-#
    #-----------------------------------------------------------------------------------#

    #Logistic Regression Meta Model Testing
    meta_logistic_regression_model = load('/Users/jamespavlicek/Desktop/QMST/4320/Project Final/meta_logistic_regression_model.joblib')
    y_true = df['Return Type_actual']
    df_for_prediction = df[columns_list_logistic]
    meta_log_predictions = meta_logistic_regression_model.predict(df_for_prediction)
    accuracy = accuracy_score(y_true, meta_log_predictions)

    # Build logistic regression trading dataframe
    df['meta_log_predictions'] = np.nan
    df['meta_log_predictions'] = meta_log_predictions
    df['majority_vote'] = df['meta_log_predictions']

    meta_log_df = df.copy()


    # Naive Bayes Meta Model Testing
    df = test_data
    meta_naive_bayes_model = load('/Users/jamespavlicek/Desktop/QMST/4320/Project/meta_naive_bayes_model_final.joblib')
    y_true = df['Return Type_actual']
    df_for_prediction = df[columns_list_nb]
    meta_nb_predictions = meta_naive_bayes_model.predict(df_for_prediction)
    accuracy = accuracy_score(y_true, meta_nb_predictions)

    # Build naive bayes trading dataframe
    df['meta_nb_predictions'] = np.nan
    df['meta_nb_predictions'] = meta_nb_predictions
    df['majority_vote'] = df['meta_nb_predictions']

    meta_nb_df = df.copy()


    #--------------------------------------------------------------#
    #-----------STEP 6: Benchmark Model Trading Testing------------#
    #--------------------------------------------------------------#

    # Chat GPT API Benchmark
    df_chat_gpt = meta_log_df.copy()
    df_chat_gpt['majority_vote'] = df_chat_gpt['Chat_GPT_pred']

    # Random Coinflip Benchmark
    df_random = meta_log_df.copy()

    # Generate a list of random choices for each row in the DataFrame aka: "a coin flip"
    df_random['random_column'] = [random.choice([1, 0]) for _ in range(len(df_random))]
    df_random['majority_vote'] = df_random['random_column']


    #--------------------------------------------------------------#
    #---------------STEP 7: Portfolio Performance------------------#
    #--------------------------------------------------------------#

    df['Filed As Of Date'] = pd.to_datetime(df['Filed As Of Date'], format='%Y%m%d')
    df['Trade Sell Date'] = df['Filed As Of Date'] + timedelta(days=7)

    stripped_df = df[['Company Name', 'Filed As Of Date',  'Trade Sell Date', 'Central Index Key', 'Ticker', 'Percent Change (1 Week)', 'majority_vote']]

    dfs = [filtered_dfs[6], filtered_dfs[7], filtered_dfs[8], filtered_dfs[9], filtered_dfs[10], meta_log_df, meta_nb_df, df_chat_gpt, df_random]

    for i, dataframe in enumerate(dfs):
        dataframe['Filed As Of Date'] = dataframe['Filed As Of Date'].astype(str)
        sorted_df = dataframe.sort_values(by='Filed As Of Date', ascending=True)
        sorted_df['Filed As Of Date'] = pd.to_datetime(sorted_df['Filed As Of Date'], format='%Y%m%d')
        sorted_df['Trade Sell Date'] = sorted_df['Filed As Of Date'] + timedelta(days=7)
        stripped_df = sorted_df[['Company Name', 'Filed As Of Date',  'Trade Sell Date', 'Central Index Key', 'Ticker', 'Percent Change (1 Week)', 'majority_vote']]
        dfs[i] = stripped_df

    #Setting the default colors and chart
    colors = ['royalblue', 'green', 'red', 'purple', 'orange', 'pink', 'lightblue', 'lightgreen', 'cyan']  
    sns.set_style("darkgrid")

    # Set variables for simulation. 
    # start_amount: Starting trading amount in USD for the trading simulation
    # years_back: How many years of data to look at. date range is [end_date - years_back) - (end_date)]
    # trade_percent: What percent of current portfolio balance you want to allocate to a simulated trade 
    # end_date: set to Jan 1st of 2024 because thats when our dataset ends
    start_amount = 100000
    years_back = 3
    trade_percent = 0.1
    end_date = pd.to_datetime('2024-01-01')
    start_date = end_date - pd.DateOffset(years=years_back)

    def simulate_trades(df, end_date, start_amount, years_back, trade_percent):
        filtered_df = df[df["Filed As Of Date"] >= start_date]
        df = filtered_df

        portfolio_value = start_amount
        
        portfolio_value = start_amount
        portfolio_values = [start_amount]
        dates = [df["Filed As Of Date"].min() - timedelta(days=1)]  # Start date

        for index, row in df.iterrows():
            trade_amount = portfolio_value * trade_percent
            percent_change = row["Percent Change (1 Week)"]
            profit_loss = trade_amount * percent_change * (1 if row["majority_vote"] == 1 else -1)
            portfolio_value += profit_loss
            portfolio_values.append(portfolio_value)
            dates.append(row["Filed As Of Date"])

        # Append the last portfolio balance to the end date so graphs end on last date and not last trade
        if dates[-1] != end_date: 
            dates.append(end_date) 
            portfolio_values.append(portfolio_value) 
        
        portfolio_start_df = pd.DataFrame({
            "Date": start_date,
            "Portfolio Balance": start_amount}, index=[0])

        portfolio_results_df = pd.DataFrame({
            "Date": dates,
            "Portfolio Balance": portfolio_values})

        final_portfolio_results_df = pd.concat([portfolio_start_df, portfolio_results_df], ignore_index=True)

        
        return dates, portfolio_values, final_portfolio_results_df

    line_labels = ["Majority Vote = 6", "Majority Vote = 7", "Majority Vote = 8", "Majority Vote = 9", "Majority Vote = 10", "Logistic Regression", "Naive Bayes", "Chat GPT", "Random"]

    porfolio_results_dataframes = {}

    for i, df in enumerate(dfs):
        df["Filed As Of Date"] = pd.to_datetime(df["Filed As Of Date"])
        dates, portfolio_values, portfolio_results_df = simulate_trades(df, end_date, start_amount, years_back, trade_percent)
        porfolio_results_dataframes[i] = portfolio_results_df

    import pandas as pd
    import numpy as np

    index_list = [0,1,2,3,4,5,6,7,8]
    annual_std_list = []
    annual_returns_list = []
    all_weekly_dfs = []

    # Calculating Returns and Standard Deviation. Also normalizing both to weekly returns.
    for i in index_list:
        df = porfolio_results_dataframes[i]

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)   
        df = df.loc[start_date:end_date]
        last_row_value = df.iloc[-1]['Portfolio Balance']
        aar = (last_row_value / start_amount) ** (1 / years_back) - 1
        aar_percentage = aar * 100
        annual_returns_list.append(aar_percentage)
        
        weekly_df = df.resample('W').last()
        weekly_df['Portfolio Balance'].fillna(method='ffill', inplace=True)
        weekly_df.reset_index(inplace=True)
        weekly_df['Return Change'] = np.log(weekly_df['Portfolio Balance'] / weekly_df['Portfolio Balance'].shift(1))
        weekly_df = weekly_df.dropna()
        
        weekly_volatility = weekly_df["Return Change"].std()
        daily_volatility = weekly_volatility / np.sqrt(5)  
        annual_std_dev = daily_volatility * np.sqrt(252)
        annual_std_dev_percent = annual_std_dev * 100
        annual_std_list.append(annual_std_dev_percent)
        all_weekly_dfs.append(weekly_df)

    collection_of_returns.append(annual_returns_list)
    collection_of_standard_deviations.append(annual_std_list)
    collection_of_weekly_values.append(all_weekly_dfs)


#--------------------------------------------------------------#
#------------------STEP 8: Results Cleaning--------------------#
#--------------------------------------------------------------#
    
collection_of_returns_df = pd.DataFrame(collection_of_returns, columns=["Majority Vote = 6", "Majority Vote = 7", "Majority Vote = 8", "Majority Vote = 9", "Majority Vote = 10", "Logistic Regression", "Naive Bayes", "Chat GPT", "Random"])
collection_of_standard_deviations_df = pd.DataFrame(collection_of_standard_deviations, columns=["Majority Vote = 6", "Majority Vote = 7", "Majority Vote = 8", "Majority Vote = 9", "Majority Vote = 10", "Logistic Regression", "Naive Bayes", "Chat GPT", "Random"])

ret_column_averages = collection_of_returns_df.mean().tolist()
std_column_averages = collection_of_standard_deviations_df.mean().tolist()

print(ret_column_averages)
print(std_column_averages)

first_models = [group[0] for group in collection_of_weekly_values if len(group) > 1]
second_models = [group[1] for group in collection_of_weekly_values if len(group) > 1]
third_models = [group[2] for group in collection_of_weekly_values if len(group) > 1]
fourth_models = [group[3] for group in collection_of_weekly_values if len(group) > 1]
fifth_models = [group[4] for group in collection_of_weekly_values if len(group) > 1]
sixth_models = [group[5] for group in collection_of_weekly_values if len(group) > 1]
seventh_models = [group[6] for group in collection_of_weekly_values if len(group) > 1]
eighth_models = [group[7] for group in collection_of_weekly_values if len(group) > 1]
ninth_models = [group[8] for group in collection_of_weekly_values if len(group) > 1]

list_of_models =[first_models,second_models,third_models,fourth_models,fifth_models,sixth_models,seventh_models,eighth_models,ninth_models]

all_models_df = pd.DataFrame()

list_of_models = [first_models, second_models, third_models, fourth_models, fifth_models, sixth_models, seventh_models, eighth_models, ninth_models]

for i, models in enumerate(list_of_models):
    for j, model in enumerate(models):
        if j == 0:
            edited_model = model[['Date', 'Portfolio Balance']].copy()
            edited_model.rename(columns={'Portfolio Balance': f'Portfolio Balance_{i}_0'}, inplace=True)
        else:
            edited_model = model[['Portfolio Balance']].copy()
            edited_model.rename(columns={'Portfolio Balance': f'Portfolio Balance_{i}_{j}'}, inplace=True)
        
        if all_models_df.empty:
            all_models_df = edited_model
        else:
            all_models_df = pd.concat([all_models_df, edited_model], axis=1)

new_row = pd.DataFrame([[start_amount] * len(all_models_df.columns)], columns=all_models_df.columns)
new_row.iloc[0, 0] = start_date
all_models_df = pd.concat([new_row, all_models_df]).reset_index(drop=True)
all_models_df = all_models_df.loc[:,~all_models_df.columns.duplicated()]

model_groups = {}

for i in range(9): 
    group_columns = [col for col in all_models_df.columns if col.startswith('Portfolio Balance_{}_'.format(i)) or col == 'Date']
    model_groups[i] = all_models_df[group_columns].copy()

for i, df in model_groups.items():
    portfolio_balance_columns = [col for col in df.columns if col != 'Date']  
    df[f'Portfolio Balance Average_{i}'] = df[portfolio_balance_columns].mean(axis=1)


#--------------------------------------------------------------#
#-------------------STEP 9: Graphing Results-------------------#
#--------------------------------------------------------------#

average_columns = []
average_columns.append(model_groups[0][['Date']].copy())

for i in range(len(model_groups)):
    avg_col_name = f'Portfolio Balance Average_{i}'
    if avg_col_name in model_groups[i].columns:
        temp_df = model_groups[i][[avg_col_name]].rename(columns={avg_col_name: f'Portfolio Balance Avg Group {i}'})
        average_columns.append(temp_df)

all_averages_df = pd.concat(average_columns, axis=1)
all_averages_df = all_averages_df.loc[:,~all_averages_df.columns.duplicated()]

print("New DataFrame with all Portfolio Balance Average columns:")
print(all_averages_df.head())

# Used to calculate average portfolio balance
all_averages_df.to_csv("/Users/jamespavlicek/Desktop/QMST/4320/Project Final/averages_csv.csv", index=False)


#Start of first Graph - Portfolio Performance over time with confidence interval bands
plt.figure(figsize=(14, 7))

column_labels = {
    'Portfolio Balance Avg Group 0': "Majority Vote = 6",
    'Portfolio Balance Avg Group 1': "Majority Vote = 7",
    'Portfolio Balance Avg Group 2': "Majority Vote = 8",
    'Portfolio Balance Avg Group 3': "Majority Vote = 9",
    'Portfolio Balance Avg Group 4': "Majority Vote = 10",
    'Portfolio Balance Avg Group 5': "Logistic Regression",
    'Portfolio Balance Avg Group 6': "Naive Bayes",
    'Portfolio Balance Avg Group 7': "Chat GPT",
    'Portfolio Balance Avg Group 8': "Random"
}

#Get VTI onto the graph
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=years_back)
vti_data = yf.download('VTI', start=start_date.strftime('%Y-%m-%d'), end=end_date)

# Calculate the portfolio value over time
initial_vti_price = vti_data['Adj Close'][0]
shares_bought = start_amount / initial_vti_price
vti_data['Portfolio Value'] = shares_bought * vti_data['Adj Close']

# Align vti_data to match the date range of all_averages_df
vti_data.index = pd.to_datetime(vti_data.index)
vti_data_aligned = vti_data.reindex(all_averages_df['Date'], method='nearest')

# Plot VTI performance
vti_line_color = 'lightgreen'
plt.plot(vti_data_aligned.index, vti_data_aligned['Portfolio Value'], label='Stock Market', color=vti_line_color)

# Calculate confidence interval
std_column_vti = np.std(vti_data_aligned['Portfolio Value'])  # Standard deviation for each column
lower_bound_vti = vti_data_aligned['Portfolio Value'] - std_column_vti
upper_bound_vti = vti_data_aligned['Portfolio Value'] + std_column_vti
    
plt.fill_between(all_averages_df['Date'], lower_bound_vti, upper_bound_vti, color=vti_line_color, alpha=0.07)

# Plot each portfolio balance average group with corresponding confidence interval
for i, column in enumerate(all_averages_df.columns[1:]):
    plt.plot(all_averages_df['Date'], all_averages_df[column], label=column_labels.get(column, "Unknown"))
    std_column = np.std(all_averages_df[column]) 
    lower_bound = all_averages_df[column] - std_column
    upper_bound = all_averages_df[column] + std_column
    plt.fill_between(all_averages_df['Date'], lower_bound, upper_bound, alpha=0.07)

plt.title('Model Performance Over Time (3Y)')
plt.xlabel('Date')
plt.ylabel('Portfolio Balance')
plt.legend()
plt.show()


print(ret_column_averages)
print(std_column_averages)

# Start of second Graph - Returns vs Standard Deviation scatter plot
line_labels = ["Majority Vote = 6", "Majority Vote = 7", "Majority Vote = 8", "Majority Vote = 9", "Majority Vote = 10", "Logistic Regression", "Naive Bayes", "Chat GPT", "Random"]

plt.figure(figsize=(10, 6))
offset_factor = 0.02 

global_offset_x = (np.max(std_column_averages) - np.min(std_column_averages)) * offset_factor
global_offset_y = (np.max(ret_column_averages) - np.min(ret_column_averages)) * offset_factor

for i in range(len(ret_column_averages)):
    plt.scatter(std_column_averages[i], ret_column_averages[i], edgecolor='k', s=100) 
    plt.text(std_column_averages[i] + global_offset_x, ret_column_averages[i] + global_offset_y, line_labels[i], fontsize=9, horizontalalignment='center', verticalalignment='bottom')

stock_market_std = 8.5
stock_market_ret = 17
plt.scatter(stock_market_std, stock_market_ret, color='lightgreen', edgecolor='k', s=100)
plt.text(stock_market_std + global_offset_x, stock_market_ret + global_offset_y, "Stock Market", fontsize=9, horizontalalignment='center', verticalalignment='bottom')
plt.title('Annual Returns vs. Standard Deviation')
plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Annual Returns')
plt.grid(True)
plt.show()

#Calculate VTI returns and standard deviation on the same weekly basis as the other models and benchmarks
vti_data = yf.download('VTI', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

initial_vti_price = vti_data['Adj Close'][0]
shares_bought = start_amount / initial_vti_price
vti_data['Portfolio Value'] = shares_bought * vti_data['Adj Close']

vti_data.index = pd.to_datetime(vti_data.index)
vti_data = vti_data[start_date:end_date]
last_row_value = vti_data.iloc[-1]['Portfolio Value']

aar = (last_row_value / start_amount) ** (1 / years_back) - 1
vti_aar_percentage = aar * 100

vti_weekly_df = vti_data.resample('W').last()
vti_weekly_df['Portfolio Value'].fillna(method='ffill', inplace=True)
vti_weekly_df.reset_index(inplace=True)

vti_weekly_df['Return Change'] = np.log(vti_weekly_df['Portfolio Value'] / vti_weekly_df['Portfolio Value'].shift(1))
vti_weekly_df = vti_weekly_df.dropna()

weekly_volatility = vti_weekly_df["Return Change"].std()
daily_volatility = weekly_volatility / np.sqrt(5)  
annual_std_dev = daily_volatility * np.sqrt(252) 
vti_annual_std_dev_percent = annual_std_dev * 100

print(vti_aar_percentage)
print(vti_annual_std_dev_percent)