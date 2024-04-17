# Utilizing Artificial Intelligence for Algorithmic Trading

## Overview
This research project was conducted by James Pavlicek, Jack Burt, and Andrew Hocher. We would like to thank and acknowledge our professor, Tahir Ekin Ph.D., for his guidance and expertise throughout our project.

Contact Info:

James Pavlicek: https://www.linkedin.com/in/jamespavlicek/ 

Jack Burt: https://www.linkedin.com/in/jackburt2/ 

Andrew Hocher: https://www.linkedin.com/in/andrew-hocher/ 

Link to our official research poster: https://github.com/james-pavlicek/algorithmic-trading-with-artificial-intelligence/blob/main/AI_for_algorithmic_trading_poster.pdf 


The code for our project is split into 5 different files. The first 4 are sequential and are labeled as step_1, step_2, etc. These files will walk you through each of our steps to our final output of “streamlit_ap.py” where we take aspects from all the previous steps to build one web app. Our   web app tests out of sample SEC 10k data and simulates a trade for the time period after the 10k was released and tracks the performance. If you have any questions about the project feel free to reach out to any of us and we would be more than happy to help out or guide you in the right direction.


## Abstract
The Efficient Market Hypothesis (EMH) has been a cornerstone of financial theory, suggesting that stock market prices fully reflect all available information. This research challenges the EMH by investigating whether natural language processing (NLP) techniques can extract predictive sentiments from 10-K financial reports to inform stock market investment strategies. We employed a variety of NLP methods to develop models capable of producing binary outputs, indicating potential market movements as either "Positive" or "Negative." These models were then integrated into several ensemble meta-models, aiming to enhance prediction accuracy by leveraging the strengths of individual models. Our findings reveal that, although our approach surpassed the no-information rate, indicating some degree of predictive capability, it failed to outperform the Total Stock Market Index (VTI) when backtesting our algorithmic trading. This outcome suggests that while NLP-based analysis of financial documents may hold predictive value. However, it does not outperform the market through our research, providing a sufficient basis to accept our null hypothesis that markets are efficient. Our research contributes to the ongoing debate on market efficiency by highlighting the complexities and limitations of applying NLP methods within financial market predictions. 

## Introduction
This project explores the application of Natural Language Processing (NLP) and Machine Learning (ML) techniques to predict stock market movements based on 10-K financial reports. We examine whether AI can effectively interpret financial reports to outperform the market, testing against the Efficient Market Hypothesis (EMH).

## Research Question
Can artificial intelligence be used to interpret 10-K financial reports and predict stock market movements?

### Hypotheses
- **Null Hypothesis (Ho):** NLP analysis of 10-K financial reports does not provide predictive insights that outperform the market, consistent with the EMH.
- **Alternative Hypothesis (Ha):** NLP analysis of 10-K financial reports provides insights that could potentially outperform the market.

## Models Employed
- **Naïve Bayes:** A classification algorithm based on Bayes' Theorem, assuming independence among predictors.
- **Random Forest:** A versatile machine learning algorithm for classification and regression tasks.
- **XGBoost:** A scalable machine learning algorithm using gradient boosting frameworks.
- **FinBERT:** A NLP model trained specifically on financial texts for sentiment analysis.
- **TextBlob:** Analyzes text sentiment by evaluating polarity and subjectivity.
- **Word Sentiment Analysis:** Frequency analysis of positive and negative finance words.

## Data and Methodology
Our data comprises a collection of 10-Ks spanning from 2000 to 2023, processed to remove HTML tags, special characters, and numbers, and to apply tokenization, stop-word removal, and lemmatization. The data was vectorized using Count Vectorizer and TF-IDF techniques for model training.

## Findings
The stock market generally outperformed all predictive models developed, suggesting robust market efficiency. However, ensemble methods and advanced ML algorithms showed potential in certain scenarios, albeit with increased investment risk.

## Conclusion
Our study supports the EMH, indicating that market prices generally reflect all available information. Despite this, AI and NLP provide valuable tools for analyzing financial data, offering insights into market sentiment and potential trends.

## Acknowledgements
This project is part of QMST 4320 Data Analytics class at the McCoy College of Business, Texas State University, under the supervision of Dr. Tahir Ekin.

## References
- Ke, Z. T., Kelly, B., & Xiu, D. (2019). Predicting Returns with Text Data. Becker Friedman Institute for Economics at The University of Chicago.
- [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
- [U.S. Securities and Exchange Commission - EDGAR Company Filings](https://www.sec.gov/edgar/searchedgar/companysearch.html)

## Notes
Link to parsed 10k (conclusion of step 1): https://drive.google.com/file/d/1HjMQ32Yu7fFwiT9b92vE60n3WDfikUsa/view?usp=sharing
