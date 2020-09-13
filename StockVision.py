# Financial Data Processing Imports
import math
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

# Data Visualization Imports
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

# Machine Learning Imports
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Sets the range of time for the stock price data to be pulled
# Input dates as "YYYY-MM-DD" for start_string and end_string
def set_data_range(start_string, end_string):
    start = start_string.split('-')
    end = end_string.split('-')
    stock_start_date = datetime.datetime(int(start[0]), int(start[1]), int(start[2]))
    stock_end_date = datetime.datetime(int(end[0]), int(end[1]), int(end[2]))

    return stock_start_date, stock_end_date


# Sets which company to track for a specific date range, and sets which financial source to extract data from
def set_tracked_company(company, start_date, end_date, financial_source='yahoo'):
    df = web.DataReader(company, financial_source, start_date, end_date)
    df.tail()
    return df


# Plotting the stock prices on a graph, as well as with the Moving Average (MA) trendline for analysis purposes.
# Note that for the data points, I will be using each day's closing price (final price) as a value.
# Moving Average is used to display the general trend of a company's stocks (upward, downward, stagnates, etc...)
def display_moving_average(dataframe, company, start, end):
    title = "[" + company + "] - " + "Stock Price Plot and Moving Average\n" + start + " to " + end

    close_price = dataframe['Adj Close']
    moving_average = close_price.rolling(window=100).mean()
    mpl.rc('figure', figsize=(15, 10))
    style.use('ggplot')
    close_price.plot(label=company)
    moving_average.plot(label='Moving Average (' + company + ')')

    plt.title(title)
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.show()


# Plots the return deviation for the stock - helps determine risk and return
def display_return_deviation(dataframe, company, start, end):
    title = "[" + company + "] - " + "Return Deviation Plot\n" + start + " to " + end

    close_price = dataframe['Adj Close']
    ret_dev= (close_price / close_price.shift(1)) - 1
    mpl.rc('figure', figsize=(15, 10))
    ret_dev.plot(label='return')

    plt.title(title)
    plt.ylabel("Return Rate")
    plt.show()


# Correlation analysis helps determine how the selected company's returns are affected by competitors
def display_correlation_analysis(company, competitor, start, end, start_string, end_string, financial_source='yahoo'):
    title = "[" + company + "] and [" + competitor + "] Correlation Analysis\n" + start_string + " to " + end_string

    competitor_df = web.DataReader([company, competitor], financial_source, start=start, end=end)['Adj Close']
    returns_comp = competitor_df.pct_change()
    mpl.rc('figure', figsize=(15, 10))
    plt.scatter(returns_comp[company], returns_comp[competitor])
    plt.title(title)
    plt.xlabel(company + " Return Rate")
    plt.ylabel(competitor + " Return Rate")
    plt.show()

    #correlation = returns_comp.corr()
    #print (correlation)


# Using machine learning models to predict future stock prices based on current, given data
def predict_future_price(df, forecast_out, prediction_model, end_string):

    # -- Pre-processing and Training --
    df = df.loc[:, ['Adj Close']]  # Using Adj. Close values for our prediction

    # Creating a new column in dataframe (called 'prediction') which will be the label (a.k.a. output)
    # Filling label column with output data to be trained upon (but shifted 'forecast_duration' days up)
    # This allows for the missing units due to the up shift to be filled with the predicted scores (the result)
    df['prediction'] = df[['Adj Close']].shift(-forecast_out)

    # X represents the array of Adj. Close values, so prediction column must be dropped
    X = np.array(df.drop(['prediction'], 1))
    X = preprocessing.scale(X)  # Normalize the data for models

    # Moving the NaN values to the forecast array since the true forecast data is yet to be populated
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out] # Removing the NaN values from X

    # Defining Output (equate it to the prediction dataframe column, and remove the days that don't have any pricing
    # data yet. These are the days that will be filled by the forecast
    Y = np.array(df['prediction'])
    Y = Y[:-forecast_out]

    # -- Model Generation, Training and Evaluation --
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    # Linear Regression Model
    clf_lin_reg = LinearRegression()
    clf_lin_reg.fit(X_train, Y_train)

    # Quadratic Regression (2) Model
    clf_quad_2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clf_quad_2.fit(X_train, Y_train)

    # Quadratic Regression (3) Model:
    clf_quad_3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clf_quad_3.fit(X_train, Y_train)

    # k-Nearest Neighbor Regression Model:
    clf_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    clf_knn.fit(X_train, Y_train)

    # Confidence Score (Coefficient of Determination - compares predicted price with actual price)
    confidence_lin_reg = clf_lin_reg.score(X_test, Y_test)
    confidence_quad_2 = clf_quad_2.score(X_test, Y_test)
    confidence_quad_3 = clf_quad_3.score(X_test, Y_test)
    confidence_knn = clf_knn.score(X_test, Y_test)

    if (prediction_model == "Linear Regression"):
        confidence_lin_reg = clf_lin_reg.score(X_test, Y_test)
        forecast = clf_lin_reg.predict(X_forecast)
    elif (prediction_model == "Quadratic Regression 1"):
        confidence_quad_2 = clf_quad_2.score(X_test, Y_test)
        forecast = clf_quad_2.predict(X_forecast)
    elif (prediction_model == "Quadratic Regression 2"):
        confidence_quad_3 = clf_quad_3.score(X_test, Y_test)
        forecast = clf_quad_3.predict(X_forecast)
    elif (prediction_model == "k-Nearest Neighbor"):
        confidence_knn = clf_knn.score(X_test, Y_test)
        forecast = clf_knn.predict(X_forecast)


    # Displaying Data on Graph
    last_date = df.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        df.loc[next_date] = [np.nan for _ in
                             range(len(df.columns) - 1)] + [i]

    df['Adj Close'].tail(500).plot()
    df['prediction'].tail(forecast_out).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend(['Closing Price', 'Predicted Closing Price'])
    title = "[" + company + "] -" + " Projected Stock Price for: Next " + str(forecast_out) + " Days beyond " + end_string
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    company = 'NVDA'
    competitor = 'AMD'
    start_string = "2015-01-01"
    end_string = "2020-08-31"
    forecast_duration = 10  # In terms of days
    prediction_models = ["Linear Regression", "Quadratic Regression 1", "Quadratic Regression 2", "k-Nearest Neighbor"]

    start, end = set_data_range(start_string, end_string)
    df = set_tracked_company(company, start, end)
    display_moving_average(df, company, start_string, end_string)
    display_return_deviation(df, company, start_string, end_string)
    display_correlation_analysis(company, competitor, start, end, start_string, end_string)

    predict_future_price(df, forecast_duration, prediction_models[3], end_string)
