# StockVision
Rudimentary stock visualization and machine-learning powered prediction algorithm.

This script offers tools to visualize, analyze, and even try to predict future prices of your chosen company's stocks based off machine-learning regression algorithms.

## Demonstration

To demonstrate the script's capabilities, I will provide an example of a case study that can be done using this script.

Let's say, for example, I would like to see how the tech company <b>NVIDIA</b> has been performing over the past 5 years.

The stock ticker for NVIDIA is "NVDA" and I will set the time range from 2015-01-01 to 2020-08-31.

### Stock Price and Moving Average Visualization:
Using the script's function, `display_moving_average`, we get the following output:

<img src="https://i.imgur.com/LD0uBfe.png">

Looking at this data, we can see that the moving average has generally been on an upward trend since 2015. Therefore, we can conclude that the company has been performing quite well over the past 5 years.

<br>

### Return Deviation Analysis:
Now, let's say that I'm interested in trying to determine the general risk and return of this stock.

Using the script's function, `display_return_deviation`, we get the following output:

<img src="https://i.imgur.com/XgAh9MA.png">

Generally, the "ideal" stock should have high returns (be positive in this graph) and be stable (not have big up and down fluctuations). If you are someone who is rather risk averse, you may want to avoid this stock after seeing the nearly 20% drops in late 2018 and early 2020. The return deviation is helpful for this type of analysis.

<br>

### Competitor Correlation Analysis:
Now, let's say that I would like to find out how NVIDIA fairs compared to its competitor AMD, and if there is a correlation between the performance of the two companies.

Using the script's function, `display_correlation_analysis`, we get the following output:

<img src="https://i.imgur.com/DcGMTKZ.png">

As we can see, despite a few outliers, NVIDIA generally performs better (has a better return rate) when AMD is also performing better, indicating a slight positive correlation between the two. This makes sense as these are both companies that have a heavy influence in the GPU industry. The positive correlation most likely is a by-product of the constant competition and new hardware releases these companies have been putting out in the past few years.

<br>

### Machine-Learning Based Stock Prediction:
(Disclaimer: by no means is this a tool that can be realistically applied to the market, and it should not be used as a reference when trading on the stock market. This was purely made with the intention to try my hand in machine learning and apply it to a financial application.)

Lastly, let's say I am curious to see how NVIDIA may perform 10 days in the future from August 31st, 2020. The script uses machine-learning based regression models (Linear Regression, Quadratic Regression and k-Nearest Neighbor) to train the prediction algorithm based off past data, and then make a prediction for how the stocks will fair in the future. 

Using the script's function, `predict_future_price`, let's see how it faired for NVIDIA:

<img src="https://i.imgur.com/zdliuu9.png">

The script gave me roughly a 93% confidence rating for this prediction. 

Looking at the actual graph of NVIDIA's stocks from September 1st to September 10th, we can see that it reaches a peak on Sep 2nd, but then is on a generally downward trend. 

<img src="https://i.imgur.com/8gu44uT.png" height="200" width="356">

Although the prediction's shape doesn't entirely mimic this trend, if we look closer at it, we see that the shape of the predicted line does quite a decent job of matching the general trend of the stock price from September 1st to September 10th. 

<img src="https://i.imgur.com/1cgUnCr.png" height="200" width="356">

It should be noted that the prediction data set was generated fully based off the data between January 1st, 2015 to August 31st, 2020 and did not have any data from September 1st to September 10th to base it off with. 

All in all, a pretty good prediction with all things considered.

<br>

Want to try it out yourself?
------------

### Requirements: 
- Python 2.7 or 3.3+

- [matplotlib](http://matplotlib.sourceforge.net)
- [pandas](http://pandas.pydata.org/)
- [numpy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org/stable/)

Once you meet the following requirements, just download the script and in the "__main__" function, input what you'd like for the following variables:
- company
- competitor
- start_string (the date in YYYY-MM-DD you'd like the script to start from)
- end_string (the date in YYYY-MM-DD you'd like the script to end from)
- forecast_duration (how many days into the future you'd like the script to predict the stock price for)

Then you can any of run the following methods:
- `display_moving_average` 
- `display_return_deviation`
- `display_correlation_analysis`
- `predict_future_price`
