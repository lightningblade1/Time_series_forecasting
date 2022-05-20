import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import LeakyReLU
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from pickle import dump

print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))
# Importing files
oil = pd.read_csv('store-sales-time-series-forecasting/oil.csv',
                  parse_dates=['date'], infer_datetime_format=True,
                  index_col='date').to_period('D')
holiday = pd.read_csv('store-sales-time-series-forecasting/holidays_events.csv',
                      parse_dates=['date'], infer_datetime_format=True, dtype={'locale': 'category'},
                      index_col='date').to_period('D')

stores = pd.read_csv("store-sales-time-series-forecasting/stores.csv", dtype={'store_nbr': 'category'})
train = pd.read_csv('store-sales-time-series-forecasting/train.csv',
                    parse_dates=['date'], infer_datetime_format=True,
                    dtype={'store_nbr': 'category',
                           'family': 'category'}, usecols=['date', 'store_nbr', 'family', 'sales'])
test = pd.read_csv('store-sales-time-series-forecasting/test.csv',
                   parse_dates=['date'], infer_datetime_format=True,
                   dtype={'store_nbr': 'category',
                          'family': 'category'}, usecols=['date', 'store_nbr', 'family'])
# dtype category saves the values as categories as enums it helps in sorting and custom order
# #- in big dataframes, categories takes less memory than strings
# - it makes it easy to check if you have an unexpected value in a df you've just imported (any value not in the listed categories will show up as NaN)
# - situational, but pretty cool if you need it: you can have a custom sort order
calendar = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31')).to_period(
    'D')  # making this is important because imported data might have missing dates and making it to period makes it datetime format and allows us to getinfo like day of week etc.

transactions = pd.read_csv("store-sales-time-series-forecasting/transactions.csv")
tree = train.store_nbr.unique()
tree = tree.sort_values()
new_df = pd.DataFrame(columns=["column"])

stores["mean"] = 0
for i in tree:
    current = pd.DataFrame(train[train.store_nbr == i])
    current = current.sales.mean()
    stores["mean"].loc[(stores.store_nbr == i)] = current

test['date'] = test.date.dt.to_period('D')
train = train.set_index(['date', 'store_nbr',
                         'family']).sort_index()  # set index of  data set to date, store number and family respectively
test = test.set_index(['date', 'store_nbr',
                       'family']).sort_index()

# OIL

oil.index = oil.index.to_timestamp()
oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="polynomial", limit_direction="both", order=7)
oil.index = oil.index.to_period('D')
oil['avg_oil'] = oil['dcoilwtico'].rolling(7).mean()  # refer to Machine learning for timeseries forecasting with python
calendar = calendar.join(oil.avg_oil)

calendar['avg_oil'].fillna(method='ffill', inplace=True)

calendar.dropna(inplace=True)
oil.dropna(inplace=True)

# plt.figure()
# sns.lineplot(data=oil.to_timestamp(freq="D"),
#             palette="flare")  # https://www.geeksforgeeks.org/python-pandas-period-to_timestamp/
# You can see that oil price is only high at 2013 to 2014, however in 2015 it's starting to go down.
# So, because we only predict 16 data points we can data from 2015
# plot_acf(calendar.avg_oil, lags=100)  # make sure NA values are dropped before plotting.
# # we need to check for multicolinearity with PACF
# plot_pacf(calendar.avg_oil, lags=12)  # Lagplot oil price (Feature Engineering)
# plt.show()
# # the plot was highly correlated so we can take last 5 lags
n_lags = 3
for l in range(1, 6):
    calendar['oil_lagl' + str(l)] = calendar.avg_oil.shift(l)
for l in range(-5, 0):
    calendar['oil_lagl' + str(l)] = calendar.avg_oil.shift(l)
calendar.dropna(inplace=True)

# Holiday
holiday = holiday[
    holiday.locale == 'National']  # filter national holidays because others might not have significant on sales
holiday = holiday[~holiday.index.duplicated(keep='first')]  # removing duplicated dates and "~" is not operator
train.head()
calendar = calendar.join(holiday)

calendar['dofw'] = calendar.index.dayofweek  # Weekly day
calendar['wd'] = 1
calendar.wd[calendar['dofw'] > 4] = 0
calendar.wd[calendar.type == 'Work Day'] = 1  # If it's Work Day event then it's a workday
calendar.wd[calendar.type == 'Additional'] = 0  # Additional holidays are days added a regular calendar holiday,
calendar.wd[(calendar.type == 'Holiday') & (calendar.transferred == False)] = 0
calendar.wd[(calendar.type == 'Holiday') & (calendar.transferred == True)] = 1
calendar = pd.get_dummies(calendar, columns=['dofw', 'type'])  # One-hot encoding
calendar.drop(['locale', 'locale_name', 'description', 'transferred'], axis=1, inplace=True)  # Unused columns
calendar['wd_lag1'] = calendar.wd.shift(1)
calendar['wd_fore1'] = calendar.wd.shift(-1)
calendar.dropna(inplace=True)

family = set(c[2] for c in train.index)  # this returns unique values for all the possible families
# for f in family:
#     ax = y.loc(axis=1)['sales', :, f].plot(legend=None)
#     ax.set_title(f)
# above code shows plot for each family
# ax = y.loc(axis = 1)['sales', :].plot(legend = None)
# above code shows plot for each family
# looking at the plots for different families it can be seen that there is trend present within the datapoint so i will take data from month 5 to month 8 as training data as we need to predict only 15 days into the future


# Sales
# plot_pacf(train.loc['2015':'2017'], lags=20)  # Lagplot oil price (Feature Engineering)
# plt.show()


# Preparing data

calendar = calendar.loc['2015':'2017']
train = train.unstack(['store_nbr', 'family']).loc['2015':'2017']  # there is a drop in oil prices after 2015
train.index = train.index.to_period('D')
columns = train.columns
# Features
for l in range(1, 10):
    for i, j, k in columns:
        train[str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l)] = train[i, j, k].shift(l)
calendar = calendar.join(train)

calendar_column = calendar.columns

#scaler = MinMaxScaler()
# calendar = pd.DataFrame(scaler.fit_transform(calendar.values), columns=calendar.columns,
#                         index=calendar.index)  # this line of code gives scaled values as df otherwise it would give np array
calendar = pd.DataFrame(calendar.values, columns=calendar.columns,
                        index=calendar.index)
calendar.dropna(inplace=True)

# calculate urself tommorow  if values are correct thrugh calculator
Target = calendar[columns]

Target = Target["2015-01-10":]
# for l in range(1, 16):
#     for i, j, k in columns:
#         Target[str(i) + "_" + str(j) + "_" + str(k) + "_" + str(-l)] = Target[i, j, k].shift(
#             -l)  # this loop will make sure next 16 days are predicted and for this we need to remove some of the rows from the bottom calender because other wise the data wouldnt match
Target.dropna(inplace=True)
Target.to_csv('Target.csv')
calendar.to_csv('calendar.csv')
# save the scaler
#dump(scaler, open('scaler.pkl', 'wb'))