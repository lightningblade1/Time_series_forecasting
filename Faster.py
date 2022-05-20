import ast

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import LeakyReLU
from pickle import load

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
test1 = pd.read_csv('store-sales-time-series-forecasting/test.csv',
                    parse_dates=['date'], infer_datetime_format=True,
                    dtype={'store_nbr': 'category',
                           'family': 'category'}, usecols=['id', 'date', 'store_nbr', 'family', 'onpromotion'])
Target = pd.read_csv('Target.csv', parse_dates=['Unnamed: 0'], infer_datetime_format=True, index_col='Unnamed: 0')
Target.index = pd.to_datetime(Target.index).to_period('D')
result_column = Target.columns
calendar = pd.read_csv('calendar.csv',
                       parse_dates=['Unnamed: 0'], infer_datetime_format=True,
                       index_col='Unnamed: 0')
calendar.index = pd.to_datetime(calendar.index).to_period('D')
print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))

test = calendar["2017-07-01":].to_numpy().tolist()

calendar = calendar.to_numpy().tolist()
test_target = Target.iloc[:46]
test_target = test_target.to_numpy().tolist()
Target = Target.to_numpy().tolist()
train_ts = TimeseriesGenerator(calendar, Target, length=30, sampling_rate=1,
                               batch_size=1)  # ,calender is the dataset on which the prediction is done,Target is the result that is expected
g = train_ts[0]
test_ts = TimeseriesGenerator(test,
                              test_target,
                              length=30,
                              sampling_rate=1, batch_size=1)
print(type(test_ts[0]))
# batch size divides training and expected results to corresponding batches.TimeseriesGenerator works like a sliding window
# think of sampling as frequency and length as window. if you have length 10 it will take 10 rows(timestamps) from calender and the model will try to predict the relevant values of Target at 11 row (timestamp).
# Now if you have length =10 and sampling_rate=2 for examples lets suppose you have these values [1,2,3,4,5,6,7,8,9,10] now sampling_rate=2 would do this to the data [1,3,5,7,9]
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1000, batch_input_shape=(1,30, 17847),
                               return_sequences=True))  # https://www.youtube.com/watch?v=B66760rvHA8&t=129s
# # https://www.kaggle.com/code/kmkarakaya/lstm-output-types-return-sequences-state/notebook
model.add(LeakyReLU(alpha=0.5))
#model.add(tf.keras.layers.Dropout(0.3)) #if you make return_sequences in next statement false you need this statement to align the output from last layer to input of next layer
model.add(tf.keras.layers.LSTM(1000, return_sequences=True))
model.add(LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(500, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1782))
for layer in model.layers:
    print(layer.output_shape)
print(model.summary())
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError()])
history = model.fit(train_ts, epochs=15, shuffle=False)
model.save('saved-model/')

# model = tf.keras.models.load_model('saved-model')
print(model.summary())
result = model.predict(test_ts)  # fix output shape of output layer,
# columns = Target.columns
# look into changing dimension of target array in test_ts
s = pd.date_range('2017-08-16', '2017-08-31')
result = pd.DataFrame(data=result, index=pd.date_range('2017-08-16', '2017-08-31'), columns=result_column).to_period(
    'D')
# result=result.stack(['store_nbr', 'family'])
result = result.stack()

result = result.reset_index()
result.level_1 = result.level_1.apply(ast.literal_eval)  # makes string to tuples
print(type(result['level_1'][0]))
result[['b1', 'store_nbr', 'family']] = result['level_1'].tolist()
result = result.join(test1[['id', 'onpromotion']])
result = result.drop(columns=['b1', 'level_1'])
result.rename(columns={'level_0': 'date', 0: 'sales'}, inplace=True)
result = result[["id", "date", 'store_nbr', 'family', "onpromotion", "sales"]]
# df['C'] = df['B'].apply(lambda x: [y[0] for y in x])
result.to_csv('result.csv')
#print(train_ts.iloc[-1])
