import pandas as pd
from feature_engineering import rsi,atr
import math
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import warnings
warnings.filterwarnings('ignore')

def downloadData(ticker, startDate):
    df = yf.download(ticker, start=startDate,auto_adjust = True)
    return df

def plotTicker(df):
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'], label='Close Price history')
    plt.xlabel("Date")
    plt.ylabel("Price");

def preprocess(df, feature_list):
    global data

    if 'Volume' in feature_list:
        data = df.filter(['Close', 'Volume'])
    else:
        data = df.filter(['Close'])

    if 'RSI' in feature_list:
        date_index = data.index
        data.index = np.arange(0, df.shape[0])
        temp = data.copy()
        data['rsi'] = rsi(temp, 14).filter(['rsi'])
        data.index = date_index
        data = data[14:]

    if 'ATR' in feature_list:
        atr_values = atr(df)
        data['atr'] = atr_values

    global dataset
    global training_data_len
    dataset = data.values
    
    #computing training data length -> 80% of dataset
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the all of the data to be values between 0 and 1
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    global y_scaler
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit_transform(dataset[:, 0].reshape(-1, 1))

    global scaled_data
    scaled_data = feature_scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, :])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays to be used in training the LSTM model.
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))

    print('\nOverview of data:')
    print(data.head())
    print('\n')

    print('Overview of scaled data:')
    print(scaled_data[0:4, :])
    print('\n')

    return x_train, y_train


def train_model(x_train, y_train, num_features):
    print('Training model...')
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], num_features)))
    model.add(LSTM(units=50))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)
    
    return model


def test_data():
    # Create the Test data set
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:, 0]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, :])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))

    return x_test, y_test


def rmse(model, x_test, y_test):
    # Getting the models predicted price values
    predictions = model.predict(x_test)

    # Undo scaling
    predictions = y_scaler.inverse_transform(predictions)

    y_test = y_test.reshape(-1, 1)

    # Calculate the value of RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f'Root Mean Squared Error = {rmse}')
    return predictions

def plot_predict(predictions):
    #Plot/Create the data for the graph
    train = data[:training_data_len]
    global valid
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def print_pred():
    print(valid)


def run_LSTM(df, feature_list):
    global num_features
    num_features = len(feature_list)+1
    x_train, y_train = preprocess(df, feature_list)
    model = train_model(x_train, y_train, num_features)
    x_test, y_test = test_data()
    pred = rmse(model, x_test, y_test)
    plot_predict(pred)
    #print_pred()