import numpy as np
#rsi function used from https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
# original authour: Stef
def average_gainloss(series, window_size, average):
    a = (window_size - 1) / window_size
    ak = a ** np.arange(len(series) - 1, -1, -1)
    return np.append(average, np.cumsum(ak * series) / ak / window_size + average * a ** np.arange(1, len(series) + 1))


def rsi(df, window_size=14):
    df['change'] = df['Close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df.loc[window_size:, 'avg_gain'] = average_gainloss(df.gain[window_size + 1:].values, window_size,
                                                        df.loc[:window_size, 'gain'].mean())
    df.loc[window_size:, 'avg_loss'] = average_gainloss(df.loss[window_size + 1:].values, window_size,
                                                        df.loc[:window_size, 'loss'].mean())
    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi'] = 100 - (100 / (1 + df.rs))

    return df

#atr function used from https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python
# original auther: Andrew Olson
def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, n=14):
    data = df.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr