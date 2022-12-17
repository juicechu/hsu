import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def mean(x):
    x

if __name__ == '__main__':
    df_crude_oil = pd.read_csv('Crude Oil Data.csv')
    df_crude_oil.info()
    df_nasdaq = pd.read_csv('NASDAQ Data.csv')
    df_nasdaq.info()
    df_crude_oil['Vol.'] = df_crude_oil['Vol.'].str.replace('K', '').astype(float)
    df_crude_oil['Vol.'].fillna(df_crude_oil['Vol.'].mean(), inplace=True)
    df_crude_oil['Date'] = pd.to_datetime(df_crude_oil['Date'])
    df_crude_oil.rename(columns={
        'Price': 'co_price',
        'Open': 'co_open',
        'High': 'co_high',
        'Low': 'co_low',
        'Vol.': 'co_volume',
        'Change %': 'co_change',
    }, inplace=True)
    df_crude_oil['co_change'] = df_crude_oil['co_change'].str.replace('%', '').astype(float)
    df_crude_oil.info()
    print(df_crude_oil.head())
    df_nasdaq['Date'] = pd.to_datetime(df_nasdaq['Date'])
    df_nasdaq.rename(columns={
        'Close': 'na_close',
        'Adj Close': 'na_adj_close',
        'Open': 'na_open',
        'High': 'na_high',
        'Low': 'na_low',
        'Volume': 'na_volume',
    }, inplace=True)
    print(df_nasdaq.head())
    df_data = pd.merge(df_crude_oil, df_nasdaq, how='inner', on='Date')
    df_data.info()
    print('The different between na_close and na_adj_close is', (df_data['na_close'] != df_data['na_adj_close']).sum())
    df_data.drop(columns=['na_adj_close'], inplace=True)
    df_data = df_data.sort_values(by=['Date'], ignore_index=True)
    df_data['na_change'] = (df_data['na_close'] - df_data['na_close'].shift(-1)) / df_data['na_close'].shift(-1) * 100
    df_data['na_change'] = df_data['na_change'].round(2)
    df_data = df_data[:-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    co_change = df_data.pop('co_change')
    na_change = df_data.pop('na_change')
    df_data.drop(columns=['Date'], inplace=True)
    data_scaled = scaler.fit_transform(df_data)
    df_X = pd.DataFrame(data_scaled, columns=df_data.columns)
    df_cleaned = df_X.copy()
    df_cleaned['co_change'] = co_change
    df_cleaned['na_change'] = na_change
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    print(df_cleaned.head())


    #predict Crude Oil
    df_data = pd.read_csv('cleaned_data.csv')
    df_Y = df_data.pop('co_change')
    df_data.pop('na_change')
    df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_data, df_Y,  test_size = 0.3, random_state=12345)
    neighbors = {'n_neighbors': list(range(2, 50))}
    knn = KNeighborsRegressor()
    model = GridSearchCV(knn, neighbors, cv=10)
    model.fit(df_X_train, df_Y_train)
    print('KNN best params: ', model.best_params_)
    pred = model.predict(df_X_test)
    error = sqrt(mean_squared_error(df_Y_test, pred)) #rmse
    print('KNN RMSE: ', error)
    #df_data = pd.read_csv('cleaned_data.csv')
    pred = model.predict(df_data) #predict All
    df_data['co_change'] = co_change
    df_data['predicted_co_change'] = pred
    print(df_data.tail())
    df_data.to_csv('curde_oil_predicted.csv', index=False)


    #predict NASDAQ index
    df_data = pd.read_csv('cleaned_data.csv')
    df_data.pop('co_change')
    df_Y = df_data.pop('na_change')
    df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_data, df_Y,  test_size = 0.3, random_state=12345)
    neighbors = {'n_neighbors': list(range(2, 50))}
    knn = KNeighborsRegressor()
    model = GridSearchCV(knn, neighbors, cv=10)
    model.fit(df_X_train, df_Y_train)
    print('KNN best params: ', model.best_params_)
    pred = model.predict(df_X_test)
    error = sqrt(mean_squared_error(df_Y_test, pred)) #rmse
    print('KNN RMSE: ', error)
    #df_data = pd.read_csv('cleaned_data.csv')
    pred = model.predict(df_data) #predict All
    df_data['na_change'] = na_change
    df_data['predicted_na_change'] = pred
    print(df_data.tail())
    df_data.to_csv('NASDAQ_predicted.csv', index=False)

