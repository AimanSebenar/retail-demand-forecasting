from data_loader_eda import get_data

df, df_test = get_data()


def engineer_features(df):
    df=df.copy()
    df['Year']=df['Date'].dt.year
    df['Month']=df['Date'].dt.month
    df['Week']=df['Date'].dt.isocalendar().week
    df['DayOfWeek']=df['Date'].dt.dayofweek
    df['IsWeekend']=df['DayOfWeek'].isin([5,6]).astype(int)
    df['DayOfMonth']=df['Date'].dt.day

    #Cyclical encoding (avoid week discontinuity)
    import numpy as np
    df['dow_sin'] = np.sin(2*np.pi*df['DayOfWeek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['DayOfWeek']/7)


    #Lag Features
    df = df.sort_values(['Store', 'Date'])
    for lag in [7, 14, 21, 28]:
        df[f'lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

    #Rolling stats
    df['rolling_7d_mean'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df['rolling_28d_mean'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    df['StateHoliday'] =df['StateHoliday'].replace({'0':0, 'a':1, 'b':2,'c':3}).astype(int)
    
    return df.dropna()

train = engineer_features(df)
X_train = train.drop(columns={'Sales'})
y_train = train['Sales']

test = engineer_features(df_test)
X_val = test.drop(columns={'Sales'})
y_val = test['Sales']

def clean_train_data():
    return X_train, y_train

def clean_test_data():
    return X_val, y_val
