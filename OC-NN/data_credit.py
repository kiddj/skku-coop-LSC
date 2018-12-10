import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dataPath = "data/creditcard.csv"

def load_credit():
    credit = pd.read_csv(dataPath)
    credit = credit.drop(['Time'], axis=1)
    credit['Amount'] = \
        StandardScaler().fit_transform(credit['Amount'].values.reshape(-1,1))

    df_normal = credit[credit['Class'] == 0]
    df_anomaly = credit[credit['Class'] == 1]
    del df_normal['Class']
    del df_anomaly['Class']

    # DataFrame to Numpy Array
    normal = df_normal.values
    anomaly = df_anomaly.values

    return [normal, anomaly]

def load_credit_old():
    # Load CreditCard Data
    credit = pd.read_csv(dataPath)
    # Delete Time column
    del credit['Time']

    #Delete Amount?
    del credit['Amount']

    # Normal / Anomaly
    df_normal = credit[credit['Class'] == 0]
    df_anomaly = credit[credit['Class'] == 1]
    del df_normal['Class']
    del df_anomaly['Class']

    # DataFrame to Numpy Array
    normal = df_normal.values
    anomaly = df_anomaly.values

    return [normal, anomaly]

if __name__ == '__main__':
    load_credit()