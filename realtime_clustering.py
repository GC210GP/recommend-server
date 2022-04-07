from sklearn.metrics import silhouette_score
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.stats import stats

def preprocessing(df,encoders=None,  scalers=None):
    scale_col = ['recency', 'frequency']
    encode_col = ['sex', 'location', 'job', 'ageGroup', 'is_donated', 'is_dormant']
    encode = OrdinalEncoder()
    scale = StandardScaler()

    #DELETE OUTLIER

    df_scaled = pd.DataFrame(scale.fit_transform(df[scale_col]))
    df_scaled.columns = scale_col

    df_encoded = encode.fit_transform(df[encode_col])
    df_encoded = pd.DataFrame(df_encoded)
    df_encoded.columns = encode_col

    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
    df_prepro = pd.DataFrame(df_prepro)

    return df_prepro

def outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3,True,False)
    return pd.Series(idx_outliers, index=col.index)


def get_category(age):
    cat = ""
    if age <= -1: cat = "Unknown"
    elif age <= 5: cat = "Baby"
    elif age <= 12: cat="Child"
    elif age <= 18: cat = "Teenager"
    elif age <= 25: cat="Student"
    elif age <= 35: cat="young Adult"
    elif age <= 60: cat = "Adult"
    else : cat = "Elderly"
    return cat
#########################################################################################################
from sklearn.cluster import DBSCAN

def clustering(df):
    model = joblib.load('./dbscanModel.pkl')
    df = preprocessing(df, encoders=None, scalers=None)
    labels = model.fit_predict(df)
    return labels

def to_csv(df):
    df = df[['user_id', 'user_name', 'recency', 'sex', 'blood_type', 'birthdate', 'location', 'job', 'frequency',
             'is_donated', 'is_dormant']]
    df['recency'].fillna(df['recency'].mean(), inplace=True)
    df['frequency'].fillna(df['frequency'].mean(), inplace=True)
    df['ageGroup'] = df['birthdate'].apply(lambda x: get_category(x))
    df = df.fillna(method='ffill')
    clustered_df = pd.DataFrame()
    if (df['blood_type'] == 'PLUS_A').any():
        df_PA = df.loc[df['blood_type'] == 'PLUS_A']
        preprocessed_PA = preprocessing(df_PA, encoders=None, scalers=None)
        df_PA['Cluster_labels'] = clustering(preprocessed_PA)
        clustered_df = pd.concat([clustered_df,df_PA], axis=0, ignore_index=True)

    if (df['blood_type'] == 'PLUS_B').any():
        df_PB = df.loc[df['blood_type'] == 'PLUS_A']
        preprocessed_PB = preprocessing(df_PB, encoders=None, scalers=None)
        df_PB['Cluster_labels'] = clustering(preprocessed_PB)
        clustered_df = pd.concat([clustered_df, df_PB], axis=0, ignore_index=True)

    if (df['blood_type'] == 'PLUS_AB').any():
        df_PAB = df.loc[df['blood_type'] == 'PLUS_AB']
        preprocessed_PAB = preprocessing(df_PAB, encoders=None, scalers=None)
        df_PAB['Cluster_labels'] = clustering(preprocessed_PAB)
        clustered_df = pd.concat([clustered_df, df_PAB], axis=0, ignore_index=True)

    if (df['blood_type'] == 'PLUS_O').any():
        df_PO = df.loc[df['blood_type'] == 'PLUS_O']
        preprocessed_PO = preprocessing(df_PO, encoders=None, scalers=None)
        df_PO['Cluster_labels'] = clustering(preprocessed_PO)
        clustered_df = pd.concat([clustered_df, df_PO], axis=0, ignore_index=True)

    if (df['blood_type'] == 'MINUS_A').any():
        df_MA = df.loc[df['blood_type'] == 'MINUS_A']
        preprocessed_MA = preprocessing(df_MA, encoders=None, scalers=None)
        df_MA['Cluster_labels'] = clustering(preprocessed_MA)
        clustered_df = pd.concat([clustered_df, df_MA], axis=0, ignore_index=True)

    if (df['blood_type'] == 'MINUS_B').any():
        df_MB = df.loc[df['blood_type'] == 'MINUS_B']
        preprocessed_MB = preprocessing(df_MB, encoders=None, scalers=None)
        df_MB['Cluster_labels'] = clustering(preprocessed_MB)
        clustered_df = pd.concat([clustered_df, df_MB], axis=0, ignore_index=True)

    if (df['blood_type'] == 'MINUS_AB').any():
        df_MAB = df.loc[df['blood_type'] == 'MINUS_AB']
        preprocessed_MAB = preprocessing(df_MAB, encoders=None, scalers=None)
        df_MAB['Cluster_labels'] = clustering(preprocessed_MAB)
        clustered_df = pd.concat([clustered_df, df_MAB], axis=0, ignore_index=True)

    if (df['blood_type'] == 'MINUS_O').any():
        df_MO = df.loc[df['blood_type'] == 'MINUS_O']
        preprocessed_MO = preprocessing(df_MO, encoders=None, scalers=None)
        df_MO['Cluster_labels'] = clustering(preprocessed_MO)
        clustered_df = pd.concat([clustered_df, df_MO], axis=0, ignore_index=True)

    return clustered_df




