import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.spatial import distance
import numpy as np


def preprocessing(df, encoders=None, scalers=None):
    encode = OrdinalEncoder()
    scale = StandardScaler()
    scale_col = ['recency', 'frequency']
    encode_col = ['sex', 'location', 'job', 'ageGroup', 'is_donated', 'is_dormant']
    # df_scale = df[scale_col].reshape(1,-1)
    df_scaled = pd.DataFrame(scale.fit_transform(df[scale_col]))
    # df_scaled = pd.DataFrame(scale.fit_transform(df_scale))
    df_scaled.columns = scale_col
    df_encoded = encode.fit_transform(df[encode_col])
    df_encoded = pd.DataFrame(df_encoded)
    df_encoded.columns = encode_col
    df_prepro = pd.concat([df_scaled, df_encoded], axis=1)
    df_prepro = pd.DataFrame(df_prepro)
    return df_prepro


def recommend(target_uuid, df):
    df['recency'].fillna(df['recency'].mean(), inplace=True)
    df['frequency'].fillna(df['frequency'].mean(), inplace=True)
    df = df.fillna(method='ffill')

    userData = df.loc[df['user_id'] == target_uuid]
    # preprocessed_userData = preprocessing(userData, encoders=None,  scalers=None,)
    user_label = userData['Cluster_labels'].reset_index(drop=True)

    cluster_members = df.loc[df['Cluster_labels'].isin(user_label)]

    preprocessed_cluster_members = preprocessing(cluster_members, encoders=None, scalers=None)

    preprocessed_cluster_members.set_index(cluster_members['user_id'], inplace=True)

    preprocessed_cluster_members = preprocessed_cluster_members.copy()
    if len(preprocessed_cluster_members) <= 1:
        recom_id = []
        recom_weight = []
    else:
        dist_array = []
        user = preprocessed_cluster_members.loc[target_uuid, :]
        user = user.drop_duplicates()
        for row in preprocessed_cluster_members.itertuples():
            row = row[1:]
            dist = distance.euclidean(user, row)
            dist_array.append(dist)
        dist_array = pd.Series(dist_array)
        dist_array = dist_array.set_axis(df['user_id'])
        sorted_dist = dist_array.sort_values()
        if len(sorted_dist < 10):
            if len(sorted_dist) == 0:
                recom_weight = []
                recom_id = []
            else:
                recom_weight = sorted_dist
                recom_uuid = recom_weight.index
                recom_id = df['user_id'].loc[df['user_id'].isin(recom_uuid)]

        else:
            recom_weight = sorted_dist[1:11]
            recom_uuid = recom_weight.index
            recom_id = df['user_id'].loc[df['user_id'].isin(recom_uuid)]
    return recom_id, recom_weight

# def recommend(target_uuid,df):
#     df['recency'].fillna(df['recency'].mean(), inplace=True)
#     df['frequency'].fillna(df['frequency'].mean(), inplace=True)
#     df = df.fillna(method='ffill')
#
#     userData = df.loc[df['user_id'] == target_uuid]
#     #preprocessed_userData = preprocessing(userData, encoders=None,  scalers=None,)
#     user_label = userData['Cluster_labels'].reset_index(drop=True)
#
#     cluster_members = df.loc[df['Cluster_labels'].isin(user_label)]
#     preprocessed_cluster_members = preprocessing(cluster_members, encoders=None,scalers=None)
#     preprocessed_cluster_members.set_index(cluster_members['user_id'], inplace=True)
#
#     user = preprocessed_cluster_members.loc[target_uuid, :]
#
#     dist_array = []
#     for row in preprocessed_cluster_members.itertuples():
#         row = row[1:]
#         dist = distance.euclidean(user, row)
#         dist_array.append(dist)
#     dist_array = pd.Series(dist_array)
#     dist_array = dist_array.set_axis(cluster_members['user_id'])
#
#     sorted_dist = dist_array.sort_values()
#     if len(sorted_dist < 10):
#         if len(sorted_dist) == 0:
#             recom_weight = []
#             recom_id = []
#         else:
#             recom_weight = sorted_dist
#             recom_uuid = recom_weight.index
#             recom_id = df['user_id'].loc[df['user_id'].isin(recom_uuid)]
#     else:
#         recom_weight = sorted_dist[1:11]
#         recom_uuid = recom_weight.index
#         recom_id = df['user_id'].loc[df['user_id'].isin(recom_uuid)]
#     return recom_id, recom_weight


# 유사도 추가 후
# dist_array = []
# for row in preprocessed_cluster_members.itertuples():
#     row = row[1:]
#     dist = distance.euclidean(user, row)
#     dist_array.append(dist)
# dist_array = pd.Series(dist_array)
# dist_array = dist_array.set_axis(cluster_members['user_id'])
#
# sorted_dist = dist_array.sort_values()
# recom_list = sorted_dist[1:11]
# recom_uuid = recom_list.index
#
# recom_result = cluster_members['user_id'].loc[cluster_members['user_id'].isin(recom_uuid)]
# return recom_result
