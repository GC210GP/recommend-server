import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.spatial import distance
import numpy as np
import recommendation
#############################################################################################
#  target_uuid and good_list's value is [uuid]
#  target_uuid = user who get recommendation
#  good_uuid = list of users who got <good> from 'target user'



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

#

def recommend(target_uuid, df):
    df['recency'].fillna(df['recency'].mean(), inplace=True)
    df['frequency'].fillna(df['frequency'].mean(), inplace=True)
    df = df.fillna(method='ffill')
    preprocessed_cluster_members = preprocessing(df, encoders=None, scalers=None)




    preprocessed_cluster_members.set_index(df['user_id'], inplace=True)




    preprocessed_cluster_members = preprocessed_cluster_members.copy()




    if len(preprocessed_cluster_members) <= 1:
        recom_id = []
        recom_weight = []
    else:
        dist_array = []
        user = preprocessed_cluster_members.loc[target_uuid, :]
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


# def recommend(target_uuid, df):
#     df['recency'].fillna(df['recency'].mean(), inplace=True)
#     df['frequency'].fillna(df['frequency'].mean(), inplace=True)
#     df = df.fillna(method='ffill')
#     preprocessed_cluster_members = preprocessing(df, encoders=None, scalers=None)
#     dist_array = []
#     user = preprocessed_cluster_members.loc[target_uuid, :]
#     for row in preprocessed_cluster_members.itertuples():
#         row = row[1:]
#         dist = distance.euclidean(user, row)
#         dist_array.append(dist)
#     dist_array = pd.Series(dist_array)
#     dist_array = dist_array.set_axis(df['user_id'])
#
#
#
#
#
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






    # ????????? ?????? ????????? ???
    # sorted_dist = dist_array.sort_values()
    # recom_list = sorted_dist[1:11]
    # recom_uuid = recom_list.index
    #
    # recom_result = df['user_id'].loc[df['user_id'].isin(recom_uuid)]
    # return recom_result

def concating(target_uuid, good_uuid, clustered_df):#99, 92,
    df = clustered_df
    userData = df.loc[df['user_id'] == target_uuid]
    user_label = userData['Cluster_labels'].reset_index(drop=True)
    cluster_members = df.loc[df['Cluster_labels'].isin(user_label)]

    # ????????? ???????????? ?????? ??? ?????? ?????????(uuid form)
    tuning_target_list = df.loc[df['user_id'].isin(good_uuid)]
    tuning_cluster = tuning_target_list['Cluster_labels'].reset_index(drop=True)
    tuning_cluster = set(tuning_cluster)
    tuning_cluster = list(tuning_cluster)
    tuning_cluster.append(userData['Cluster_labels'].reset_index(drop=True))
    tuned_cluster = df.loc[df['Cluster_labels'].isin(tuning_cluster)]
    all_clusters = pd.concat([tuned_cluster, cluster_members])
    all_clusters.drop_duplicates(['user_id'],inplace=True)
    result = recommend(target_uuid, all_clusters) #92,
    return result


