import pandas as pd
import numpy as np

df = pd.read_csv('clustered.csv')
def recommendation(uuid, df):

    df = pd.read_csv('clustered.csv')
    userData = df.loc[df['uuid'] == uuid]
    user_label = userData['Cluster_labels'].reset_index(drop=True)
    print(user_label)
    cluseter_members = df.loc[df['Cluster_labels'].isin(user_label)]
    cluseter_members = cluseter_members['uuid']
    print(cluseter_members)
    return cluseter_members

recommendation('4d23021e-1b66-4daa-9d68-17673153d4f3', df)