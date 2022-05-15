# Import Class Libraries
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
#########################USER DATA(Dataframe)#########################
# user_id       non-null  int
# user_name     non-null  str
# recency       non-null  int
# sex           non-null  str
# blood_type    non-null  str
# birthdate     non-null  int
# location      non-null  str
# job           non-null  str
# frequency     non-null  int
# is_donated    non-null  boolean
# is_dormant    non-null  boolean
######################################################################


scale_col = ['recency', 'frequency']
encode_col = ['sex','location','job', 'ageGroup','is_donated','is_dormant']
def preprocessing(df):
    df = pd.DataFrame(df)
    df = df[['user_id', 'user_name', 'recency', 'sex', 'blood_type', 'birthdate', 'location', 'job', 'frequency','is_donated', 'is_dormant']]
    df['recency'].fillna(df['recency'].mean(), inplace=True)
    df['frequency'].fillna(df['frequency'].mean(), inplace=True)
    df['ageGroup'] = df['birthdate'].apply(lambda x: get_category(x))
    df = df.fillna(method='ffill')
    result = AutoML(df, scale_col=scale_col, encode_col=encode_col, encoders=None,
                    scalers=None)
    joblib.dump(result.best_estimator_, './dbscanModel.pkl')

# age categorized
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
# Find outlers
def outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3,True,False)
    return pd.Series(idx_outliers, index=col.index)



# Description = Calculate the silhouette score and return the value
# Input  = kind of model, Dataset
# Output = Silhouette score
def cv_silhouette_scorer(estimator, X):
    print("Randomized Searching... : ", estimator)

    # If GMM(EM) handle separately
    if type(estimator) is sklearn.mixture._gaussian_mixture.GaussianMixture:
        labels = estimator.fit_predict(X)
        score = silhouette_score(X, labels, metric='euclidean')
        return score


    else:
        cluster_labels = estimator.fit_predict(X)
        num_labels = len(set(cluster_labels))
        num_samples = len(X.index)
        if num_labels == 1 or num_labels == num_samples:
            return -1

        else:
            return silhouette_score(X, cluster_labels)

def AutoML(X, scale_col=None, encode_col = None, encoders=None, scalers=None,scores=None, score_param=None):
    model = DBSCAN()
    scaler = StandardScaler()
    encoder = OrdinalEncoder()
    df_scaled = scaler.fit_transform(X[scale_col])
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_col

    if scores is None:
        score = ['''silhouette_score()''']
    else:
        score = scores

        # Set Score parameter
    if score_param is None:
        score_parameter = [None]
    else:
        score_parameter = score_param

    # Encoding
    if encode_col is not None:
        df_encoded = encoder.fit_transform(X[encode_col])
        df_encoded = pd.DataFrame(df_encoded)
        df_encoded.columns = encode_col
        df_prepro = pd.concat([df_scaled, df_encoded], axis=1)

    df_prepro = pd.DataFrame(df_prepro)
    df_prepro = pd.DataFrame(df_prepro)
    param_grid = {
        'eps':[0.75,0.1,0.25,0.5,1],
        'min_samples':[5,20,100,200,500]
    }
    model_tuned = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                                               scoring=cv_silhouette_scorer)
    result = model_tuned.fit(df_prepro)
    
    
    # Log trained data to csv file
    timestamp = datetime.now()
    
    best_model = result.best_estimator_
    labels = best_model.labels_

    # xx = pd.DataFrame({"aa": [1, 2, 3, 4], "bb": [2, 3, 4, 5]})
    out = pd.DataFrame(X)
    out["label"] = labels
    
    dir = "./logs/cluster-" + str(timestamp) + ".csv"
    dir = dir.replace(":", "_").replace("-", "_").replace(" ", "__")
    out.to_csv(dir)

    model_result_out = pd.DataFrame(
        {
            "Date": [timestamp],
            "num_of_people": [len(X)],
            "eps": [best_model.eps],
            "min_samples": [best_model.min_samples],
            "num_of_cluster": [max(best_model.labels_)],
        }
    )

    dir = "./logs/result-" + str(timestamp) + ".csv"
    dir = dir.replace(":", "_").replace("-", "_").replace(" ", "__")
    model_result_out.to_csv(dir)

    return result
    # Auto Find Best Accuracy


