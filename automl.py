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

original_df = pd.read_csv('new_data.csv')
original_df = original_df[['name','Recency','Sex','blood_type','age','location','job']]
original_df['Recency'].fillna(original_df['Recency'].mean(), inplace=True)
original_df = original_df.fillna(method='ffill')

indexed_df = original_df.set_index('name')

encode_col = ['Sex','blood_type','age','location','job']
scale_col = ['Recency']


# Find outlers
def outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3,True,False)
    return pd.Series(idx_outliers, index=col.index)

#Remove outliers
for n in range(len(scale_col)):
    idx = None
    idx = outliers(indexed_df.iloc[:,n])
    indexed_df = indexed_df.loc[idx==False]


# Description = Calculate the silhouette score and return the value
# Input  = kind of model, Dataset
# Output = Silhouette score
def cv_silhouette_scorer(estimator, X):
    print("Randomized Searching... : ", estimator)

    # If GMM(EM) handle separately
    if type(estimator) is sklearn.mixture._gaussian_mixture.GaussianMixture:
        # print("it's GaussianMixture()")
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
    df_scaled = pd.DataFrame(scaler.fit_transform(X[scale_col]))
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
        'min_samples':[5,20,100,200]
    }
    model_tuned = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                                               scoring=cv_silhouette_scorer)
    result = model_tuned.fit(df_prepro)
    score = scores
    print(score)
    best_model = result.best_estimator_
    best_params = result.best_params_
    print('best params type: ', type(best_params))
    print('best params: ', best_params)
    return best_params
    # Auto Find Best Accuracy
print("Auto Find Best Accuracy")
best_param = AutoML(indexed_df, scale_col=scale_col, encode_col=encode_col, encoders=None,
       scalers=None)

