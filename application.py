from flask import Flask, escape, request, Response, jsonify
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime



def connectDB():
    conn = None
    cur = None
    sql = ""
    # conn = dbinfo

    #########################USER DATA(DB)#########################
    # user_id , birthdate, blood_type, frequency(added), is_donated(bool)(added), is_dormant(bool)(added), job
    # location, user_name, recency(xxxx-xx-xx), sex
    ###########################################################
    cur = conn.cursor()
    cur.execute("SELECT user_id, birthdate, blood_type, frequency, is_donated, is_dormant, job, location,user_name, recency, sex FROM user")
    user_col = [i[0] for i in cur.description]
    user_df = pd.DataFrame(cur.fetchall())
    user_df.columns = user_col
    pd.set_option('display.max_columns', None)
    today =datetime.now()

    birth = pd.to_datetime(user_df['birthdate'])
    age = today.year - birth.dt.year

    recency_date = pd.to_datetime(user_df['recency'])
    recency_list = []
    for i in range(len(recency_date)):
        if today.year == recency_date[i].year:
            recency = today.month - recency_date[i].month
        else:
            if today.month > recency_date[i].month:
                recency = (today.year - recency_date[i].year) * 12 + (today.month - recency_date[i].month)
            else:
                recency = (today.year - recency_date[i].year) * 12 + (12 - recency_date[i].month + today.month)
        recency_list.append(recency)
    user_df['birthdate'] = age
    user_df['recency'] =  recency_list
    user_df = user_df[['user_id','user_name','recency','sex','blood_type','birthdate','location',
    'job','frequency','is_donated','is_dormant']]
    return user_df
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


connectDB()
application = Flask(__name__)

########################################################################
# Get information of  user
# data:
# uuid is not null
# name is not null
# Recency is null
# Sex is not null
# blood_type is not null
# age is not null
# location is not null
# job is null
# good_list is null(uuid form)




########################################################################
# MODEL TRAINING
########################################################################

@application.route('/recommend/model', methods = (['POST']))
def updateModel():
    import automl
    userDB = connectDB()
    automl.preprocessing(userDB)
    return jsonify("model updated")
########################################################################


########################################################################
# Processing
########################################################################
@application.route('/recommend', methods = ['POST'])
def process():
    import realtime_clustering
    df = connectDB()
    clustered_df = realtime_clustering.to_csv(df)

    if (request.method == 'POST'):
        userId = request.json['userId']
        likedList = request.json['likedList']
        if (clustered_df['user_id'] == userId).any():
            if len(likedList) == 0:
                import recommendation
                result = recommendation.recommend(userId, clustered_df)
            else:
                import tuning_weight
                result = tuning_weight.concating(userId, likedList, clustered_df)
            result = result.tolist()
        else:
            result = []

    print(result)
    return jsonify( {'result': result})

import platform

if __name__ == "__main__":
    application.debug = True

    print("Current version of Python is ", platform.python_version())

    application.run()
    # application.run(debug = True, threaded=True, host="127.0.0.1", port=5001)
