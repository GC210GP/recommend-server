from flask import Flask, escape, request, Response
import pandas as pd
import numpy as np

app = Flask(__name__)
@app.route('/connect', methods = ('GET','POST'))


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

def get_info():
    if request.method == 'POST':
        data = request.form['input']
        info_data = pd.Series(data[:-1])
        good_list = pd.Series(data[-1])

    return info_data, good_list

########################################################################
# Processing
########################################################################
def process():
    info, good = get_info()
    user_uuid = info['uuid']
    original_df = pd.read_csv('new_data.csv')
    clustered_df = pd.read_csv('clustered.csv')
    print(clustered_df['uuid'] == user_uuid).any()
    import recommendation
    if (clustered_df['uuid'] == user_uuid).any():
        if good is None:
            result = recommendation.recommend(user_uuid,clustered_df)
            print(result)
        else:
            import tuning_weight
            result = tuning_weight.concating(user_uuid, good)
            print(result)
    else:
        import realtime_clustering
        realtime_clustering.to_csv(original_df)
        result = recommendation.recommend(user_uuid, clustered_df)
        print(result)
    if request.method == 'GET':
        Response(result.to_json(orient='records'), mimetype='application/json')


if __name__ == "__main__":
    app.run(debug = True, threaded=True, host="127.0.0.1", port=5000)






