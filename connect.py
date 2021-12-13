from flask import Flask, escape, request, Response
import pandas as pd
import numpy as np
app = Flask(__name__)



@app.route('/connect', methods = ('GET','POST'))
def information():
    if request.method == 'GET':
        uuid = request.args.get("uuid")
        #uuid = '4d23021e-1b66-4daa-9d68-17673153d4f3'
        df = pd.read_csv('clustered.csv')
        userData = df.loc[df['uuid'] == uuid]
        user_label = userData['Cluster_labels'].reset_index(drop=True)
        print(user_label)
        cluseter_members = df.loc[df['Cluster_labels'].isin(user_label)]
        cluseter_members = cluseter_members['uuid']
        print(cluseter_members)

        if len(cluseter_members) <= 10:
            return Response(cluseter_members.to_json(orient='records'), mimetype='application/json')

        else:
            recommended = cluseter_members.sample(10)
            return Response(recommended.to_json(orient='records'), mimetype='application/json')




if __name__ == "__main__":
    app.run(debug = True, threaded=True, host="127.0.0.1", port=5000)


