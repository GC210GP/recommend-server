import pandas as pd
import numpy as np
from random import choice
import uuid
####################################################################################################################
# Columns Information
#1. Recency - months since last donation(NULL) o
#2. Sex - Gender(NOT NULL)
#3. blood_type -  blood type (NOT NULL)
#4. Age - Age (NOT NULL)
#5. location - County (NOT NULL)
#6. job - Job (NOT NULL)

original_df = pd.read_csv('blood.csv')

original_df.drop(['V2','V3','V4','Class'], axis=1,inplace=True)
original_df.rename(columns = {'V1':'Recency'},inplace=True)
choice_list = []
name_list = []
for i in range(0,len(original_df)):
    valueslist = original_df['Recency'].tolist()
    namelist = original_df['name'].tolist()
    for _ in range(100):
        selection = choice(valueslist)
        choice_list.append(selection)
        nameSelec = choice(namelist)
        name_list.append(nameSelec)

expanded_df= pd.DataFrame({'Recency':choice_list,'name':name_list})
frames = [original_df,expanded_df]
df = pd.concat(frames, ignore_index=True)
#expanded = {'Recency':choice_list}
#expanded_name = {'name':name_list}
#df['Recency'] = df['Recency'].append(pd.DataFrame(expanded))
#df['name'] = df['name'].append(pd.DataFrame(expanded_name))


#Insert additional column for new information
# Additional datas are randomly assigned according to real statistic
df["Sex"] = np.random.choice(["Male", "Female"], len(df), p=[0.58, 0.42])
df['blood_type'] = np.random.choice(["O","A","B","AB"], len(df), p = [0.28,0.33,0.27,0.12])
df['age'] = np.random.choice(['16-19','20-29','30-39','40-49','50-59','60세 이상'],len(df), p = [0.14,0.39,0.19,0.17,0.09,0.02])
df['location'] = np.random.choice(['서울','부산','대구','인천','광주','대전','울산','경기','강원','충북','충남','전북','전남','경북','경남','제주','기타'],len(df),
                                 p=[0.187,0.065,0.046,0.057,0.028,0.028,0.022,0.26,0.029,0.031,0.041,0.035,0.036,0.051,0.064,0.013,0.007])
df['job'] = np.random.choice(['고등학생','대학생','군인','회사원','공무원','자영업','종교직','가사','기타'],len(df),
                             p=[0.124,0.207,0.14,0.325,0.051,0.025,0.003,0.022,0.103])
df['uuid'] = [uuid.uuid4() for _ in range(len(df))]
df = df[['uuid', 'name', 'Recency', 'Sex', 'blood_type', 'age', 'location', 'job']]

df.to_csv('new_data.csv', index=False, encoding='utf-8-sig')
#1284
