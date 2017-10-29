import pandas as pd
import numpy as np
import xgboost as xgb
from porter2stemmer import Porter2Stemmer
from sklearn.metrics import r2_score

def common_count(str):
    stra, strb = str.split('\t')
    count = 0
    for word in stra.strip().split(' '):
        if strb.find(word) >= 0:
            count += 1
    return count

dfs = dict()
dfs['train'] = pd.read_csv('train.csv', encoding = "ISO-8859-1")
dfs['test'] = pd.read_csv('test.csv', encoding = "ISO-8859-1")
dsc = pd.read_csv('product_descriptions.csv', encoding = "utf-8")
att = pd.read_csv('attributes.csv', encoding = "utf-8")

stemmer = Porter2Stemmer()
aux_dsc = att[att['name'] == 'Bullet01'][["product_uid", "value"]].rename(columns = {"value" : "auxilary_description"})
brands_list = att[att['name'] == 'MFG Brand Name'][["product_uid", "value"]].rename(columns = {"value" : "brand"})

for i, k in enumerate(dfs):
    dfs[k] = pd.merge(dfs[k], dsc, how = 'left', on='product_uid')
    dfs[k] = pd.merge(dfs[k], aux_dsc, how = 'left', on='product_uid')
    dfs[k] = pd.merge(dfs[k], brands_list, how = 'left', on='product_uid')

    for field in ['product_title', 'brand', 'product_description', 'auxilary_description', 'search_term']:
        dfs[k][field] = dfs[k][field].map(lambda x: " ".join([stemmer.stem(word) for word in str(x).lower().split(' ')]))

    dfs[k]['search_len'] = dfs[k]['search_term'].map(lambda x: len(x.split(' '))).astype(np.int32)
    dfs[k]['title_common_count'] = (dfs[k]['search_term'] + '\t' + dfs[k]['product_title']).map(lambda x: common_count(x)).astype(np.int32)
    dfs[k]['brand_common_count'] = (dfs[k]['search_term'] + '\t' + dfs[k]['brand']).map(lambda x: common_count(x)).astype(np.int32)
    dfs[k]['desc_common_count'] = (dfs[k]['search_term'] + '\t' + dfs[k]['product_description']).map(lambda x: common_count(x)).astype(np.int32)
    dfs[k]['aux_desc_common_count'] = (dfs[k]['search_term'] + '\t' + dfs[k]['auxilary_description']).map(lambda x: common_count(x)).astype(np.int32)

x_train = dfs['train'][['search_len', 'title_common_count', 'brand_common_count', 'desc_common_count', 'aux_desc_common_count']].values
y_train = dfs['train']['relevance'].values
x_test = dfs['test'][['search_len', 'title_common_count', 'brand_common_count', 'desc_common_count', 'aux_desc_common_count']].values

model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
model.fit(x_train, y_train)
y = model.predict(x_train)
print(r2_score(y_train, y))

y = model.predict(x_test)
ans = pd.DataFrame({"id": dfs['test']['id'], "relevance": y})
ans.to_csv('answers.csv',index=False)
