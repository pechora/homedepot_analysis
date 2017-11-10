import pandas as pd
import numpy as np
import xgboost as xgb
import gensim
from porter2stemmer import Porter2Stemmer
from sklearn.metrics import r2_score
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz


def token_set(x):
    stra, strb = x.split('\t')
    return fuzz.token_set_ratio(stra, strb)

def token_sort(x):
    stra, strb = x.split('\t')
    return fuzz.token_sort_ratio(stra, strb)

def wv_sim(str, model):
    stra, strb = str.split('\t')
    count = 0.0
    wc = 0
    for word in stra.strip().split(' '):
        if word in model.wv.vocab:
            agg = 0
            for term in strb.strip().split(' '):
                if term in model.wv.vocab:
                    tx = (model.wv.similarity(word, term))
                    if(tx > agg):
                        agg = tx
            count += agg
            wc += 1
    return count / (wc if wc > 0 else 1)

def common_count(str):
    stra, strb = str.split('\t')
    count = 0
    for word in stra.strip().split(' '):
        if strb.find(word) >= 0:
            count += 1
    return count

op = None
r2 = 0.0

dfs = dict()
dfs['train'] = pd.read_csv('train.csv', encoding="ISO-8859-1")
dfs['test'] = pd.read_csv('test.csv', encoding="ISO-8859-1")
dsc = pd.read_csv('product_descriptions.csv', encoding="utf-8")
att = pd.read_csv('attributes.csv', encoding="utf-8")

stemmer = Porter2Stemmer()
brands_list = att[att['name'] == 'MFG Brand Name'][[
    "product_uid", "value"]].rename(columns={"value": "brand"})
aux_dsc = att[att['name'] == 'Bullet01'][["product_uid", "value"]].rename(
    columns={"value": "auxilary_description"})

sentences = np.array([])
for i, k in enumerate(dfs):
    for field in ['product_title', 'search_term']:
        sentences = np.append(sentences, dfs[k][field].map(lambda x: [stemmer.stem(word) for word in str(x).lower().split(' ')]).values)
sentences = np.append(sentences, dsc['product_description'].map(
    lambda x: [stemmer.stem(word) for word in str(x).lower().split(' ')]).values)
sentences = np.append(sentences, brands_list['brand'].map(
    lambda x: [stemmer.stem(word) for word in str(x).lower().split(' ')]).values)
sentences = np.append(sentences, aux_dsc['auxilary_description'].map(
    lambda x: [stemmer.stem(word) for word in str(x).lower().split(' ')]).values)

bigram_transformer = gensim.models.Phrases(sentences)
wv_model = Word2Vec(bigram_transformer[sentences], size=16, window=5, min_count=5, workers=4)

for i, k in enumerate(dfs):
    dfs[k] = pd.merge(dfs[k], dsc, how='left', on='product_uid')
    dfs[k] = pd.merge(dfs[k], brands_list, how='left', on='product_uid')
    dfs[k] = pd.merge(dfs[k], aux_dsc, how='left', on='product_uid')
    for field in ['product_title', 'brand', 'product_description', 'auxilary_description', 'search_term']:
        dfs[k][field] = dfs[k][field].map(lambda x: " ".join(
            [stemmer.stem(word) for word in str(x).lower().split(' ')]))
    dfs[k]['search_len'] = dfs[k]['search_term'].map(
        lambda x: len(x.split(' '))).astype(np.int32)
    for field in ['product_title', 'brand', 'product_description', 'auxilary_description']:
        dfs[k][field + '_common_count'] = (dfs[k]['search_term'] + '\t' + dfs[k][field]).map(
            lambda x: common_count(x)).astype(np.float32)
        dfs[k][field + '_wv_sim'] = (dfs[k]['search_term'] + '\t' + dfs[k][field]).map(
            lambda x: wv_sim(x, wv_model)).astype(np.float32)
        dfs[k][field + '_token_set'] = (dfs[k]['search_term'] + '\t' + dfs[k][field]).map(
            lambda x: token_set(x)).astype(np.float32)
        dfs[k][field + '_token_sort'] = (dfs[k]['search_term'] + '\t' + dfs[k][field]).map(
            lambda x: token_sort(x)).astype(np.float32)

y_train = dfs['train']['relevance'].values
x_test = dfs['test'][['search_len', 'product_title_common_count', 'product_title_wv_sim', 'product_title_token_set', 'product_title_token_sort', 'brand_common_count', 'brand_wv_sim', 'brand_token_set', 'brand_token_sort', 'product_description_common_count',
                      'product_description_wv_sim', 'product_description_token_set', 'product_description_token_sort', 'auxilary_description_common_count', 'auxilary_description_wv_sim', 'auxilary_description_token_set', 'auxilary_description_token_sort']].values
x_train = dfs['train'][['search_len', 'product_title_common_count', 'product_title_wv_sim', 'product_title_token_set', 'product_title_token_sort', 'brand_common_count', 'brand_wv_sim', 'brand_token_set', 'brand_token_sort', 'product_description_common_count',
                      'product_description_wv_sim', 'product_description_token_set', 'product_description_token_sort', 'auxilary_description_common_count', 'auxilary_description_wv_sim', 'auxilary_description_token_set', 'auxilary_description_token_sort']].values


model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08,
                         gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
model.fit(x_train, y_train)
y = model.predict(x_train)
print(r2_score(y_train, y))
print(np.mean((y_train - y) ** 2) ** 0.5)

y = model.predict(x_test)
ans = pd.DataFrame({"id": dfs['test']['id'], "relevance": y})
ans.to_csv('answers.csv', index=False)
