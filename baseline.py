'''
移动基站退服预测 baseline 0.616
团队： 虎牙大哥
'''

import os
import pandas as pd
import numpy as np
import datetime
import requests
import json
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,roc_curve,auc,accuracy_score,precision_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import lightgbm as lgb
import gc
from tqdm import tqdm
import time
import datetime

train_path = '/mnt/5/Alert_BTS_HW_1001-0309'
# 0316-0322
test_0322_path = '/mnt/5/Alert_BTS_HW_0316-0322'
#0324-0330
test_0330_path = '/mnt/5/Alert_BTS_HW_0324-0330'

# 处理测试数据集
all_test_data = pd.DataFrame()
for now_csv in tqdm(os.listdir(test_0322_path)):
    data = pd.read_csv(os.path.join(test_0322_path, now_csv))
    data['end_time'] = '2020-03-23'
    data['label'] = -1
    data['ID'] = data['基站名称']
    all_test_data = all_test_data.append(data)

for now_csv in tqdm(os.listdir(test_0330_path)):
    data = pd.read_csv(os.path.join(test_0330_path, now_csv))
    data['end_time'] = '2020-03-31'
    data['label'] = -1
    data['ID'] = data['基站名称']
    all_test_data = all_test_data.append(data)
all_test_data['end_time'] = pd.to_datetime(all_test_data['end_time'], format='%Y-%m-%d')
all_test_data['告警开始时间'] = pd.to_datetime(all_test_data['告警开始时间'], format='%Y-%m-%d %H:%M:%S')
all_test_data['time_gap'] = all_test_data.apply(lambda x: (x['end_time'] - x['告警开始时间']).days, axis=1)
all_test_data['time_gap'] += 1

# 构造训练数据集
def get_all_train_data(train_path):
    all_train_data = pd.DataFrame()
    ID = 0
    print(len(os.listdir(train_path)))
    for now_csv in tqdm(os.listdir(train_path)):
        data = pd.read_csv(os.path.join(train_path,now_csv))
        all_train_data = all_train_data.append(data)
    all_train_data['end_time'] = '2020-03-10'
    all_train_data['告警开始时间'] = pd.to_datetime(all_train_data['告警开始时间'], format='%Y-%m-%d %H:%M:%S')
    all_train_data['end_time'] = pd.to_datetime(all_train_data['end_time'], format='%Y-%m-%d')
    all_train_data['time_gap'] = all_train_data.apply(lambda x: (x['end_time']-x['告警开始时间']).days, axis=1)
    return all_train_data
save_path = './all_train_data.csv'
if not os.path.exists(save_path):
    all_train_data = get_all_train_data(train_path)
    all_train_data.to_csv(save_path, index=False)
all_train_data = pd.read_csv('all_train_data.csv')


# 生成训练集
def gener_train_data(all_data, times):
    '''
    start_i: 从第几天开始抽取
    times： 抽取多少次
    '''
    print('*********')
    res_data = pd.DataFrame()
    all_data['end_time'] = pd.to_datetime(all_data['end_time'], format='%Y-%m-%d %H:%M:%S')
    for i in tqdm(range(times)):
        label_data = all_data[all_data['time_gap'] == i]
        # 生成label
        label_data['label'] = label_data['告警名称'].apply(
            lambda x: 1 if x.strip() == '网元连接中断' or x.strip() == '小区不可用告警' else 0)
        label_data = label_data.groupby('基站名称')['label'].agg('sum').reset_index()
        label_data['label'] = label_data['label'].apply(lambda x: 1 if x > 0 else 0)

        # 取前7天的训练数据
        tmp_data = all_data[(all_data['time_gap'] > i) & (all_data['time_gap'] <= i + 7)]
        # 处理时间 保持同一个窗口内，大小为7i
        tmp_data['time_gap'] = tmp_data['time_gap'] - i
        tmp_data = tmp_data.merge(label_data, on='基站名称', how='left')

        tmp_data['ID'] = tmp_data['基站名称'] + '_' + str(i)
        tmp_data['end_time'] = tmp_data['end_time'] - datetime.timedelta(days=i)

        res_data = res_data.append(tmp_data)
    return res_data

# 生成 的训练数据
def get_train_data(all_data, times=1):
    save_path = './all_train_data_sample_'+str(times)+'.csv'
    if not os.path.exists(save_path):
        res_data = gener_train_data(all_data, times)
        res_data.to_csv(save_path, index=False)
    res_data = pd.read_csv(save_path)
    return res_data
offline_train_data = get_train_data(all_train_data, times=30)


# 构造特征
def gener_fea(data):
    res = data[['ID', 'end_time', 'label']].drop_duplicates()
    # 构造特征
    # 1、统计每个样本在1，2，3，4，5，6，7天内故障出现的总次数
    tmp = data.groupby(by=['ID', 'time_gap']).agg({'基站名称': 'count'}).reset_index()
    tmp = pd.pivot_table(tmp, index='ID', columns='time_gap', values='基站名称').reset_index()
    # 7天内故障的总次数
    tmp['sum_guzhang_7'] = tmp[[1, 2, 3, 4, 5, 6, 7]].apply(lambda x: x.sum(), axis=1)
    tmp.rename(columns={1: 'guzhang_1', 2: 'guzhang_2', 3: 'guzhang_3', 4: 'guzhang_4', 5: 'guzhang_5', 6: 'guzhang_6',
                        7: 'guzhang_7'}, inplace=True)
    res = res.merge(tmp, on='ID', how='left')

    # 7天内出现故障的总类型数量
    tmp = data.groupby(by=['ID']).agg({'告警名称': 'nunique'}).reset_index().rename(columns={'告警名称': '7_gaojing_nunique'})
    res = res.merge(tmp, on='ID', how='left')

    # 每天出现故障的类型数量
    tmp = data.groupby(by=['ID', 'time_gap']).agg({'告警名称': 'nunique'}).reset_index()
    tmp = pd.pivot_table(tmp, index='ID', columns='time_gap', values='告警名称').reset_index()

    tmp.rename(columns={1: 'guzhang_types_1', 2: 'guzhang_types_2', 3: 'guzhang_types_3', 4: 'guzhang_types_4',
                        5: 'guzhang_types_5',
                        6: 'guzhang_types_6', 7: 'guzhang_types_7'}, inplace=True)
    res = res.merge(tmp, on='ID', how='left')

    # 7天内出现故障的天数
    tmp = data.groupby(by=['ID']).agg({'time_gap': 'nunique'}).reset_index().rename(
        columns={'time_gap': 'time_gap_nunique'})
    res = res.merge(tmp, on='ID', how='left')
    # 7天内，平均每天的故障次数
    res['sum_guzhang_7/7'] = res['sum_guzhang_7'] / 7

    # 发生故障时，平均每天的故障次数
    res['sum_guzhang_7/time_gap_nunique'] = res['sum_guzhang_7'] / res['time_gap_nunique']

    # 故障告警、异常告警、失败告警、
    def get_guzhang(x, gaojing_type):
        res = 0
        for i in x:
            if i.find(gaojing_type) != -1:
                res += 1
        return res

    # 故障类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, '故障')).reset_index().rename(columns={'告警名称': 'guzhang_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 异常类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, '异常')).reset_index().rename(columns={'告警名称': 'yichang_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 失败类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, '失败')).reset_index().rename(columns={'告警名称': 'shibai_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 小区类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, '小区')).reset_index().rename(columns={'告警名称': 'xiaoqu_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 射频类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, '射频')).reset_index().rename(columns={'告警名称': 'shepin_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # BBU类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, 'BBU')).reset_index().rename(columns={'告警名称': 'BBU_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # RHUB类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, 'RHUB')).reset_index().rename(columns={'告警名称': 'RHUB_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 射频类告警的次数
    tmp = data.groupby(by=['ID'])['告警名称'].apply(
        lambda x: get_guzhang(x, 'RRU')).reset_index().rename(columns={'告警名称': 'RRU_sum'})
    res = res.merge(tmp, on='ID', how='left')

    # 7天内发生 网元连接中断 和 小区不可用告警 的天数

    tmp = data[(data['告警名称'] == '网元连接中断') | (data['告警名称'] == '小区不可用告警')]
    tmp['start_time'] = tmp['告警开始时间'].apply(lambda x: str(x)[:10])
    tmp = tmp.drop_duplicates(subset=['ID', 'start_time'], keep='first')
    tmp = tmp.groupby(['ID']).agg({'start_time': 'nunique'}).reset_index().rename(columns={'start_time': 'label1_days'})
    res = res.merge(tmp, on='ID', how='left')

    # 最近一次发生 网元连接中断 和 小区不可用告警 的天数 距今的时间
    tmp = data[(data['告警名称'] == '网元连接中断') | (data['告警名称'] == '小区不可用告警')]
    tmp['告警开始时间'] = pd.to_datetime(tmp['告警开始时间'], format='%Y-%m-%d %H:%M:%S')
    tmp['end_time'] = pd.to_datetime(tmp['end_time'], format='%Y-%m-%d %H:%M:%S')
    tmp['near_label1_gaojing_gap'] = tmp.apply(lambda x: (x['end_time'] - x['告警开始时间']).days, axis=1)

    tmp2 = tmp.groupby(by=['ID', '告警名称']).agg({'near_label1_gaojing_gap': 'min'}).reset_index()
    tmp2 = pd.pivot_table(tmp2, index='ID', columns='告警名称', values='near_label1_gaojing_gap').reset_index().rename(
        columns={'网元连接中断': 'near_wangyuan_gap', '小区不可用告警': 'near_xiaoqu_gap'})
    res = res.merge(tmp2, on='ID', how='left')

    # 每个  网元连接中断 或 小区不可用告警 的时间间隔

    tmp3 = tmp
    rename = 'label1_gap'
    tmp3.sort_values(['ID', '告警开始时间'], inplace=True)
    tmp3['next_gaojing_time'] = tmp3.groupby(by=['ID'])['告警开始时间'].shift(-1)
    tmp3['gaojing_gaps'] = tmp3.apply(lambda x: (x['next_gaojing_time'] - x['告警开始时间']).seconds, axis=1)
    tmp3 = tmp3[['ID', '告警开始时间', 'next_gaojing_time', 'gaojing_gaps']]
    tmp3.dropna(subset=['gaojing_gaps'], inplace=True)

    tmp3 = tmp3.groupby(by=['ID'])['gaojing_gaps'].agg(['max', 'min', 'mean', 'std', 'skew']).reset_index().rename(
        columns={'max': rename + '_max', 'min': rename + '_min', 'mean': rename + '_mean', 'std': rename + '_std',
                 'skew': rename + '_skew'})
    res = res.merge(tmp3, on='ID', how='left')
    #     print(tmp3)

    # 统计每个告警类型，在7天内发生的次数

    tmp = data.groupby(by=['ID', '告警名称']).agg({'基站名称': 'count'}).reset_index()
    tmp = pd.pivot_table(tmp, index='ID', columns='告警名称', values='基站名称').reset_index()
    cols = {}
    i = 0
    for col in tmp.columns:
        if col not in ['ID', 'end_time', 'label']:
            cols[col] = i
            i += 1
    tmp.rename(columns=cols, inplace=True)
    res = res.merge(tmp, on='ID', how='left')
    res['0/sum7'] = res[0] / res['sum_guzhang_7']
    res['1/sum7'] = res[1] / res['sum_guzhang_7']
    res['3/sum7'] = res[3] / res['sum_guzhang_7']

    return res



def search_threthold(true, pred):
    score = 0
    bestThrethold = 0
    for i in np.arange(0, 1, 0.01):
        if f1_score(true, np.where(pred > i, 1, 0)) > score:
            score = f1_score(true, np.where(pred > i, 1, 0))
            bestThrethold = i
        else:
            pass
    return bestThrethold

def train_lgb_model(train_, valid_, valid_2, id_name, label_name, categorical_feature=None, seed=1024, is_shuffle=True):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_['res'] = 0
    pred = [col for col in train_.columns if col not in [id_name, label_name, 'res']]
    print('特征数量为：', len(pred))
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    sub_preds2 = np.zeros((valid_2.shape[0], folds.n_splits))
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 32,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': 1,
        # 'device': 'gpu',
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 28,
        'nthread': -1,
        'verbose': -1,
    }
    fea_impor = pd.DataFrame()
    fea_impor['column'] = train_[pred].columns
    fea_impor['importance'] = 0

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_[label_name]), start=1):
        print(f'the {n_fold} training start ...')

        train_x, train_y = train_[pred].iloc[train_idx], train_[label_name].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label_name].iloc[valid_idx]

        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=100,
            # feval=fscore,
            verbose_eval=100,
        )
        fea_impor['tmp'] = clf.feature_importance()
        fea_impor['importance'] = fea_impor['importance'] + fea_impor['tmp']

        sub_preds[:, n_fold - 1] = clf.predict(valid_[pred], num_iteration=clf.best_iteration)
        sub_preds2[:, n_fold - 1] = clf.predict(valid_2[pred], num_iteration=clf.best_iteration)
        train_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
        tmp_score = roc_auc_score(valid_y, train_pred)
        train_['res'].iloc[valid_idx] = train_['res'].iloc[valid_idx] + train_pred
        print(f'Orange roc_auc_score score: {tmp_score}')

    tmp_score = roc_auc_score(train_[label_name], train_['res'])
    print(f'five flod roc_auc_score score: {tmp_score}')
    train_.sort_values(by=['res'], ascending=False, inplace=True)

    # 按照0.5划分
    th = search_threthold(train_[label_name], train_['res'])
    train_['res'] = train_['res'].apply(lambda x: 1 if x > th else 0)

    tmp_f1 = f1_score(train_[label_name], train_['res'])

    print(f'five flod tmp_f1 score: {th, tmp_f1}')

    valid_[label_name] = np.mean(sub_preds, axis=1)
    valid_2[label_name] = np.mean(sub_preds2, axis=1)

    valid_['基站名称'] = valid_[id_name]
    valid_2['基站名称'] = valid_2[id_name]

    valid_['未来24小时发生退服类告警的概率'] = valid_[label_name]
    valid_2['未来24小时发生退服类告警的概率'] = valid_2[label_name]

    return th, valid_[['基站名称', '未来24小时发生退服类告警的概率']], valid_2[['基站名称', '未来24小时发生退服类告警的概率']],valid_2[['基站名称', '未来24小时发生退服类告警的概率']], valid_2[['基站名称', '未来24小时发生退服类告警的概率']]

online_train_data = get_train_data(all_train_data, start_i=0, times=30)
all_data = online_train_data.append(all_test_data)
all_data_fea = gener_fea(all_data)
tmp = online_train_data.groupby(by=['ID'])['告警名称'].apply(lambda x: ' '.join(x.tolist())).reset_index()

all_data_fea['end_time'] = all_data_fea['end_time'].apply(lambda x:str(x)[:10])
train_data=all_data_fea[(all_data_fea['end_time']!='2020-03-23') & (all_data_fea['end_time']!='2020-03-31')]
test1_data=all_data_fea[all_data_fea['end_time']=='2020-03-23']
test2_data=all_data_fea[all_data_fea['end_time']=='2020-03-31']

cols = [col for col in train_data.columns if col not in ['ID','label','end_time']]
val1, val2 = train_lgb_model(train_data[cols+['ID','label']], test1_data[cols+['ID']], test2_data[cols+['ID']], 'ID', 'label')
val1.to_csv('./Sample23_1日.csv', index=False)
val2.to_csv('./Sample31_1日.csv', index=False)

cols = [col for col in train_data.columns if col not in ['ID','label','end_time']]

th, val1, val2 = train_lgb_model(train_data[cols+['ID','label']], test1_data[cols+['ID']], test2_data[cols+['ID']], 'ID', 'label')
val1.to_csv('./Sample23_1日.csv', index=False)
val2.to_csv('./Sample31_1日.csv', index=False)


# 生成提交文件
sub1 = pd.read_csv('/mnt/5/提交文件样例/Sample23日.csv', encoding='gbk')
sub2 = pd.read_csv('/mnt/5/提交文件样例/Sample31日.csv', encoding='gbk')

val1['未来24小时发生退服类告警的概率'] = val1['未来24小时发生退服类告警的概率'].apply(lambda x: 1 if x>th else 0)
val2['未来24小时发生退服类告警的概率'] = val2['未来24小时发生退服类告警的概率'].apply(lambda x: 1 if x>th else 0)

sub1 = sub1[['基站名称']].merge(val1, on='基站名称', how='left')
sub2 = sub2[['基站名称']].merge(val2, on='基站名称', how='left')
sub1.to_csv('/root/models/results/Sample23日.csv', index=False)
sub2.to_csv('/root/models/results/Sample31日.csv', index=False)

import sklearn.preprocessing as pre_processing
import numpy as np

label = pre_processing.LabelEncoder()
label.fit()
labels = label.fit_transform(['中国', '美国', '法国', '德国'])
print(labels)