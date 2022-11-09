import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time
# 设置随机种子
features = np.load("/gpfs/home3/wangxi/RF_tune_pars/DATA_inputs/inputs_asc_final.npy")[:,:11]
target = np.load("/gpfs/home3/wangxi/RF_tune_pars/DATA_inputs/spl3_asc_final.npy")
print("特征数组的大小为：{}，目标数组的大小为：{}". format(features.shape,target.shape))

##================================================================================================#
# 分割无缺的特征数据组和SMAP SM数据
# 为了便于索引，存放在pandas里面进行处理
vars_train = ['doy','lon','lat', 'bulk','clay','sand','elev','slope','roughness','prcp','lst']
vars_tar = ['smap']
features_pd = pd.DataFrame(features)
tar_pd = pd.DataFrame(target)
features_pd.columns = vars_train
tar_pd.columns = vars_tar
# 找到水体部分进行标识
#id_water = features_pd[features_pd['lc'].isin([0])].index
features_copy = features_pd.copy()
tar_copy = tar_pd.copy()
#features_copy.loc[id_water] = np.nan
#tar_copy.loc[id_water] = np.nan
# 分割SMAP存在数据的网格和
id_miss = tar_copy[tar_copy['smap'].isnull()].index
id_rcmd = tar_copy[tar_copy['smap'].notnull()].index
smap_rcmd = tar_copy.loc[id_rcmd]
smap_gaps = tar_copy.loc[id_miss]
features_train = features_copy.loc[id_rcmd]
features_gaps = features_copy.loc[id_miss]
print("SMAP升轨数据中可作为目标的推荐值数据量为{}，缺失数据量为{}(包含水体区域), 参与训练的特征矩阵大小为{}". 
      format(smap_rcmd.size, smap_gaps.size,features_train.shape))


nan_rows = features_train[features_train.isnull().any(axis=1)].index
train_X = features_train.drop(nan_rows)
train_Y = smap_rcmd.drop(nan_rows)
# Read the pars calculated from the HPC
# import pickle
# par_path = r'E:\Iscience修订\hyper_pars_CV_results\final_BestPars_asc_HFGSCV.pkl'
# with open(par_path,'rb') as f:
#     hyperPars = pickle.load(f)
# hyperPars

# use the trained model to predict the missing values 
import time
start = time.time()
rfc = RandomForestRegressor(n_estimators = 1200,
                            max_depth = 20,
                            max_features = 0.5,
                            min_samples_split = 2,
                            min_samples_leaf = 8,
                            n_jobs = -1, 
                            bootstrap = True,
                            random_state = 18, 
                            oob_score = True,)
rf_model = rfc.fit(train_X.values, train_Y.values.ravel())
end = time.time()
print ("该cell耗时：{:.2f}秒". format(end-start))
print(rf_model.oob_score_, rf_model.score(train_X.values, train_Y.values.ravel()))

import joblib
# save
# joblib.dump(rf_model, r"/gpfs/home3/wangxi/RF_tune_pars/RFreg_find_HyperPars/rf_asc_model_v2.joblib")

# Predict
# 删除含有nan的所有行
import time
gaps_auxi =  features_gaps.copy()
auxi_inputs = gaps_auxi[~gaps_auxi.isnull().any(axis = 1)] # 1448,335
loc_auxi_fill = auxi_inputs.index
# 输入无缺的辅助数据，然后放入RF模型查看结果
#imp = SimpleImputer(missing_values= np.nan, strategy = 'constant', fill_value = -9999)
#auxi_inputs = imp.fit_transform(gaps_auxi)
try:
    gaps_sm = rf_model.predict(auxi_inputs)
except:
    print("回归预测失败")
gaps_fill = pd.DataFrame(gaps_sm)
gaps_fill.index = loc_auxi_fill
# 回填到SMAP预测位置
smap_fillgaps = smap_gaps.copy()
smap_fillgaps.loc[loc_auxi_fill] = gaps_fill
# 回填到loc_miss中
smap_nogaps = tar_copy.copy()
smap_fillgaps.index = id_miss
smap_nogaps.loc[id_miss] = smap_fillgaps
smap_gapfilled = np.array(smap_nogaps)
smap_gapfilled = smap_gapfilled.reshape(1737,46,30)
# 原始三维SMAP数组
# smap_og = target.reshape(1737,46,30)
import scipy.io as io
mat_path = '/gpfs/home3/wangxi/RF_tune_pars/rf_outputs/RF_SPL3SM_asc_v0.mat'
io.savemat(mat_path, {'rfsmap_asc': smap_gapfilled})