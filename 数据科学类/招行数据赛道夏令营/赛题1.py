import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# 加载数据
train_dataset = pd.read_csv("data/train_dataset.csv")
test_dataset = pd.read_csv('data/test_dataset.csv')
user_features = pd.read_csv('data/user_features.csv')
ad_features = pd.read_csv("data/ad_features.csv")

# 合并数据集
data = pd.concat([train_dataset, test_dataset], ignore_index=True)
data['exposure_time'] = pd.to_datetime(data['exposure_time'])  # 时间格式转换

# 合并特征
data = data.merge(user_features, on='user_id', how='left')  # 合并user特征
data = data.merge(ad_features, on='ad_id', how='left')  # 合并ad特征

# 1. 时间衍生特征
data['week'] = data['exposure_time'].dt.week
data['hour'] = data['exposure_time'].dt.hour
data['hour_m'] = data['exposure_time'].dt.hour + data['exposure_time'].dt.minute/60
data['cos_hour'] = np.cos(2 * np.pi * data['hour_m'] / 24)

# 2. user侧特征
# LabelEncoder
label_encoders = {}
for col in ['occupation', 'category', 'material_type', 'region', 'device']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 计算单位活跃度下的购买能力
data['purchase_efficiency'] = data['purchase_history'] / (data['activity_score'] + 1e-6)

# 频率编码
for col in ['occupation', 'category', 'material_type', 'region', 'device']:
    # 计算每个类别的频率
    freq_map = data[col].value_counts(normalize=True).to_dict()
    # 创建新特征:{col}_freq
    data[f'{col}_freq'] = data[col].map(freq_map)

# 3. ad侧特征
data['ad_quality_score'] = data['advertiser_score'] * data['historical_ctr']  # 广告主质量与广告表现的综合得分

# 4. user & ad交互特征
# 其它
# 这题能做的特征工程应该还是挺多的(看起来训练集和测试集是按时间划分的)，比如:
# 对每个user，统计历史点击频次/以及点击的时间偏好等等，注意不要穿越、user对不同广告的偏好、用户x广告or用户x类别等交叉特征
# 甚至是用户行为序列特征提取emb等等

# 准备训练数据
LABEL = 'is_click'
feats = [f for f in data.columns if f not in [LABEL, "exposure_time", "prediction"]]
df_train = data[~data[LABEL].isna()].copy()
df_test = data[data[LABEL].isna()].copy()

# 模型参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 1000
}

# 5折交叉验证
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储OOF预测结果
oof_preds = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))
auc_scores = []

# 开始5折交叉验证
for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train[LABEL])):
    print(f"\n--- Fold {fold + 1} ---")
    X_tr, X_val = df_train[feats].iloc[train_idx], df_train[feats].iloc[val_idx]
    y_tr, y_val = df_train[LABEL].iloc[train_idx], df_train[LABEL].iloc[val_idx]
    
    model = LGBMClassifier(**params)
    
    # 模型训练
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    # 验证集预测
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    # 计算AUC
    auc = roc_auc_score(y_val, val_pred)
    auc_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.5f}")
    
    # 测试集预测
    test_pred = model.predict_proba(df_test[feats])[:, 1]
    test_preds += test_pred / n_splits

# 输出结果
print(f"\n平均AUC: {np.mean(auc_scores):.5f}")
print(f"OOF预测结果 shape: {oof_preds.shape}")
print(f"测试集预测结果 shape: {test_preds.shape}")

# 准备提交文件
submission = df_test[['user_id', 'ad_id', 'exposure_time']].copy()
submission['prediction'] = test_preds
submission = submission.reset_index(drop=True)

# 保存提交文件
submission_file = '/work/submit_3.csv'
submission.to_csv(submission_file, index=False)
print(f'Submission file saved to {submission_file}')

# 检查提交文件格式
print(f"Test data rows: {len(df_test)}, Submission rows: {len(submission)}")
print('\nSubmission head:')
print(submission.head())