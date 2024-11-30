import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# 读入数据
data = pd.read_excel(r'2000 data after labeling.xlsx')

# 分离目标变量和特征变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义目标函数
def rf_cv(n_estimators, learning_rate, max_depth):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        random_state=0
    )
    cv_results = []
    for train_idx, val_idx in kfold.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
        r2 = r2_score(y_cv_val, y_cv_pred)
        cv_results.append((rmse, r2))
    avg_rmse = np.mean([res[0] for res in cv_results])
    return -avg_rmse  # Bayesian Optimization maximizes the objective function

# 建立贝叶斯优化对象
rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (10, 200), 'learning_rate': (0.01, 0.999), 'max_depth': (1, 10)}
)
# 开始优化
rf_bo.maximize(n_iter=50)

# 使用最佳参数训练模型
best_params = rf_bo.max['params']
best_gbr = xgb.XGBRegressor(
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    random_state=0
)
best_gbr.fit(X_train, y_train)

# 预测测试集
y_pred = best_gbr.predict(X_test)
# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test))

# 输出评估指标
print("Best Model R²: ", best_gbr.score(X_test, y_test))
print("RMSE: ", rmse)
print("MSE: ", mse)
print("MAE: ", mae)
print("MAPE: ", mape)
print("Best Parameters: ", rf_bo.max)

# 计算5折交叉验证的RMSE和R²
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
rmse_list = []
r2_list = []

for train_idx, val_idx in kfold.split(X_train):
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    best_gbr.fit(X_cv_train, y_cv_train)
    y_cv_pred = best_gbr.predict(X_cv_val)
    rmse_list.append(np.sqrt(mean_squared_error(y_cv_val, y_cv_pred)))
    r2_list.append(r2_score(y_cv_val, y_cv_pred))

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), rmse_list, marker='o', label='RMSE')
plt.plot(range(1, 6), r2_list, marker='o', label='R²')
plt.title("5-Fold Cross-Validation RMSE and R²")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.xticks(range(1, 6))
plt.legend()
plt.grid()
plt.show()
