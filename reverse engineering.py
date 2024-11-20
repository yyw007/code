import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 读取原始数据集
data_path = r"1000 raw data"
original_data = pd.read_excel(data_path)

# 将自变量和因变量分开
X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 定义模型并进行训练
model = xgb.XGBRegressor(n_estimators=124, learning_rate=0.47053079467824593, max_depth=5, random_state=0)
model.fit(X_train, y_train)

# 反向设计参数
n_solutions =10  # 目标数量（可自由修改）
X_range = [(0.001, 1), (1, 1e15), (1, 6)]  # 自变量范围
X_names = X.columns  # 自变量名称
target_y_threshold = 22.5  # 目标y值的阈值
target_y_high = 35  #
solutions = []  # 存储解决方案

# 读取已保存的解决方案
saved_solutions_path = r"Create a new excel file to save the results"
saved_solutions = pd.read_excel(saved_solutions_path)

max_y = float('-inf')  # 记录当前找到的最大y值

max_no_solution_count = 100000  # 最大连续没有解决方案的次数
no_solution_count = 0  # 连续没有解决方案的次数
no_solution_count_high = 0  # 连续没有解决方案的次数的高阈值

while len(solutions) < n_solutions and no_solution_count < max_no_solution_count:
    current_solution = X.iloc[0].copy()  # 复制一份当前解
    for i in range(len(X_names)):
        new_value = random.uniform(X_range[i][0], X_range[i][1])  # 生成在范围内的随机值
        current_solution[X_names[i]] = new_value  # 将新生成的值按照列名对应添加到当前解中

    # 判断当前解是否满足要求
    pred_y = model.predict(current_solution.values.reshape(1, -1))[0]

    if target_y_threshold <= pred_y <= target_y_high:
        current_solution = pd.DataFrame(current_solution).T  # 将Series转换为DataFrame
        current_solution['eta(%)'] = pred_y  # 添加目标y值到最后一列
        solutions.append(current_solution)
        no_solution_count = 0  # 重置连续没有解决方案的次数

        # 更新最大y值和对应的解决方案
        if pred_y > max_y:
            max_y = pred_y

    else:
        no_solution_count += 1  # 连续没有解决方案的次数加1
        no_solution_count_high += 1  # 连续没有解决方案的次数的高阈值加1

    # 每100次连续无解，将目标y值的最高值减小0.05
    if no_solution_count_high >= 1000:
        target_y_high -= 0.05
        no_solution_count_high = 0

# 对解决方案按预测y值排序
solutions = pd.concat(solutions).sort_values('eta(%)', ascending=False).reset_index(drop=True)
top_solutions = solutions.head(n_solutions)

# 保存最优解决方案
top_solutions.to_excel(r"Create a new excel file to save the results", index=False)

# 输出参数范围和最优参数组合
print("Parameter ranges for the top  solutions:")
print(top_solutions.describe().loc[['min', 'max']])
print("\nBest parameter combination:")
print(top_solutions.iloc[0])
