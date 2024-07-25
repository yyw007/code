import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def optimize_model(X, y):
    xgb_model = xgb.XGBRegressor(random_state=0)
    search_spaces = {
        'n_estimators': Integer(100, 200),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 1.0, 'log-uniform')
    }
    bayes_cv = BayesSearchCV(xgb_model, search_spaces, n_iter=32, scoring='r2', cv=3, n_jobs=-1, random_state=0)
    bayes_cv.fit(X, y)
    return bayes_cv.best_estimator_, bayes_cv.best_score_

def train_model(X, y, best_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def add_and_retrain(model, new_data_path, X, y):
    new_X, new_y = load_and_prepare_data(new_data_path)
    X = pd.concat([X, new_X])
    y = pd.concat([y, new_y])
    model.fit(X, y)
    accuracy = model.score(X, y)
    return model, X, y, accuracy

def generate_solutions(model, X_ranges, target_y_threshold, target_y_high, n_solutions):
    X_names = list(X_ranges.keys())
    solutions = []
    max_no_solution_count = 100000
    no_solution_count = 0
    round_ranges = {name: [] for name in X_names}

    while len(solutions) < n_solutions and no_solution_count < max_no_solution_count:
        current_solution = {name: random.uniform(*X_ranges[name]) for name in X_names}
        pred_y = model.predict(pd.DataFrame([current_solution]))[0]
        for name in X_names:
            round_ranges[name].append(current_solution[name])

        if target_y_threshold <= pred_y <= target_y_high:
            current_solution['PCE (%)'] = pred_y
            solutions.append(current_solution)
            no_solution_count = 0
        else:
            no_solution_count += 1

    solutions_df = pd.DataFrame(solutions)
    return solutions_df.sort_values('PCE (%)', ascending=False).head(n_solutions), round_ranges


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_ranges(history_ranges, X_ranges):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust for two rows and three columns
    axs = axs.ravel()  # Flatten the axes array for easy iteration
    colors = ['#8ecae6', '#95d5b2', '#f4acb7', 'red']  # Colors for each round

    for i, key in enumerate(history_ranges.keys()):
        # Prepare the data
        data = []
        for j, ranges in enumerate(history_ranges[key]):
            data.extend([{'value': value, 'Round': f'Round {j}'} for value in ranges])
        df = pd.DataFrame(data)

        # Draw the histograms with KDE for each round
        unique_rounds = df['Round'].unique()
        for idx, round_num in enumerate(unique_rounds):
            sns.histplot(df[df['Round'] == round_num]['value'], ax=axs[i],
                         color=colors[idx % len(colors)],
                         label=f'{round_num}', kde=True, element="step", stat="density", alpha=0.5)

        axs[i].set_title(f'Distribution of {key}')
        axs[i].set_xlabel(key)
        axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()

    plt.show()

# Example usage of your plotting function here, assuming 'history_ranges' and 'X_ranges' are properly defined
# plot_ranges(history_ranges, X_ranges)


# Initialize and run the main program logic
original_data_path = r"Raw data.xlsx"
X, y = load_and_prepare_data(original_data_path)

# Using Bayesian optimization to find the best parameters
best_model, best_score = optimize_model(X, y)
print(f"Best model parameters found: {best_model.get_params()}")
print(f"Best R^2 score from optimization: {best_score:.4f}")

# Train the model with the best parameters
model, initial_accuracy = train_model(X, y, best_model.get_params())
print(f"Initial model accuracy (R^2): {initial_accuracy:.4f}")

# Parameter ranges and initial range sampling
X_ranges = {
    'Temperature': (125, 175),
    'Speed': (100, 300),
    'Spray Flow': (2000, 5000),
    'Plamsa Height': (0.8, 1.2),
    'Plasma Gas Flow': (15, 35),
    'Plasma DC': (25, 100)
}
n_solutions = 10  # Initial number of solutions
_, initial_ranges = generate_solutions(model, X_ranges, target_y_threshold=13, target_y_high=20, n_solutions=n_solutions )
history_ranges = {name: [initial_ranges[name]] for name in X_ranges.keys()}

new_data_paths = [r"New Data 11.xlsx", r"New Data 12.xlsx", r"New Data 13.xlsx"]
for new_data_path in new_data_paths:
    if n_solutions > 1:
        n_solutions = max(1, n_solutions - 2)
    model, X, y, accuracy = add_and_retrain(model, new_data_path, X, y)
    print(f"Model accuracy after adding {new_data_path} (R^2): {accuracy:.4f}")
    top_solutions, round_ranges = generate_solutions(model, X_ranges, target_y_threshold=13, target_y_high=20, n_solutions=n_solutions)
    for name in X_ranges.keys():
        history_ranges[name].append(round_ranges[name])
        X_ranges[name] = (min(round_ranges[name]), max(round_ranges[name]))

plot_ranges(history_ranges, X_ranges)
