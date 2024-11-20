import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Define the immune system optimization algorithm functions
def objective_function(params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def random_params():
    return {
        'n_estimators': random.randint(50, 150),
        'learning_rate': random.uniform(0.01, 0.5),
        'max_depth': random.randint(3, 10)
    }

# Parameters for the immune algorithm
population_size = 50
max_iterations = 20

# Read the original dataset
original_data = pd.read_excel(r"2000 data after labeling.xlsx")

# Separate independent variables and dependent variable
X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the population
population = [random_params() for _ in range(population_size)]

for iteration in range(max_iterations):
    # Evaluate the population
    evaluated = [(individual, objective_function(individual)) for individual in population]

    # Select the best solutions
    best_individuals = sorted(evaluated, key=lambda x: x[1])[:2]

    # Clone and mutate
    new_population = []
    for best in best_individuals:
        for _ in range(population_size // 2):
            new_individual = best[0].copy()
            # Random mutation
            if random.random() < 0.4:
                new_individual['n_estimators'] = random.randint(50, 150)
            if random.random() < 0.4:
                new_individual['learning_rate'] = random.uniform(0.01, 0.5)
            if random.random() < 0.4:
                new_individual['max_depth'] = random.randint(3, 10)
            new_population.append(new_individual)

    # Replace the population
    population = new_population

# Output the best parameters
best_params = min(population, key=objective_function)

# Train the XGBoost model with the best parameters
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Reverse design
n_solutions = 20  # Target number of solutions
X_range = [(0.001, 1), (1, 1e15), (1, 6)]  # Range for independent variables
X_names = X.columns  # Names of independent variables
target_y_threshold = 21.5  # Threshold for target y value
target_y_high = 35  # Upper limit for target y value
solutions = []  # Store solutions

# Read saved solutions
saved_solutions = pd.read_excel(r"optimization results.xlsx.xlsx")

max_y = float('-inf')  # Record the highest y value found

max_no_solution_count = 10000  # Maximum number of iterations without a solution
no_solution_count = 0  # Counter for iterations without a solution

while len(solutions) < n_solutions and no_solution_count < max_no_solution_count:
    current_solution = X.iloc[0].copy()  # Copy the current solution
    for i in range(len(X_names)):
        new_value = random.uniform(X_range[i][0], X_range[i][1])  # Generate a random value within range
        current_solution[X_names[i]] = new_value  # Add the new value to the current solution

    # Check if the current solution meets the requirements
    pred_y = model.predict(current_solution.values.reshape(1, -1))

    if pred_y > target_y_threshold and target_y_high - 0.5 <= pred_y <= target_y_high and all(~X.equals(current_solution) for X in solutions):
        current_solution = current_solution.to_frame().T  # Convert Series to DataFrame
        current_solution.insert(len(current_solution.columns), "eta(%)", pred_y)  # Add target y value to the last column
        solutions.append(current_solution)
        no_solution_count = 0  # Reset the counter

        # Update the highest y value and corresponding solution
        if pred_y > max_y:
            max_y = pred_y
            max_solution = current_solution.copy()
    else:
        no_solution_count += 1  # Increment the counter

    # Decrease the highest target y value by 0.05 every 100 iterations without a solution
    if no_solution_count_high >= 100:
        target_y_high -= 0.05
        no_solution_count_high = 0

    # Output the current target y value
    if no_solution_count >= max_no_solution_count:
        print("Current target y value:", target_y_threshold)

# Add the solution with the highest y value to the saved solutions
saved_solutions = saved_solutions.append(max_solution, ignore_index=True)

# Save the new solutions to a file
saved_solutions.to_excel(r"optimization results.xlsx", index=False)

# Output results and save solutions
print("Best parameters:", best_params)
print("Number of solutions found:", len(solutions))
print("Highest y value:", max_y)

# Output results
print(f"Optimized model R²: {r2}")
print(f"Optimized model MSE: {mse}")
print(f"Optimized model MAE: {mae}")


