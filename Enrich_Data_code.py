import pandas as pd
import random
import lightgbm as lgb

# Read the original dataset
original_data = pd.read_excel(r"1000 raw data ")

# Separate independent variables and dependent variable
X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]

# Define and train the model
model = lgb.LGBMRegressor(n_estimators=87, learning_rate=0.04968, max_depth=9)
model.fit(X, y)

# Reverse design
n_solutions = 5000  # Target number of solutions
X_range = [(0.001, 1), (1, 1e15), (1, 6)]  # Range of independent variables
X_names = X.columns  # Names of independent variables
target_y_threshold = random.uniform(10, 25)  # Threshold for target y value
target_y_high = 40  # Upper limit for target y value
solutions = []  # Store solutions

# Read the saved solutions
saved_solutions = pd.read_excel(r"Expanded to 100,000 pieces of data.xlsx")

max_no_solution_count = 1000  # Maximum number of iterations without finding a solution
no_solution_count = 0  # Counter for iterations without finding a solution

while len(solutions) < n_solutions and no_solution_count < max_no_solution_count:
    current_solution = X.iloc[0].copy()  # Copy the current solution
    for i in range(len(X_names)):
        new_value = random.uniform(X_range[i][0], X_range[i][1])  # Generate random value within range
        current_solution[X_names[i]] = new_value  # Add the new value to the current solution

    # Check if the current solution meets the requirements
    pred_y = model.predict(current_solution.values.reshape(1, -1))[0]

    if pred_y > target_y_threshold and pred_y < target_y_high and all(~X.equals(current_solution) for X in solutions):
        current_solution = current_solution.to_frame().T  # Convert Series to DataFrame
        current_solution.insert(len(current_solution.columns), "eta(%)", pred_y)  # Add target y value to the last column
        solutions.append(current_solution)
        no_solution_count = 0  # Reset the counter
    else:
        no_solution_count += 1  # Increment the counter

    # Update target y value
    if no_solution_count >= max_no_solution_count:
        print("Current target y value:", target_y_threshold)

# Add generated solutions to the saved solutions
for solution in solutions:
    saved_solutions = saved_solutions.append(solution, ignore_index=True)

# Save the new solutions to a file
saved_solutions.to_excel(r"Expanded to 100,000 pieces of data.xlsx", index=False)

# Add new feasible solutions to the original dataset
expanded_data = original_data.append(solutions, ignore_index=True)


