import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Start the H2O service
h2o.init()

# Load the data
# Make sure to replace the file path with your own
data = pd.read_excel(r"2000 data after labeling.xlsx")

# Convert pandas DataFrame to H2OFrame
hf = h2o.H2OFrame(data)

# Split the data into training and test sets
train, test = hf.split_frame(ratios=[.8])

# Set predictors and response variables
predictors = ['total defect density', 'beta_0', 'gap']
response = 'eta(%)'

# Use project_name parameter to avoid running the same models multiple times
project_name = "my_unique_automl_project"

# Run AutoML
automl = H2OAutoML(max_models=20, seed=1, project_name=project_name, exclude_algos=["StackedEnsemble"])
automl.train(x=predictors, y=response, training_frame=train)

# Get the IDs of all models
model_ids = list(automl.leaderboard['model_id'].as_data_frame().iloc[:,0])
for m_id in model_ids:
    # Get the performance metrics for each model
    model = h2o.get_model(m_id)
    performance = model.model_performance(test)
    print(f'Performance metrics for model {m_id}:')
    print(f'MAE: {performance.mae()}')
    print(f'MSE: {performance.mse()}')
    print(f'RMSE: {performance.rmse()}')
    print(f'R^2: {performance.r2()}')
    print('\n')
