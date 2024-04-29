import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load the small dataset with labels
labeled_data = pd.read_excel(r'1000 raw data.xlsx')
X_labeled = labeled_data.iloc[:, :3].values
y_labeled = labeled_data.iloc[:, 3].values

# Load a large dataset without labels
unlabeled_data = pd.read_excel(r'Expanded to 100,000 pieces of data.xlsx')
X_unlabeled = unlabeled_data.values

# Define a function to calculate the distance to the hyperplane
def acquisition(w, bias, data):
    result = np.abs(np.dot(data, w) + bias) / np.linalg.norm(w)
    return result

# Train the initial model using the small labeled dataset
labeled_model = KNeighborsRegressor(n_neighbors=8)
labeled_model.fit(X_labeled, y_labeled)

# Define the SVR model
model = SVR(kernel='linear', C=1)
model.fit(X_labeled, y_labeled)

# Use SVR active learning method to select high-quality data
n_iter = 800
for i in range(n_iter):
    # Get the hyperplane vector and intercept
    w = model.coef_[0]
    bias = model.intercept_[0]

    # Adjust the shape of the hyperplane vector
    w = np.resize(w, (X_unlabeled.shape[1],))

    # Randomly select data
    X_acq = X_unlabeled[np.random.choice(X_unlabeled.shape[0], size=5000, replace=False)]

    # Distance to the hyperplane
    acq = acquisition(w, bias, X_acq)

    # Select the data point with the smallest distance
    x_next = X_acq[np.argmin(acq)]

    # Calculate the label value for the point in the small labeled dataset
    y_next = labeled_model.predict(x_next[:3].reshape(1, -1))  # Modified to include only the first 3 features

    # Update the dataset
    X_labeled = np.append(X_labeled, x_next[:3].reshape(1, -1), axis=0)  # Modified to include only the first 3 features
    y_labeled = np.append(y_labeled, y_next)
    X_unlabeled = np.delete(X_unlabeled, np.where((X_unlabeled == x_next).all(axis=1)), axis=0)

    # Update the hyperplane
    model.fit(X_labeled, y_labeled)

# Output the selected high-quality data
df3 = pd.DataFrame(X_labeled, columns=unlabeled_data.columns[:3])
df4 = pd.DataFrame(y_labeled, columns=[labeled_data.columns[3]])
df5 = pd.concat([df3, df4], axis=1)
df5.to_excel(r'2000 data after labeling.xlsx', index=False)
print(df5)
