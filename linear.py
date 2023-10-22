# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from the CSV file
data = pd.read_csv('Salary_dataset.csv')

# Extract features (X) and target variable (y) from the data
X = data[['YearsExperience']]
y = data['Salary']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on all data points
predictions = model.predict(X)

# Calculate mean squared error and R-squared on the entire dataset
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)

# Visualize the linear regression line and all data points
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Years of Experience')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients and intercept of the linear regression model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
