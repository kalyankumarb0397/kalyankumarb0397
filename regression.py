import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv(r"D:\Data Science\VS CODE\polyninomial regression\emp_sal.csv")
X = data.iloc[:, 1:2].values  # Assuming experience is in the second column
y = data.iloc[:, 2].values  # Assuming salary is in the third column

# Streamlit App Layout
st.title("Employee Salary Prediction Based on Experience")

# Display the dataset
st.subheader("Dataset")
st.dataframe(data)

# Pair plot of the dataset showing original salary
st.subheader("Pair Plot of the Dataset")
pair_plot_fig = sns.pairplot(data, diag_kind='kde')
pair_plot_fig.fig.suptitle("Pair Plot of Employee Experience and Salary", y=1.02)

# Display the pair plot in Streamlit
st.pyplot(pair_plot_fig)

# Input for experience
experience_input = st.number_input("Enter the Years of Experience for Prediction:", min_value=0.0, max_value=30.0, value=5.0)

# Slider for KNN Neighbors
n_neighbors = st.slider("Select Number of Neighbors for KNN Regression:", min_value=1, max_value=10, value=4)

# Separate sliders for Polynomial Degree and SVR Degree
poly_degree = st.slider("Select the Degree of Polynomial for Polynomial Regression:", min_value=1, max_value=6, value=3)
svr_degree = st.slider("Select the Degree of Polynomial for SVR (for Polynomial Kernel only):", min_value=1, max_value=6, value=3)

# Prepare a DataFrame to store results
predictions = {
    'Model': [],
    'Prediction': [],
    'R² Score': [],
    'Mean Squared Error': []  # Changed back to MSE
}

# Function to plot predictions for each model
def plot_model_predictions(model_name, model, input_value, r2, mse):
    # Predict using the model
    predictions_values = model.predict(X)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Salary', s=50)
    plt.scatter(X, predictions_values, color='red', label='Predicted Salary', s=50)
    plt.scatter([[input_value]], model.predict([[input_value]]), color='green', s=100, label='Input Prediction', marker='x')
    plt.title(f"{model_name} Prediction\nR² Score: {r2:.2f} | MSE: {mse:.2f}")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_r2 = r2_score(y, lin_reg.predict(X))
lin_mse = mean_squared_error(y, lin_reg.predict(X))
predictions['Model'].append('Linear Regression')
predictions['Prediction'].append(f"${lin_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(lin_r2)
predictions['Mean Squared Error'].append(lin_mse)
plot_model_predictions('Linear Regression', lin_reg, experience_input, lin_r2, lin_mse)

# Ridge Regression
ridge_reg = Ridge()
ridge_reg.fit(X, y)
ridge_r2 = r2_score(y, ridge_reg.predict(X))
ridge_mse = mean_squared_error(y, ridge_reg.predict(X))
predictions['Model'].append('Ridge Regression')
predictions['Prediction'].append(f"${ridge_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(ridge_r2)
predictions['Mean Squared Error'].append(ridge_mse)
plot_model_predictions('Ridge Regression', ridge_reg, experience_input, ridge_r2, ridge_mse)

# Lasso Regression
lasso_reg = Lasso()
lasso_reg.fit(X, y)
lasso_r2 = r2_score(y, lasso_reg.predict(X))
lasso_mse = mean_squared_error(y, lasso_reg.predict(X))
predictions['Model'].append('Lasso Regression')
predictions['Prediction'].append(f"${lasso_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(lasso_r2)
predictions['Mean Squared Error'].append(lasso_mse)
plot_model_predictions('Lasso Regression', lasso_reg, experience_input, lasso_r2, lasso_mse)

# SVR with all kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svr_reg = SVR(kernel=kernel, degree=svr_degree if kernel == 'poly' else 3)  # Use degree only for poly kernel
    svr_reg.fit(X, y)
    svr_r2 = r2_score(y, svr_reg.predict(X))
    svr_mse = mean_squared_error(y, svr_reg.predict(X))
    predictions['Model'].append(f'SVR ({kernel})')
    predictions['Prediction'].append(f"${svr_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
    predictions['R² Score'].append(svr_r2)
    predictions['Mean Squared Error'].append(svr_mse)
    plot_model_predictions(f'SVR ({kernel})', svr_reg, experience_input, svr_r2, svr_mse)

# KNN Regression with selected number of neighbors
knn_reg = KNeighborsRegressor(n_neighbors=n_neighbors)
knn_reg.fit(X, y)
knn_r2 = r2_score(y, knn_reg.predict(X))
knn_mse = mean_squared_error(y, knn_reg.predict(X))
predictions['Model'].append('KNN Regression')
predictions['Prediction'].append(f"${knn_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(knn_r2)
predictions['Mean Squared Error'].append(knn_mse)
plot_model_predictions('KNN Regression', knn_reg, experience_input, knn_r2, knn_mse)

# Decision Tree Regression
dtree_reg = DecisionTreeRegressor(random_state=0)
dtree_reg.fit(X, y)
dtree_r2 = r2_score(y, dtree_reg.predict(X))
dtree_mse = mean_squared_error(y, dtree_reg.predict(X))
predictions['Model'].append('Decision Tree Regression')
predictions['Prediction'].append(f"${dtree_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(dtree_r2)
predictions['Mean Squared Error'].append(dtree_mse)
plot_model_predictions('Decision Tree Regression', dtree_reg, experience_input, dtree_r2, dtree_mse)

# Random Forest Regression
rf_reg = RandomForestRegressor()
rf_reg.fit(X, y)
rf_r2 = r2_score(y, rf_reg.predict(X))
rf_mse = mean_squared_error(y, rf_reg.predict(X))
predictions['Model'].append('Random Forest Regression')
predictions['Prediction'].append(f"${rf_reg.predict([[experience_input]])[0]:,.2f}")  # Format as currency
predictions['R² Score'].append(rf_r2)
predictions['Mean Squared Error'].append(rf_mse)
plot_model_predictions('Random Forest Regression', rf_reg, experience_input, rf_r2, rf_mse)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Display prediction and accuracy results in a single table
st.subheader("Predicted Salary, Model Accuracy, and Mean Squared Error")
st.dataframe(predictions_df)
