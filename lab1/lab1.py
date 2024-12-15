
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('PlayerAttributeData.csv')

# Select relevant attributes
attributes = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure',
            'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving',
                'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy',
                'ID', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',
                'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power',
                'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength',
                'Vision', 'Volleys']
data = data[attributes]

# Define target and features
X = data.drop(columns=['Stamina', 'GK diving',
                'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes','ID','Composure','Ball control',
                        'Composure',
                'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving',
                'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy',
                'ID', 'Interceptions', 'Long passing', 'Long shots', 'Marking',
                'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power',
                'Sliding tackle', 'Vision', 'Volleys' , 'Standing tackle'])  # Features
y = data['Stamina']  # Target variable

# Function to extract numeric values
def extract_numeric(value):
    if isinstance(value, str):
        # Handle arithmetic expressions like "72+12" or "71-13"
        if any(op in value for op in ['+', '-']):
            try:
                # Evaluate the arithmetic expression
                return eval(value)
            except Exception:
                pass

        # Extract numeric part by keeping only digits and dots
        parts = ''.join(c if c.isdigit() or c == '.' else ' ' for c in value).split()
        return float(parts[0]) if parts else np.nan

    # Return non-string values as is
    return value

# Clean string columns in the features
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Cleaning column '{col}'...")
        X[col] = X[col].apply(extract_numeric)

# Clean the target variable
if y.dtype == 'object':
    print("Cleaning target variable 'y'...")
    y = y.apply(extract_numeric)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Machine": SVR(kernel='rbf', C=1, gamma='scale'),
    "Naive Bayes": GaussianNB()  # Note: Naive Bayes is primarily for classification
}

# Evaluate models
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R-squared (R²):", r2_score(y_test, y_pred))

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.title(f"Actual vs Predicted - {name}")
    plt.xlabel("Actual Positioning")
    plt.ylabel("Predicted Positioning")
    plt.show()

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)
