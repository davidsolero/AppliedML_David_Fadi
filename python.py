# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\Dasun\OneDrive\Documentos\GitHub\AppliedML_David_Fadi\PlayerAttributeData.csv'
data = pd.read_csv(file_path)

# Inspect the data
print(data.head())
print(data.info())
print(data.isnull().sum())

# Data preprocessing
# Drop irrelevant columns like Player Names, Flags, or Images
irrelevant_columns = ['ID', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy',  'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Standing tackle', 'Strength', 'Vision', 'Volleys']
data = data.drop(columns=[col for col in irrelevant_columns if col in data.columns])

# Handle missing values (fill or drop rows)
data = data.dropna()  # For simplicity, drop rows with missing values

# Convert all columns to strings to handle mixed types
data = data.astype(str)

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data.drop(columns=['Stamina'])  # Replace 'Position' with your target variable
y = data['Stamina'].astype(float)  # Ensure target variable is continuous

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Support Vector Regressor
svm_model = SVR(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the models
def evaluate_model(name, y_test, y_pred):
    print(f"Model: {name}")
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {name}")
    plt.show()

# Evaluate each model
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)

# Cross-validation
for model, name in zip([rf_model, svm_model], ["Random Forest", "SVM"]):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} Cross-Validation MSE: {-np.mean(scores):.4f} ± {np.std(scores):.4f}")
