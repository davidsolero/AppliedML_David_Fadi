import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


# Load the datasets
data = pd.read_excel('Feature_Track.xlsx')

# Drop the 'subject' column or any other non-numeric identifier columns
data.drop(columns=['subject'], inplace=True, errors='ignore')

# Select only the features and the new target variable 'frustration'
selected_features = [
   "dummy",
   "straight",
   "traffic",
   "hurry",
   "habituation"
]

# Filter data to only use the selected features and the target variable
data = data[selected_features + ['surprise']]

# Handle missing values (filling with median for numeric columns)
data.fillna(data.median(), inplace=True)

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data)

# Feature-target split
X = data.drop('surprise', axis=1)
y = data['surprise']

# Map target variable for classification (1 = yes, 2 = no)
y = y.map({1: 'yes', 2: 'no'})

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr', random_state=42),
    'SVM': SVC(kernel='linear', decision_function_shape='ovr', random_state=42, probability=True),  # Enable probability
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Classification Report
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Yes', 'No'],
                yticklabels=['Yes', 'No'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Feature Importance (only for Random Forest)
    if model_name == 'Random Forest':
        feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
        feature_importances.sort_values(by='importance', ascending=False).head(10).plot(kind='barh')
        plt.title(f'{model_name} - Top 10 Feature Importances')
        plt.show()

# ROC-AUC for binary classification
y_train_bin = label_binarize(y_train, classes=['yes', 'no'])
y_test_bin = label_binarize(y_test, classes=['yes', 'no'])

# Calculate ROC-AUC score for each model
for model_name, model in models.items():
    if model_name in ['Logistic Regression', 'SVM', 'KNN', 'Random Forest']:
        # Get probabilities for the positive class ('yes')
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get the probability for the 'yes' class (positive class)

        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y_test_bin, y_pred_prob)
        print(f"{model_name} - ROC-AUC Score: {roc_auc:.2f}")