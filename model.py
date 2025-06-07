import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('places.csv')

# Print columns to debug
print("Columns in the dataset:", data.columns)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Print the first few rows to inspect the data
print(data.head())

# Prepare features and target
X = data[['latitude', 'longitude', 'Popularity Index']]
y = (data['Popularity Index'] < 0.51).astype(float)  # Label places with less than 30 visits as '1'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Random Forest model trained and saved as model.pkl")
