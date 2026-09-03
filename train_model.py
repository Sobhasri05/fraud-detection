import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Create FAKE transaction data (1000 rows)
np.random.seed(42)
n_samples = 1000

# Features: amount (10 to 1000), location (0=NY, 1=CA, 2=TX), hour_of_day (0-23)
amounts = np.random.uniform(10, 1000, n_samples)
locations = np.random.choice([0, 1, 2], n_samples)
hours = np.random.randint(0, 24, n_samples)

# Create a fake "fraud" label: 
# If amount > 800 AND hour > 22, it's fraud (80% chance). Otherwise, 5% chance.
fraud = []
for i in range(n_samples):
    if amounts[i] > 800 and hours[i] > 22:
        fraud.append(1 if np.random.random() < 0.8 else 0)
    else:
        fraud.append(1 if np.random.random() < 0.05 else 0)

# Make a DataFrame
data = pd.DataFrame({
    'amount': amounts,
    'location': locations,
    'hour': hours,
    'fraud': fraud
})

# Step 2: Preprocess (convert location to dummy variables)
X = pd.get_dummies(data[['amount', 'location', 'hour']], columns=['location'], drop_first=True)
y = data['fraud']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the model AND the column names (important for later)
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✅ Model trained and saved as 'fraud_model.pkl'")
print(f"✅ Model accuracy: {model.score(X_test, y_test):.2%}")
print(f"✅ Feature columns saved: {X.columns.tolist()}")