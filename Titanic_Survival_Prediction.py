import pandas as pd

# Load the data
data = pd.read_csv('encoded_train.csv')

# Display the first few rows of the data
data.head()

from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy

#user_data = [[PassengerId, Pclass, Age, # of Siblings + Spouse, # of Parents + Children, Male (1) or Female (0), Embarked from Queenstown, Embarked from Southampton]]
user_data = [[3, 1, 22.0, 1, 0, 7.25, 0, 0, 1]]

# Use the trained model to make a prediction
prediction = clf.predict(user_data)

# Output the prediction (1 for survived, 0 for not survived)
print(prediction[0])
