from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import hashlib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and data
clf = RandomForestClassifier(random_state=42)
data = pd.read_csv('encoded_train.csv')
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Define the Flask route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        name = request.form['name']
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        gender = int(request.form['gender'])

        embarked = request.form['embarked']
        embarked_q = 1 if embarked == 'Queenstown' else 0
        embarked_s = 1 if embarked == 'Southampton' else 0

        # Convert name to a numerical value using hashing
        name_hash = hashlib.md5(name.encode()).hexdigest()
        name_num = int(name_hash, 16)

        # Create user data list
        user_data = [[name_num, pclass, age, sibsp, parch, fare, gender, embarked_q, embarked_s]]

        # Use the trained model to make a prediction
        prediction = clf.predict(user_data)

        # Output the prediction (1 for survived, 0 for not survived)
        result = 'Survived' if prediction[0] == 1 else 'Not Survived'
        return render_template('result.html', result=result)
    return render_template('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
