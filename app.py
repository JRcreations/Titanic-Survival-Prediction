from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import hashlib
import random

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
        salary = int(request.form['salary'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        gender = int(request.form['gender'])
        embarked = request.form['embarked']
        embarked_q = 1 if embarked == 'Queenstown' else 0
        embarked_s = 1 if embarked == 'Southampton' else 0

        # Convert salary into Pclass (1, 2, or 3) and fare ($)
        if salary >= 80000:
            pclass = 1
            fare = random.uniform(60, 250)
        elif salary >= 50000:
            pclass = 2
            fare = random.uniform(20, 60)
        else:
            pclass = 3
            fare = random.uniform(0, 20)

        # Convert name to a numerical value using hashing
        name_hash = hashlib.md5(name.encode()).hexdigest()
        name_num = int(name_hash, 16)

        # Create user data list
        user_data = [[name_num, pclass, age, sibsp, parch, fare, gender, embarked_q, embarked_s]]

        # Use the trained model to make a prediction
        prediction = clf.predict(user_data)

        # Output the prediction (1 for survived, 0 for not survived)
        if prediction[0] == 1:
            result = 'You survived'
            image_path = 'survived.jpg'
        else:
            result = 'You died'
            image_path = 'dead.jpg'

        return render_template('result.html', result=result, image_path=image_path)
    return render_template('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
