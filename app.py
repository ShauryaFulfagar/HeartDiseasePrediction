# Flask for connecting everythong with the frontend
from flask import *

# import all modules
import numpy as np

# import pandas for the csv dataset reading
import pandas as pd

# import split so that we can use the same data in two different ways for train and for testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Get the dataset using pandas
heart_data = pd.read_csv('./data.csv')

# Classify the split data into 2 variables 'X' and 'Y'
X = heart_data.drop(axis=1, columns="target")
Y = heart_data['target']

# Split the dataset into train and test with an 80% uneven distribution
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)

# Define the model by referrring to one instance of the logistic regression model
model = LogisticRegression()

# Train the logistic regression model
model.fit(X_train, Y_train)

# AccuracyScore on train data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# AccuracyScore on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol =int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = int(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal)
        data = np.array(input_data)
        data_reshaped = data.reshape(1, -1)

        prediction = model.predict(data_reshaped)
        return render_template('result.html', prediction=prediction)

    else:
        return render_template('result.html', prediction=0)


if __name__ == '__main__':
    app.run(debug=True)
