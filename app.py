import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle as pkl
import joblib
import sqlite3
import random

# create flask app
app = Flask(__name__, static_url_path='/static')

# load pickle file
model = pkl.load(open("model.pkl","rb"))

# load joblib file
# model = joblib.load("model.joblib")

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    input_features = [x for x in request.form.values()]
    features = [np.array(input_features)]
    
    prediction = model.predict(features)
    predictionList = ['mammal','bird','reptile','fish','amphibian','bug','invertebrate']
    
    return render_template("index.html", prediction_text = "The predicted animal is a {}.".format(predictionList[prediction[0] - 1]))

# Function to fetch data from the SQLite database
def fetch_data_from_database():

    connection = sqlite3.connect("zoo_animals.sqlite")
    
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM zoo_animals")
    
    records = cursor.fetchall()
    random.shuffle(records)
    connection.close()
    return records

@app.route("/database")
def database():
    
    records = fetch_data_from_database()
    # Run predictions for each record
    predictions = []
    for record in records:
        # Extract features from the record (assuming the features are in the same order as the input to the model)
        input_features = record[1:-1]  # Exclude the first column (animal_name) and last column (class_type)
        
        # Convert input_features to a numpy array
        features = np.array([input_features])
        
        # Run prediction using the model
        prediction = model.predict(features)
        predictionList = ['mammal','bird','reptile','fish','amphibian','bug','invertebrate']
        # Append the prediction to the predictions list
        predictions.append(predictionList[prediction[0] - 1])
        # Append the prediction to the predictions list
        #predictions.append(prediction[0])
        
    class_type_dict = {
        1: 'mammal',
        2: 'bird',
        3: 'reptile',
        4: 'fish',
        5: 'amphibian',
        6: 'bug',
        7: 'invertebrate'
    }

    # Render the template with data and predictions
    return render_template("database.html", data=records, predictions=predictions,class_type_dict=class_type_dict)
    
if __name__ == "__main__":
    app.run(debug=True)
