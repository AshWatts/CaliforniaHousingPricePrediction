import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print("Received data:", data)

#     input_features = np.array(list(data.values())).reshape(1, -1)
#     print("Input shape:", input_features.shape)

#     # Check if input matches StandardScaler expectations
#     if input_features.shape[1] != scalar.n_features_in_:
#         return jsonify({'error': f'Expected {scalar.n_features_in_} features, but got {input_features.shape[1]}'}), 400

#     new_data = scalar.transform(input_features)
#     output = regmodel.predict(new_data)
#     print("Prediction:", output[0])

#     return jsonify({'prediction': output[0]})

if __name__ == '__main__':
    app.run(debug=True)
