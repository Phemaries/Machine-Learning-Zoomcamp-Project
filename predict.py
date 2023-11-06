from flask import Flask
from flask import request
from flask import jsonify
import sklearn
import pickle
import pandas as pd

app = Flask("concern")
dv, model = pickle.load(open("model.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    concern = y_pred >= 0.5

    result = {
        'monkeypox_tendecy': float(y_pred),
        'concern': bool(concern)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
