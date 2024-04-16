import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, IntegerField, FloatField, TextField, IntegrityError, DatabaseError
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# Database Configuration
DB = connect(os.environ.get('DATABASE_URL', 'sqlite:///predictions.db'))

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# Load Model and Columns Metadata
with open('columns.json', 'r') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# Flask App Configuration
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    if not all(k in obs_dict['observation'] for k in columns):
        return jsonify({'error': 'Missing fields in observation'}), 400

    _id = obs_dict['id']
    observation = obs_dict['observation']
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)

    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}

    try:
        p = Prediction(
            observation_id=_id,
            observation=json.dumps(observation),
            proba=proba
        )
        p.save()
    except IntegrityError:
        DB.rollback()
        existing_proba = Prediction.get(Prediction.observation_id == _id).proba
        return jsonify({'error': 'Observation ID already exists', 'proba': existing_proba}), 409

    return jsonify(response)

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    if not data or 'id' not in data or 'true_class' not in data:
        return jsonify({'error': 'Request must contain id and true_class'}), 400

    try:
        observation_id = int(data['id'])  # Ensuring the ID is an integer
        p = Prediction.get(Prediction.observation_id == observation_id)
    except ValueError:
        return jsonify({'error': 'ID must be an integer'}), 400
    except Prediction.DoesNotExist:
        return jsonify({'error': 'Observation ID does not exist'}), 404

    p.true_class = data['true_class']
    p.save()
    return jsonify(model_to_dict(p))

