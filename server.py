import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#pip install flask
#python3 server.py

app = Flask(__name__)
model = joblib.load('model_files/embedding_model.joblib')
model_tokenizer = joblib.load('model_files/embedding_tokenizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    # model_engine = int_features[1]
    
    test_encoded = model_tokenizer.texts_to_sequences([int_features[0]])
    test_padded = pad_sequences(test_encoded, maxlen=100, padding='post')
    prediction =  model.predict(test_padded).round()[0][0]
    result = "Positive :)" if prediction == 1 else "Negative :("
 
    return render_template('index.html',
                           prediction_text='The rating \
                           of this review is {}'.format(result))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
