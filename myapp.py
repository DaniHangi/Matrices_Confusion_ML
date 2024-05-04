import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#appel du modele deja entraine dans le fichier model.py
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_feature = [eval(x) for x in request.form.values()]
    final_feature = np.array(int_feature).reshape(1, -1)  # Reshape to (1, 1) or (1, n) depending on model needs
    prediction = model.predict(final_feature)
    
    output = np.round(prediction[0], 2)

    return render_template('index.html', prediction_text="Weight prediction : {} kg's".format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)