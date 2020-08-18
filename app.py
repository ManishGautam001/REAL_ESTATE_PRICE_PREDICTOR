from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('price_predictor_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def calc():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Home price should be $ {}'.format(output))




if __name__ ==  '__main__':
        app.run(debug=True)    
