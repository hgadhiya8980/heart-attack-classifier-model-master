import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("heart_attack_classifier_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    
    input_feature = [float(x) for x in request.form.values()]
    feature_value = [np.array(input_feature)]
    
    feature_name = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    
    
    df = pd.DataFrame(feature_value, columns=(feature_name))
    
    output = model.predict(df)
    
    if output == 1:
        result_value = "** Will be coming soon heart attack so carefully checkup family docter **"
    else:
        result_value = "Congresulation! you are safe. lets enjoy your life"

    return render_template("index.html", prediction_text="{}".format(result_value))


if __name__ == "__main__":
    app.run()