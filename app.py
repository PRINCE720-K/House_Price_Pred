# Libraries importing
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load Saved Model, Scaler, and Feature Names
model = joblib.load('house_price_model.joblib')
sc = joblib.load('scaler.joblib')
feature_names = joblib.load('features.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        area = int(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = int(request.form['mainroad'])
        guestroom = int(request.form['guestroom'])
        basement = int(request.form['basement'])
        hotwaterheating = int(request.form['hotwaterheating'])
        airconditioning = int(request.form['airconditioning'])
        parking = int(request.form['parking'])
        prefarea = int(request.form['prefarea'])
        furnishingstatus = int(request.form['furnishingstatus'])

        # Create DataFrame (VERY IMPORTANT)
        input_dict = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus
}

        input_data = pd.DataFrame([input_dict])

        # Scale using loaded scaler
        scaled_data = sc.transform(input_data)

        # Predict Price
        prediction = model.predict(scaled_data)[0]

        return render_template('output.html',
                               price=round(prediction,2))

    return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)