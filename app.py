from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import random

app = Flask(__name__)

# Assuming label_encoder is already defined and fitted on all_categories
all_categories = ['lite', 'heavy', 'rain', 'sunny', 'moderate']
label_encoder = LabelEncoder()
label_encoder.fit(all_categories)

# Load the trained model
knn_model = joblib.load('time_pred_model.sav')

def categorize_classes(date_time):
    input_day = date_time.day
    input_hour = date_time.hour
    input_day_of_week = date_time.strftime('%A')

    school = 'lite'
    company = 'lite'
    traffic = 'lite'
    rainy_months = [9, 11, 4]

    if input_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if 6 <= input_hour < 10 or 13 <= input_hour < 16:
            school = 'heavy'

    if input_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if 6 <= input_hour < 10 or 16 <= input_hour < 21:
            company = 'heavy'

    if date_time.month in rainy_months:
        weather = random.choice(['rain', 'sunny'])
    else:
        weather = 'sunny' 

    if input_day_of_week in ['Saturday', 'Sunday']:
        traffic = 'lite'
    elif input_hour in range(9, 16):
        traffic = 'moderate'
    elif input_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] and school == 'heavy' and company == 'heavy':
        traffic = 'heavy'

    return school, company, traffic, weather

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Convert user input to datetime
        input_date_time = pd.to_datetime(input_data['datetime'])

        # Apply categorize_classes
        input_school, input_company, input_traffic, input_weather = categorize_classes(input_date_time)

        # Encode categorical variables
        input_school_encoded = label_encoder.transform([input_school])[0]
        input_company_encoded = label_encoder.transform([input_company])[0]
        input_traffic_encoded = label_encoder.transform([input_traffic])[0]
        input_weather_encoded = label_encoder.transform([input_weather])[0]

        # Create input DataFrame
        input_df = pd.DataFrame({
            'school_encoded': [input_school_encoded],
            'company_encoded': [input_company_encoded],
            'traffic_encoded': [input_traffic_encoded],
            'weather_encoded': [input_weather_encoded],
            'day': [input_date_time.day],
            'hour': [input_date_time.hour]
        })

        # Make prediction
        prediction = knn_model.predict(input_df)
        
        # Return the prediction and additional information as JSON
        response_data = {
            'prediction': prediction.tolist(),
            'school_encoded': input_school_encoded,
            'company_encoded': input_company_encoded,
            'traffic_encoded': input_traffic_encoded,
            'weather_encoded': input_weather_encoded,
            'day': input_date_time.day,
            'hour': input_date_time.hour
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
