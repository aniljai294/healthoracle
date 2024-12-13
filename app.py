from flask import Flask, render_template, request
import joblib
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load the model
try:
    diabetes_model = joblib.load('./models/diabetes_model.pkl')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    diabetes_model = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    bmi_value = None
    glucose_value = None
    glucose_percent = None  # Added for normalized progress bar

    if request.method == "POST":
        try:
            # Retrieve data from form
            age = float(request.form["age"])
            bmi_value = float(request.form["bmi"])
            glucose_value = float(request.form["glucose_level"])

            # Validate input
            if age < 0:
                raise ValueError("Age cannot be negative.")
            if bmi_value < 18 or bmi_value > 35:
                raise ValueError("BMI must be between 18 and 35.")
            if glucose_value < 70 or glucose_value > 200:
                raise ValueError("Glucose level must be between 70 and 200.")
            
            # Normalize glucose for progress bar
            glucose_percent = (glucose_value / 200) * 100

            # Model prediction
            prediction_raw = diabetes_model.predict([[age, bmi_value, glucose_value]])[0]
            prediction = "No Risk" if prediction_raw == 0 else "At Risk"

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html", 
        prediction=prediction, 
        bmi_value=bmi_value, 
        glucose_value=glucose_value, 
        glucose_percent=glucose_percent  # Pass the normalized value
    )

if __name__ == "__main__":
    app.run(debug=True)
