from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Update your model path and load the logistic regression model
diabetes_model_path = 'model04.pkl'  # Replace with your model path
with open(diabetes_model_path, 'rb') as file:
    diabetes_model = pickle.load(file)

app = Flask(__name__)

# Define the route for diabetes prediction
@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    # Extract data from form for diabetes prediction
    pregnant = request.form.get('Pregnant', type=int)
    insulin = request.form.get('Insulin', type=float)
    bmi = request.form.get('BMI', type=float)
    age = request.form.get('Age', type=int)
    glucose = request.form.get('Glucose', type=float)
    bp = request.form.get('Blood Pressure', type=float)
    pedigree = request.form.get('Diabetes Pedigree Function', type=float)

    # Validate inputs
    if (pregnant is None or insulin is None or bmi is None or
        age is None or glucose is None or bp is None or pedigree is None):
        return render_template('diabetes.html', prediction_text='Invalid input. Please provide all fields.')

    # Create numpy array for prediction
    final_features = np.array([[pregnant, insulin, bmi, age,
                                glucose, bp, pedigree]])

    # Make diabetes prediction
    predicted_diabetes = diabetes_model.predict(final_features)[0]

    # Prepare the result message
    if predicted_diabetes == 1:
        result_text = 'The patient is predicted to have diabetes.'
    else:
        result_text = 'The patient is predicted not to have diabetes.'

    return render_template('diabetes.html', prediction_text=result_text)

# Define another route for the main page
@app.route('/')
def main_page():
    return render_template('diabetes.html')

if __name__ == "__main__":
    app.run(debug=True)