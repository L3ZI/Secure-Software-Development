from flask import Flask, render_template, request
import pickle
import numpy as np
import string
import random

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/predictor.pickle', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_password', methods=['POST'])
def generate_password():
    # Retrieve form data
    length = int(request.form.get("length"))
    include_uppercase = bool(request.form.get("include_uppercase"))
    include_lowercase = bool(request.form.get("include_lowercase"))
    include_digits = bool(request.form.get("include_digits"))
    include_special_chars = bool(request.form.get("include_special_chars"))

    characters = ""
    # Generate character pool based on form inputs
    if include_uppercase:
        characters += string.ascii_uppercase
    if include_lowercase:
        characters += string.ascii_lowercase
    if include_digits:
        characters += string.digits
    if include_special_chars:
        characters += string.punctuation

    # Generate a random password using the character pool
    password = generate_random_password(length, characters)

    # Convert password to numerical features
    password_vector = np.array([password])
    password_vector = np.zeros((password_vector.shape[0], 124))

    # Make prediction using the trained model
    prediction = model.predict(password_vector)

    # Determine the password strength based on the prediction
    if prediction == 0:
        strength = 'Weak'
    elif prediction == 1:
        strength = 'Medium'
    else:
        strength = 'Strong'

    return render_template('index.html', password=password, strength=strength)

@app.route('/check_password', methods=['POST'])
def check_password():
    # Retrieve form data
    password = request.form['password']

    # Convert password to numerical features
    password_vector = np.array([password])
    password_vector = np.zeros((password_vector.shape[0], 124))

    # Make prediction using the trained model
    prediction = model.predict(password_vector)

    # Determine the password strength based on the prediction
    if prediction == 0:
        strength = 'Weak'
    elif prediction == 1:
        strength = 'Medium'
    else:
        strength = 'Strong'

    return render_template('index.html', password=password, strength=strength)

def generate_random_password(length, characters):
    # Generate a random password by selecting characters from the pool
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

if __name__ == '__main__':
    app.run(debug=True)
