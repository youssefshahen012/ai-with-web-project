from flask import flash, render_template, session
from jinja2.exceptions import TemplateNotFound
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import secrets

app = Flask(__name__)

# Set a secret key for the session
app.secret_key = secrets.token_hex(16)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['breast_cancer_db']
collection = db['user_data']

# ... (rest of your code remains unchanged)


# Set up upload folder for ultrasound images
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class_names = ['benign', 'malignant', 'normal']

segmentor_model = load_model('BreastCancerSegmentor.h5')
classification_model = load_model('valid_classifier.h5')

def prepare_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize pixel values
    return img

def prepare_image2(mask_pred, target_size=(256, 256)):
    img2 = image.load_img(mask_pred, color_mode='grayscale', target_size=target_size)
    img2 = image.img_to_array(img2)
    img2 = img2 / 255.0
    img2 = np.expand_dims(img2, axis=-1)
    return img2

def predict_image(image_path):
    img2 = prepare_image(image_path)
    mask_pred = segmentor_model.predict(np.expand_dims(img2, axis=0))[0][:, :, 0]
    temp_mask_path = 'temp_mask.png'
    plt.imsave(temp_mask_path, mask_pred, cmap='gray')

    img2 = prepare_image2(temp_mask_path)
    class_pred = classification_model.predict(np.expand_dims(img2, axis=0))[0]
    predicted_class = class_names[np.argmax(class_pred)]

    os.remove(temp_mask_path)

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        prediction_result = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        user_data = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'age': request.form['age'],
            'email': request.form['email'],
            'phone_number': request.form['phone_number'],
            'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'prediction_result': prediction_result
        }
        collection.insert_one(user_data)

        return redirect(url_for('result'))

    return 'Invalid file format'

@app.route('/result')
def result():
    latest_data = collection.find_one(sort=[('_id', -1)])
    user_data = {
        'first_name': latest_data.get('first_name', ''),
        'last_name': latest_data.get('last_name', ''),
        'age': latest_data.get('age', ''),
        'email': latest_data.get('email', ''),
        'phone_number': latest_data.get('phone_number', ''),
        'image_path': latest_data.get('image_path', ''),
    }
    prediction_result = latest_data.get('prediction_result', 'Malignant')

    return render_template('result.html', user_data=user_data, prediction_result=prediction_result)

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    age = data.get('age')
    email = data.get('email')
    phone_number = data.get('phone_number')

    user_data = {
        'first_name': first_name,
        'last_name': last_name,
        'age': age,
        'email': email,
        'phone_number': phone_number,
        'image_path': 'path_to_default_image',
        'prediction_result': 'No Prediction Yet'
    }
    result = collection.insert_one(user_data)

    return jsonify({'message': 'User added successfully', 'user_id': str(result.inserted_id)})

# Route to view all users
@app.route('/view_users')
def view_users():
    # Retrieve a list of users from MongoDB
    user_list = collection.find()
    return render_template('view_users.html', user_list=user_list)

# Route to view a specific user
@app.route('/view_user/<user_id>', methods=['GET'])
def view_user(user_id):
    try:
        user_data = get_user_data_by_id(user_id)  # Replace with your actual function to get user data
        return render_template('view_user.html', user_data=user_data)
    except TemplateNotFound:
        flash('User template not found.', 'error')
        return redirect(url_for('view_users'))  # Redirect to the view_users page or another page

# Route to edit a specific user
@app.route('/edit_user/<user_id>')
def edit_user(user_id):
    # Retrieve the user details from MongoDB based on user_id
    user_data = collection.find_one({'_id': ObjectId(user_id)})
    return render_template('edit_user.html', user_data=user_data)

# Route to delete a specific user
@app.route('/delete_user/<user_id>')
def delete_user(user_id):
    # Delete the user from MongoDB based on user_id
    collection.delete_one({'_id': ObjectId(user_id)})
    return redirect(url_for('view_users'))

# Additional Flask route to handle user updates
@app.route('/update_user/<user_id>', methods=['POST'])
def update_user(user_id):
    # Update the user information in MongoDB based on user_id
    collection.update_one({'_id': ObjectId(user_id)}, {
        '$set': {
            'email': request.form['email'],
            'phone_number': request.form['phone_number'],
        }
    })
    return redirect(url_for('view_user', user_id=user_id))

# Error handling for TemplateNotFound
@app.errorhandler(TemplateNotFound)
def template_not_found_error(e):
    flash('Template not found.', 'error')
    return redirect(url_for('index'))

# General error handling
@app.errorhandler(Exception)
def handle_exception(e):
    flash(f"An error occurred: {str(e)}", 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Exception: {e}")