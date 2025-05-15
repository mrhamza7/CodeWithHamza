from flask import Flask, send_from_directory, render_template, request, url_for
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and vectorizer
try:
    with open('models/spam_model.pkl', 'rb') as f:
        spam_model = pickle.load(f)
    with open('models/count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    spam_model = None
    count_vectorizer = None

# Route to serve the main portfolio page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve images from static/images
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory('static/images', filename)

# Route for Spam Classifier
@app.route('/spam_classifier', methods=['GET', 'POST'])
def spam_classifier():
    prediction = None
    if request.method == 'POST':
        email_text = request.form.get('message')  # Match form field name
        if email_text and spam_model and count_vectorizer:
            try:
                # Transform input text
                text_vector = count_vectorizer.transform([email_text])
                # Predict
                pred = spam_model.predict(text_vector)[0]
                prediction = 'Spam' if pred == 1 else 'Not Spam'
            except Exception as e:
                print(f"Error processing prediction: {e}")
    return render_template('spam_classifier.html', prediction=prediction)

# Route to serve the animals classification page (placeholder)
@app.route('/animals_classification')
def animals_classification():
    return render_template('animals_classification.html')

# Route to serve the bones fracture page (placeholder)
@app.route('/bones_fracture')
def bones_fracture():
    return render_template('bones_fracture.html')

# Route to serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/favicon_io'), 'favicon.ico', mimetype='image/x-icon')

if __name__ == '__main__':
    app.run(debug=True)