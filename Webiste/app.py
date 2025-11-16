import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure Google Gemini API with secure environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = False

if GEMINI_API_KEY:
    try:
        # Test API connectivity with a simple request
        test_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        test_headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        test_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Test"
                        }
                    ]
                }
            ]
        }
        
        # Quick test to verify API key works
        test_response = requests.post(test_url, headers=test_headers, json=test_data, timeout=5)
        if test_response.status_code == 200:
            GEMINI_AVAILABLE = True
            print("✓ Google Gemini API configured successfully")
        else:
            print(f"✗ Gemini API test failed: HTTP {test_response.status_code}")
            
    except Exception as e:
        print(f"✗ Gemini API configuration error: {e}")
        GEMINI_AVAILABLE = False
else:
    print("✗ GEMINI_API_KEY not found in environment variables")
    print("  Please set your API key in the .env file")

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
MODEL_PATH = 'inception_model_T2.keras'  # Update with your actual model path

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Define Royal Enfield bike models - UPDATE WITH YOUR ACTUAL CLASSES
BIKE_MODELS = [
    'Bear 650',
    'Bullet 350',
    'Classic 350',
    'Classic 650',
    'Continental GT 650',
    'Goan Classic 350',
    'Guerilla 450',
    'Himalayan',
    'Hunter 350',
    'Interceptor 650',
    'Meteor 350',
    'Scram 440',
    'Shotgun 650',
    'Super Meteor 650'
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for InceptionV3 model"""
    # InceptionV3 expects 299x299 images
    IMG_SIZE = (299, 299)
    
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    
    # InceptionV3 preprocessing - normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'AI model not loaded. Please check server configuration.'
        }), 500
        
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Preprocess and predict
            processed_img = preprocess_image(filepath)
            
            # Make prediction
            predictions = model.predict(processed_img, verbose=0)
            
            # Get top 5 predictions for better results
            prediction_probs = predictions[0]
            top_indices = np.argsort(prediction_probs)[-5:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(BIKE_MODELS):  # Ensure index is valid
                    confidence = float(prediction_probs[idx])
                    if confidence > 0.005:  # Show predictions above 0.5%
                        results.append({
                            'model': BIKE_MODELS[idx],
                            'confidence': round(confidence * 100, 2),
                            'rank': len(results) + 1
                        })
            
            # Get detailed information for the top prediction using Gemini API
            bike_details = None
            if results:
                top_bike = results[0]['model']
                top_confidence = results[0]['confidence']
                bike_details = get_bike_details_from_gemini(top_bike, top_confidence)
            
            # Clean up - delete uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Enhanced response with metadata and detailed bike information
            response_data = {
                'success': True,
                'predictions': results,
                'top_prediction': results[0] if results else None,
                'bike_details': bike_details,
                'total_predictions': len(results),
                'analysis_metadata': {
                    'model_version': '2.0',
                    'processing_time': 'Real-time',
                    'confidence_threshold': '0.5%'
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Prediction error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'Invalid file type. Please upload a valid image file (JPG, PNG, WEBP, GIF).'
    }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__
    })

def get_bike_details_from_gemini(bike_model, confidence):
    """Get detailed information about a Royal Enfield bike using Gemini API"""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return get_fallback_bike_details(bike_model, confidence)
    
    try:
        prompt = f"""
        Provide detailed information about the Royal Enfield {bike_model} motorcycle in JSON format. Include:
        1. Current price in Indian market (INR)
        2. Engine specifications (displacement, power, torque)
        3. Key features (5-7 bullet points)
        4. Target audience/riding style
        5. Brief description (2-3 sentences)
        6. Fuel efficiency (km/l)
        7. Weight and dimensions
        8. Available colors/variants

        Format the response as valid JSON with these exact keys:
        {{
            "model": "{bike_model}",
            "price_inr": "₹X.XX Lakh",
            "price_range": "₹X.XX - ₹X.XX Lakh",
            "engine": {{
                "displacement": "XXXcc",
                "power": "XX bhp",
                "torque": "XX Nm"
            }},
            "features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
            "target_audience": "Description of ideal rider",
            "description": "Brief compelling description",
            "fuel_efficiency": "XX km/l",
            "weight": "XXX kg",
            "dimensions": "Length x Width x Height",
            "colors": ["Color1", "Color2", "Color3"]
        }}

        Provide accurate, current information for 2024-2025 models.
        """
        
        # Use Gemini REST API directly
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract text from response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                content = response_data['candidates'][0]['content']['parts'][0]['text']
                
                # Clean up the response if it has markdown formatting
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                
                bike_info = json.loads(content)
                bike_info['confidence'] = confidence
                bike_info['source'] = 'Gemini AI'
                
                return bike_info
            else:
                print("No candidates in Gemini response")
                return get_fallback_bike_details(bike_model, confidence)
        else:
            print(f"Gemini API request failed: HTTP {response.status_code}")
            return get_fallback_bike_details(bike_model, confidence)
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_fallback_bike_details(bike_model, confidence)

def get_fallback_bike_details(bike_model, confidence):
    """Fallback bike information when Gemini API is not available"""
    # Predefined bike information as fallback
    bike_database = {
        'Classic 350': {
            "model": "Classic 350",
            "price_inr": "₹1.93 Lakh",
            "price_range": "₹1.93 - ₹2.18 Lakh",
            "engine": {
                "displacement": "349cc",
                "power": "20.2 bhp",
                "torque": "27 Nm"
            },
            "features": [
                "Classic retro styling",
                "Long seat for comfortable touring",
                "Chrome exhaust with authentic thump",
                "Analog instrument cluster",
                "Electric start with kick start backup"
            ],
            "target_audience": "Riders who love classic styling and relaxed cruising",
            "description": "The iconic Royal Enfield Classic 350 combines timeless design with modern reliability, perfect for both city rides and weekend adventures.",
            "fuel_efficiency": "35-40 km/l",
            "weight": "195 kg",
            "dimensions": "2090mm x 800mm x 1075mm",
            "colors": ["Chrome Black", "Chrome Bronze", "Chrome Blue", "Halcyon Green"]
        },
        'Himalayan': {
            "model": "Himalayan",
            "price_inr": "₹2.16 Lakh",
            "price_range": "₹2.16 - ₹2.30 Lakh",
            "engine": {
                "displacement": "411cc",
                "power": "24.3 bhp",
                "torque": "32 Nm"
            },
            "features": [
                "Adventure touring capability",
                "High ground clearance (220mm)",
                "Long travel suspension",
                "Purpose-built luggage options",
                "Compass in instrument cluster"
            ],
            "target_audience": "Adventure enthusiasts and touring riders",
            "description": "Built for the mountains and beyond, the Himalayan is Royal Enfield's adventure motorcycle designed to go anywhere.",
            "fuel_efficiency": "30-35 km/l",
            "weight": "199 kg",
            "dimensions": "2190mm x 840mm x 1360mm",
            "colors": ["Granite Black", "Mirage Silver", "Pine Green"]
        },
        'Interceptor 650': {
            "model": "Interceptor 650",
            "price_inr": "₹2.85 Lakh",
            "price_range": "₹2.85 - ₹3.03 Lakh",
            "engine": {
                "displacement": "648cc",
                "power": "47 bhp",
                "torque": "52 Nm"
            },
            "features": [
                "Parallel twin 650cc engine",
                "Classic roadster styling",
                "Dual channel ABS",
                "Twin exhaust pipes",
                "Comfortable upright riding position"
            ],
            "target_audience": "Riders seeking modern performance with classic appeal",
            "description": "The Interceptor 650 brings back the golden era of motorcycling with its parallel twin engine and timeless design.",
            "fuel_efficiency": "25-30 km/l",
            "weight": "202 kg",
            "dimensions": "2122mm x 789mm x 1080mm",
            "colors": ["Chrome Red", "Mark Three Black", "Silver Spectre", "Baker Express"]
        }
    }
    
    # Get bike info from database or create generic info
    bike_info = bike_database.get(bike_model, {
        "model": bike_model,
        "price_inr": "₹2.00 - ₹3.50 Lakh",
        "price_range": "Price varies by variant",
        "engine": {
            "displacement": "Variable",
            "power": "Variable",
            "torque": "Variable"
        },
        "features": [
            "Royal Enfield heritage design",
            "Reliable performance",
            "Classic styling",
            "Modern features",
            "Comfortable riding experience"
        ],
        "target_audience": "Royal Enfield enthusiasts",
        "description": f"The {bike_model} embodies Royal Enfield's commitment to craftsmanship and riding pleasure.",
        "fuel_efficiency": "25-35 km/l",
        "weight": "180-220 kg",
        "dimensions": "Standard motorcycle dimensions",
        "colors": ["Multiple color options available"]
    })
    
    bike_info['confidence'] = confidence
    bike_info['source'] = 'Database'
    
    return bike_info

if __name__ == '__main__':
    print("="*60)
    print("Royal Enfield Model Identifier")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Model loaded: {'Yes' if model else 'No'}")
    
    if model:
        print(f"Number of classes: {len(BIKE_MODELS)}")
    
    print("="*60)
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)