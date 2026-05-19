from flask import Flask, request, jsonify
from flask_cors import CORS
from models.health_risk_predictor import HealthRiskPredictor
from models.audio.audio_processor import AudioProcessor
import os
import werkzeug
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Configure standard logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize models
predictor = HealthRiskPredictor()
audio_processor = AudioProcessor()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"Received form data: {request.form}")
        # 1. Get numeric markers
        age = float(request.form.get('age'))
        systolic_bp = float(request.form.get('systolic_bp'))
        diastolic_bp = float(request.form.get('diastolic_bp'))
        bs = float(request.form.get('bs'))
        heart_rate = float(request.form.get('heart_rate'))
        body_temp_c = float(request.form.get('body_temp'))
        
        # Convert Celsius to Fahrenheit for the model
        body_temp_f = (body_temp_c * 9/5) + 32
        
        markers_data = {
            'Age': age,
            'SystolicBP': systolic_bp,
            'DiastolicBP': diastolic_bp,
            'BS': bs,
            'HeartRate': heart_rate,
            'BodyTemp': body_temp_f
        }
        
        # Predict health risk
        risk_level = predictor.predict(markers_data)
        
        # 2. Process audio
        audio_file = request.files.get('audio')
        bypass_transcription = request.form.get('bypass_transcription') == 'true'
        transcription_method = request.form.get('transcription_method', 'whisper')
        audio_results = None
        
        if audio_file:
            filename = werkzeug.utils.secure_filename(audio_file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                audio_file.save(filepath)
                audio_results = audio_processor.process_audio(
                    filepath, 
                    bypass_transcription=bypass_transcription,
                    transcription_method=transcription_method
                )
            finally:
                # Clean up upload even if processing fails
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify({
            "status": "success",
            "health_risk_level": risk_level,
            "audio_analysis": audio_results
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in /predict: {error_details}")
        return jsonify({"status": "error", "message": str(e), "details": error_details}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
