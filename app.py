from flask import Flask, request, jsonify, send_from_directory, current_app, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import uuid
import json
from datetime import datetime
import logging
import time
from model_pipeline import ObjectCounter
from monitoring import metrics_collector
from few_shot_learning import few_shot_learner
from image_generator import ImageGenerator
from safety_module import safety_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///object_counting.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
CORS(app)
db = SQLAlchemy(app)

# Initialize AI model pipeline
object_counter = ObjectCounter()

# Initialize image generator
image_generator = ImageGenerator()

# Database Models
class CountingResult(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image_path = db.Column(db.String(255), nullable=False)
    item_type = db.Column(db.String(100), nullable=False)
    predicted_count = db.Column(db.Integer, nullable=False)
    corrected_count = db.Column(db.Integer, nullable=True)
    confidence_score = db.Column(db.Float, nullable=False, default=0.0)
    processing_time = db.Column(db.Float, nullable=False, default=0.0)
    user_feedback = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'image_path': self.image_path,
            'item_type': self.item_type,
            'predicted_count': self.predicted_count,
            'corrected_count': self.corrected_count,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'user_feedback': self.user_feedback
        }

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predefined object types (as specified in requirements)
from config import OBJECT_TYPES, normalize_item_type

@app.route('/api/count', methods=['POST'])
def count_objects():
    """
    API endpoint to upload an image and count objects of a specific type.
    
    Expected input:
    - image: image file (multipart/form-data)
    - item_type: string from predefined list
    
    Returns:
    - JSON response with count results
    """
    start = time.time()
    try:
        raw_item_type = request.form.get('item_type', None)
        fileobj = request.files.get('image')

        # Run safety pre-check immediately before any validation (TEXT ONLY)
        text_to_check = f"{raw_item_type} {request.form.get('description', '')}"
        safety_violations = safety_module.check_text_safety(text_to_check)
        if safety_violations:
            for violation in safety_violations:
                safety_module.log_violation(violation, "uploaded_image")
            reasons = [v.violation_type for v in safety_violations]
            evidence = {
                "violations": [
                    {"reason": v.violation_type, "details": v.evidence}
                    for v in safety_violations
                ]
            }
            violations = [
                {"type": v.violation_type, "reason": v.violation_type, "details": v.evidence}
                for v in safety_violations
            ]
            return jsonify({"status": "blocked", "reasons": reasons, "evidence": evidence, "violations": violations}), 403

        # Validate item_type
        if not raw_item_type:
            return jsonify({"error": "No item type specified"}), 400
        item_type = normalize_item_type(raw_item_type)
        if item_type is None:
            return jsonify({"error": "Invalid or missing item type"}), 400

        # Validate file presence and extension
        if fileobj is None or fileobj.filename == '':
            return jsonify({"error": "No image file provided"}), 400

        if not allowed_file(fileobj.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {list(ALLOWED_EXTENSIONS)}'
            }), 400

        # IMPORTANT: Do NOT read() the stream before saving (avoids empty file issue)
        filename = secure_filename(fileobj.filename) or f"upload_{int(time.time())}.jpg"
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fileobj.save(temp_file_path)

        # Quick sanity check that it's a real image
        try:
            from PIL import Image as _PIL_Image
            with _PIL_Image.open(temp_file_path) as _im:
                _im.verify()
        except Exception:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({'error': 'Invalid image file format'}), 400

        # Secondary text safety (kept for backward compatibility)
        text_violations = safety_module.check_text_safety(f"{item_type} {request.form.get('description', '')}")
        if text_violations:
            for violation in text_violations:
                safety_module.log_violation(violation, "uploaded_image")
                metrics_collector.record_blocked_request(
                    reason=violation.violation_type,
                    pipeline_version="1.0.0"
                )
            evidence = {
                "violations": [
                    {
                        "type": v.violation_type,
                        "reason": v.reason,
                        "confidence": v.confidence,
                        "evidence": v.evidence
                    } for v in text_violations
                ],
                "timestamp": time.time(),
                "image_path": "uploaded_image"
            }
            evidence_file = os.path.join(
                safety_module.evidence_dir,
                f"blocked_request_{int(time.time())}.json"
            )
            try:
                with open(evidence_file, 'w') as f:
                    json.dump(evidence, f, indent=2)
            except Exception as _e:
                logger.error(f"Failed to save evidence file: {_e}")

            return jsonify({
                'error': 'Request blocked due to safety policy violation',
                'reason': 'Military vehicle counting detected',
                'evidence_file': evidence_file,
                'violations': [
                    {
                        'type': v.violation_type,
                        'reason': v.reason,
                        'confidence': v.confidence
                    } for v in text_violations
                ]
            }), 403

        # Process image with AI pipeline using the saved file path
        try:
            result = object_counter.count_objects(temp_file_path, target_item_type=item_type)
        except Exception as e:
            if "UnidentifiedImageError" in str(e) or "cannot identify image file" in str(e):
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return jsonify({'error': 'Invalid image file format'}), 400
            else:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                raise

        # Clean up the temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

        # Response shaping helper
        def make_count_response(result_dict, db_record=None, processing_time=None, item_type=None):
            count = int(result_dict.get("count", 0))
            response = {
                "id": None,
                "total": count,
                "count": count,
                "item_type": item_type,
                "confidence": result_dict.get("confidence", 0.0),
                "confidence_score": result_dict.get("confidence", 0.0),  # keep both fields
                "processing_time": processing_time,
                "details": result_dict.get("details", {}),
                "meta": result_dict.get("meta", {})
            }
            if db_record is not None and getattr(db_record, "id", None) is not None:
                response["id"] = db_record.id
            return response

        processing_time = time.time() - start

        # Persist to DB
        try:
            db_res = CountingResult(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                image_path="uploaded_images/unknown.jpg",
                item_type=item_type,
                predicted_count=int(result.get("count", 0)),
                corrected_count=None,
                confidence_score=float(result.get("confidence", 0.0)),
                processing_time=float(processing_time)
            )
            db.session.add(db_res)
            db.session.commit()
            db.session.refresh(db_res)
        except Exception:
            current_app.logger.exception("DB write failed")
            db_res = None

        # Metrics
        response_time = time.time() - start
        try:
            metrics_collector.record_request('/api/count', 'POST', 200, response_time, pipeline_version="1.0.0")
            metrics_collector.record_counting_result(result, pipeline_version="1.0.0")
        except Exception:
            current_app.logger.exception("metrics.record_request failed")

        # Build response
        response_data = make_count_response(result, db_res, response_time, item_type)
        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.exception("Unhandled /api/count exception")
        response_time = time.time() - start
        try:
            metrics_collector.record_request('/api/count', 'POST', 500, response_time, pipeline_version="1.0.0")
        except Exception:
            current_app.logger.exception("metrics.record_request failed in exception path")
        return jsonify({"error": "internal_server_error", "details": str(e)}), 500

@app.route('/api/correct', methods=['GET', 'POST'])
def correct_count():
    """
    API endpoint to submit corrections for count results.
    
    Expected input:
    - result_id: string (UUID of the result to correct)
    - corrected_count: integer (the correct count)
    - user_feedback: string (optional feedback)
    
    Returns:
    - JSON response with confirmation
    """
    current_app.logger.debug("correct_count invoked; method=%s", request.method)
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        result_id = data.get('result_id')
        corrected_count = data.get('corrected_count')
        user_feedback = data.get('user_feedback', '')
        
        # Validate required fields
        if not result_id:
            return jsonify({'error': 'result_id is required'}), 400
        
        if corrected_count is None or not isinstance(corrected_count, int):
            return jsonify({'error': 'corrected_count must be an integer'}), 400
        
        # Find the result in database
        db_result = CountingResult.query.get(result_id)
        if not db_result:
            return jsonify({'error': 'Result not found'}), 404
        
        # Update the result
        db_result.corrected_count = corrected_count
        db_result.user_feedback = user_feedback
        db.session.commit()
        
        # Update metrics with corrected values
        try:
            result_dict = {
                'item_type': db_result.item_type,
                'count': db_result.predicted_count,
                'confidence': db_result.confidence_score,
                'processing_time': db_result.processing_time,
                'corrected_count': corrected_count
            }
            metrics_collector.record_counting_result(result_dict, pipeline_version="1.0.0")
        except Exception as e:
            logger.error(f"Error updating metrics after correction: {str(e)}")
        
        logger.info(f"Count corrected: {result_id} -> {corrected_count}")
        
        return jsonify({
            'message': 'Count corrected successfully',
            'result_id': result_id,
            'corrected_count': corrected_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error correcting count: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """
    API endpoint to retrieve previous counting results.
    
    Query parameters:
    - item_type: filter by object type (optional)
    - limit: number of results to return (optional, default 50)
    - offset: number of results to skip (optional, default 0)
    
    Returns:
    - JSON response with list of results
    """
    try:
        # Get query parameters
        item_type = request.args.get('item_type')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Build query
        query = CountingResult.query
        
        if item_type:
            if item_type not in OBJECT_TYPES:
                return jsonify({'error': f'Invalid item type: {item_type}'}), 400
            query = query.filter(CountingResult.item_type == item_type)
        
        # Order by timestamp (newest first) and apply pagination
        results = query.order_by(CountingResult.timestamp.desc()).offset(offset).limit(limit).all()
        
        # Convert to list of dictionaries
        results_list = [result.to_dict() for result in results]
        
        # Get total count for pagination info
        total_count = query.count()
        
        response = {
            'total': total_count,
            'page': (offset // limit) + 1,  # Calculate page number
            'per_page': limit,
            'results': results_list,
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'AI Object Counting API'
    }), 200

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """
    API documentation endpoint.
    
    Returns:
    - JSON response with API information
    """
    return jsonify({
        "name": "AI Object Counting API",
        "version": "1.0.0",
        "endpoints": ["/api/count", "/api/results", "/api/health", "/metrics"]
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    OpenMetrics endpoint for Prometheus scraping.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        output = generate_latest(metrics_collector.registry)
        return Response(output, content_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        return jsonify({'error': 'Failed to generate metrics'}), 500

@app.route('/api/status', methods=['GET'])
def status_check():
    """
    Status check endpoint for Flutter frontend compatibility.
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'AI Object Counting API (Real AI Models)'
    }), 200

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get paginated history of counting results for Flutter frontend.
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Query database with pagination
        results = CountingResult.query.order_by(CountingResult.timestamp.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return jsonify({
            'results': [result.to_dict() for result in results.items],
            'page': page,
            'per_page': per_page,
            'total': results.total,
            'has_more': results.has_next
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files (for development purposes).
    In production, use a proper file server or CDN.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Few-shot learning endpoints
@app.route('/api/learn', methods=['POST'])
def learn_new_object():
    """
    Learn a new object type from provided images.
    
    Expected input:
    - object_name: string (name of the new object type)
    - images: list of image files (multipart/form-data)
    
    Returns:
    - JSON response with learning results
    """
    try:
        # Get object name
        object_name = request.form.get('object_name')
        if not object_name:
            return jsonify({'error': 'Object name is required'}), 400
        
        # Get uploaded images
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if len(files) < 2:
            return jsonify({'error': 'At least 2 images are required for learning'}), 400
        
        # Save uploaded images
        image_paths = []
        for i, file in enumerate(files):
            if file and file.filename:
                filename = secure_filename(f"{object_name}_{i}_{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_paths.append(file_path)
        
        # Learn the new object
        learning_result = few_shot_learner.learn_new_object(object_name, image_paths)
        
        if learning_result['learning_successful']:
            return jsonify(learning_result), 200
        else:
            return jsonify(learning_result), 400
            
    except Exception as e:
        logger.error(f"Error learning new object: {str(e)}")
        return jsonify({'error': f'Error learning new object: {str(e)}'}), 500

@app.route('/api/learned-objects', methods=['GET'])
def list_learned_objects():
    """
    List all learned object types.
    
    Returns:
    - JSON response with list of learned objects
    """
    try:
        objects = few_shot_learner.list_learned_objects()
        return jsonify({
            'learned_objects': objects,
            'count': len(objects)
        }), 200
    except Exception as e:
        logger.error(f"Error listing learned objects: {str(e)}")
        return jsonify({'error': f'Error listing learned objects: {str(e)}'}), 500

@app.route('/api/count-learned', methods=['POST'])
def count_learned_objects():
    """
    Count instances of a learned object type in an image.
    
    Expected input:
    - image: image file (multipart/form-data)
    - object_name: string (name of the learned object type)
    
    Returns:
    - JSON response with counting results
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get object name
        object_name = request.form.get('object_name')
        if not object_name:
            return jsonify({'error': 'Object name is required'}), 400
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Count learned objects
        counting_result = few_shot_learner.count_learned_objects(file_path, object_name)
        
        # Add metadata
        counting_result['image_path'] = file_path
        counting_result['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(counting_result), 200
        
    except Exception as e:
        logger.error(f"Error counting learned objects: {str(e)}")
        return jsonify({'error': f'Error counting learned objects: {str(e)}'}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_objects():
    """
    Recognize learned objects in an image.
    
    Expected input:
    - image: image file (multipart/form-data)
    - threshold: float (optional, similarity threshold)
    
    Returns:
    - JSON response with recognition results
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get threshold
        threshold = float(request.form.get('threshold', 0.5))
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Recognize objects
        recognition_result = few_shot_learner.recognize_object(file_path, threshold)
        
        # Add metadata
        recognition_result['image_path'] = file_path
        recognition_result['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(recognition_result), 200
        
    except Exception as e:
        logger.error(f"Error recognizing objects: {str(e)}")
        return jsonify({'error': f'Error recognizing objects: {str(e)}'}), 500

@app.route('/api/delete-learned-object', methods=['DELETE'])
def delete_learned_object():
    """
    Delete a learned object type.
    
    Expected input:
    - object_name: string (name of the object to delete)
    
    Returns:
    - JSON response with deletion results
    """
    try:
        data = request.get_json()
        object_name = data.get('object_name')
        
        if not object_name:
            return jsonify({'error': 'Object name is required'}), 400
        
        success = few_shot_learner.delete_object(object_name)
        
        if success:
            return jsonify({
                'message': f'Object "{object_name}" deleted successfully',
                'success': True
            }), 200
        else:
            return jsonify({
                'error': f'Object "{object_name}" not found',
                'success': False
            }), 404
            
    except Exception as e:
        logger.error(f"Error deleting learned object: {str(e)}")
        return jsonify({'error': f'Error deleting learned object: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# Image Generation Endpoints
@app.route('/api/generate-image', methods=['POST'])
def generate_single_image():
    """Generate a single synthetic test image"""
    try:
        data = request.get_json()
        
        # Extract parameters
        object_type = data.get('object_type', 'car')
        count = data.get('count', 3)
        width = int(data.get('size', '512x512').split('x')[0])
        height = int(data.get('size', '512x512').split('x')[1])
        background_type = data.get('background', 'white')
        clarity_level = data.get('clarity', 0.8)
        noise_level = data.get('noise', 10)
        rotation_angle = data.get('rotation', 0)
        
        # Generate image
        image, metadata = image_generator.generate_synthetic_image(
            object_type=object_type,
            count=count,
            width=width,
            height=height,
            background_type=background_type,
            clarity_level=clarity_level,
            noise_level=noise_level,
            rotation_range=(rotation_angle, rotation_angle)
        )
        
        # Save image
        image_id = str(uuid.uuid4())
        image_filename = f"generated_{image_id}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)
        
        # Test the generated image with AI
        start_time = time.time()
        counting_result = object_counter.count_objects(image_path, object_type)
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics_collector.record_request('/api/generate-image', 'POST', 200, processing_time, object_type, pipeline_version="1.0.0")
        
        return jsonify({
            'success': True,
            'image_id': image_id,
            'image_path': image_path,
            'generation_metadata': metadata,
            'ai_test_result': {
                'predicted_count': counting_result.get('count', 0),
                'confidence': counting_result.get('confidence', 0.0),
                'processing_time': processing_time,
                'true_count': count,
                'accuracy': 1.0 if counting_result.get('count', 0) == count else 0.0
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-batch-test', methods=['POST'])
def run_batch_test():
    """Run batch testing with multiple generated images"""
    try:
        data = request.get_json()
        
        # Extract parameters
        num_tests = data.get('num_tests', 10)
        object_type = data.get('object_type', 'car')
        width = int(data.get('size', '512x512').split('x')[0])
        height = int(data.get('size', '512x512').split('x')[1])
        background_type = data.get('background', 'white')
        clarity_level = data.get('clarity', 0.8)
        noise_level = data.get('noise', 10)
        rotation_angle = data.get('rotation', 0)
        
        test_results = []
        successful_tests = 0
        total_processing_time = 0
        
        for i in range(num_tests):
            try:
                # Generate random count for each test
                count = (i % 5) + 1  # 1-5 objects
                
                # Generate image
                image, metadata = image_generator.generate_synthetic_image(
                    object_type=object_type,
                    count=count,
                    width=width,
                    height=height,
                    background_type=background_type,
                    clarity_level=clarity_level,
                    noise_level=noise_level,
                    rotation_range=(rotation_angle, rotation_angle)
                )
                
                # Save image
                image_id = str(uuid.uuid4())
                image_filename = f"batch_test_{image_id}.png"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image.save(image_path)
                
                # Test with AI
                start_time = time.time()
                counting_result = object_counter.count_objects(image_path, object_type)
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Calculate accuracy
                predicted_count = counting_result.get('count', 0)
                accuracy = 1.0 if predicted_count == count else 0.0
                if accuracy == 1.0:
                    successful_tests += 1
                
                # Record metrics
                metrics_collector.record_request('/api/run-batch-test', 'POST', 200, processing_time, object_type, pipeline_version="1.0.0")
                
                test_result = {
                    'test_id': i + 1,
                    'object_type': object_type,
                    'true_count': count,
                    'predicted_count': predicted_count,
                    'accuracy': accuracy,
                    'confidence': counting_result.get('confidence', 0.0),
                    'response_time': processing_time,
                    'image_size': f"{width}x{height}",
                    'image_id': image_id
                }
                
                test_results.append(test_result)
                
                # Clean up image file
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
            except Exception as e:
                logger.error(f"Error in batch test {i+1}: {str(e)}")
                test_result = {
                    'test_id': i + 1,
                    'object_type': object_type,
                    'true_count': count,
                    'predicted_count': 0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'response_time': 0.0,
                    'image_size': f"{width}x{height}",
                    'error': str(e)
                }
                test_results.append(test_result)
        
        # Calculate summary statistics
        accuracy_rate = (successful_tests / num_tests) * 100 if num_tests > 0 else 0.0
        avg_response_time = total_processing_time / num_tests if num_tests > 0 else 0.0
        
        return jsonify({
            'success': True,
            'summary': {
                'total_tests': num_tests,
                'successful_tests': successful_tests,
                'accuracy_rate': accuracy_rate,
                'avg_response_time': avg_response_time
            },
            'test_results': test_results
        })
        
    except Exception as e:
        logger.error(f"Error running batch test: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generated-image/<image_id>')
def get_generated_image(image_id):
    """Serve generated image files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], f"generated_{image_id}.png")
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/safety/stats', methods=['GET'])
def get_safety_stats():
    """Get safety violation statistics"""
    try:
        stats = safety_module.get_violation_stats()
        return jsonify({
            'success': True,
            'safety_stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting safety stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/safety/evidence/<filename>')
def get_safety_evidence(filename):
    """Serve safety evidence files"""
    try:
        return send_from_directory(safety_module.evidence_dir, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Evidence file not found'}), 404

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
    
    # Run the application
    port = int(os.environ.get('API_PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)