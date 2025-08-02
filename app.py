import os
import logging
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_from_directory, Blueprint
import google.generativeai as genai
import json
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('employee_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration class"""
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    DATA_FILE = os.getenv('DATA_FILE', 'employees.json')

@dataclass
class Employee:
    """Employee data model"""
    name: str
    email: str
    department: str
    job_title: str
    joining_date: str
    employee_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

class EmployeeValidator:
    """Validator class for employee data"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if email == 'unknown':
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate employee name"""
        return name != 'unknown' and len(name.strip()) >= 2
    
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

class DataManager:
    """Handles data persistence and retrieval"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.employees: List[Employee] = []
        self.load_data()
    
    def load_data(self):
        """Load employee data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.employees = [Employee(**emp) for emp in data]
                logger.info(f"Loaded {len(self.employees)} employees from {self.data_file}")
            else:
                logger.info("No existing data file found. Starting with empty employee list.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.employees = []
    
    def save_data(self):
        """Save employee data to file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(emp) for emp in self.employees], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.employees)} employees to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def add_employee(self, employee: Employee) -> int:
        """Add new employee and return assigned ID"""
        employee.employee_id = len(self.employees) + 1
        self.employees.append(employee)
        self.save_data()
        return employee.employee_id
    
    def get_all_employees(self) -> List[Dict]:
        """Get all employees as dictionaries"""
        return [asdict(emp) for emp in self.employees]
    
    def get_employee_by_id(self, employee_id: int) -> Optional[Employee]:
        """Get employee by ID"""
        return next((emp for emp in self.employees if emp.employee_id == employee_id), None)

class AIEmployeeExtractor:
    """AI-powered employee data extraction using Google Gemini"""
    
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("Google API key is required")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized AI extractor with model: {model_name}")
    
    def extract_employee_data(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Extract structured employee data from unstructured input"""
        try:
            instruction = self._build_extraction_prompt(prompt)
            response = self.model.generate_content(instruction)
            extracted_text = response.text.strip()
            
            logger.info(f"AI Response: {extracted_text}")
            
            return self._parse_ai_response(extracted_text)
            
        except Exception as e:
            logger.error(f"Error during AI processing: {e}")
            return None
    
    def _build_extraction_prompt(self, prompt: str) -> str:
        """Build the AI extraction prompt"""
        today = date.today().strftime('%Y-%m-%d')
        return f"""
        You are an expert at extracting employee information from text. 
        
        Given this employee information: '{prompt}'
        
        Extract the following details:
        - name: Full name of the employee
        - email: Email address
        - department: Department/Division
        - job_title: Job title/position
        - joining_date: Date of joining (YYYY-MM-DD format)
        
        Rules:
        1. If joining date is not provided, use today's date: {today}
        2. If any information is missing or unclear, use 'unknown'
        3. Format the response exactly as: name: <value>, email: <value>, department: <value>, job_title: <value>, joining_date: <value>
        4. Ensure email addresses are valid format
        5. Use proper capitalization for names and titles
        
        Extract only factual information from the text.
        """
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, str]:
        """Parse AI response into structured data"""
        data = {}
        
        # Split by comma and parse key-value pairs
        for item in response_text.split(','):
            item = item.strip()
            if ':' in item:
                key, value = item.split(':', 1)
                data[key.strip()] = value.strip()
            else:
                logger.warning(f"Could not parse item: {item}")
        
        # Ensure all required fields are present
        required_fields = ['name', 'email', 'department', 'job_title', 'joining_date']
        for field in required_fields:
            if field not in data:
                data[field] = 'unknown'
        
        # Set default joining date if unknown
        if data.get('joining_date') == 'unknown':
            data['joining_date'] = date.today().strftime('%Y-%m-%d')
        
        return data

app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for all routes
CORS(app)

# Create API v1 blueprint
api_v1 = Blueprint("api_v1", __name__)

# Swagger UI Configuration
SWAGGER_URL = '/docs'  # URL for exposing Swagger UI
API_URL = '/swagger.yaml'  # Our API spec URL

# Create blueprint for swagger UI
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Employee Management System API",
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete'],
        'tryItOutEnabled': True,
        'displayRequestDuration': True,
        'docExpansion': 'list',
        'defaultModelsExpandDepth': 2,
        'defaultModelExpandDepth': 2,
        'deepLinking': True
    }
)

# Register swagger UI blueprint
app.register_blueprint(swaggerui_blueprint)

# Initialize components
try:
    data_manager = DataManager(Config.DATA_FILE)
    ai_extractor = AIEmployeeExtractor(Config.GOOGLE_API_KEY, Config.MODEL_NAME)
    validator = EmployeeValidator()
except Exception as e:
    logger.error(f"Failed to initialize application components: {e}")
    raise

# Serve swagger.yaml file
@app.route('/swagger.yaml')
def swagger_spec():
    """Serve the swagger specification file"""
    try:
        return send_from_directory('.', 'swagger.yaml', mimetype='text/yaml')
    except FileNotFoundError:
        logger.error("swagger.yaml file not found")
        return jsonify({'error': 'Swagger specification file not found'}), 404

# Root route - redirect to docs
@api_v1.route('/')
def root():
    """Root endpoint - provides API information and links"""
    return jsonify({
        'message': 'Welcome to Employee Management System API',
        'version': '1.0.0',
        'documentation': request.host_url.rstrip('/') + '/docs',
        'health_check': request.host_url.rstrip('/') + '/api/v1/health',
        'endpoints': {
            'employees': request.host_url.rstrip('/') + '/api/v1/employees',
            'statistics': request.host_url.rstrip('/') + '/api/v1/stats'
        }
    }), 200

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'The requested resource was not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@api_v1.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'total_employees': len(data_manager.employees),
        'api_version': '1.0.0',
        'documentation': request.host_url.rstrip('/') + '/docs'
    }), 200

@api_v1.route('/employees', methods=['POST'])
def create_employee():
    """Create a new employee from unstructured text input"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        prompt = request.json.get('prompt')
        if not prompt or not prompt.strip():
            return jsonify({'error': 'Missing or empty prompt field'}), 400
        
        logger.info(f"Processing employee creation request with prompt: {prompt[:100]}...")
        
        # Extract data using AI
        extracted_data = ai_extractor.extract_employee_data(prompt)
        if not extracted_data:
            return jsonify({'error': 'Failed to extract employee data from prompt'}), 500
        
        # Validate extracted data
        validation_errors = []
        
        if not validator.validate_name(extracted_data.get('name', '')):
            validation_errors.append("Invalid or missing employee name")
        
        if not validator.validate_email(extracted_data.get('email', '')):
            validation_errors.append("Invalid or missing email address")
        
        if not validator.validate_date(extracted_data.get('joining_date', '')):
            validation_errors.append("Invalid joining date format")
        
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors,
                'extracted_data': extracted_data
            }), 400
        
        # Create employee object
        employee = Employee(**extracted_data)
        
        # Save to database
        employee_id = data_manager.add_employee(employee)
        
        logger.info(f"Successfully created employee with ID: {employee_id}")
        
        return jsonify({
            'message': 'Employee created successfully',
            'employee_id': employee_id,
            'employee_data': asdict(employee)
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@api_v1.route('/employees', methods=['GET'])
def list_employees():
    """List all employees with optional filtering"""
    try:
        # Get query parameters for filtering
        department = request.args.get('department')
        limit = request.args.get('limit', type=int)
        
        employees = data_manager.get_all_employees()
        
        # Apply filters
        if department:
            employees = [emp for emp in employees if emp.get('department', '').lower() == department.lower()]
        
        if limit and limit > 0:
            employees = employees[:limit]
        
        return jsonify({
            'employees': employees,
            'total_count': len(employees),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing employees: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@api_v1.route('/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    """Get a specific employee by ID"""
    try:
        employee = data_manager.get_employee_by_id(employee_id)
        
        if not employee:
            return jsonify({'error': 'Employee not found'}), 404
        
        return jsonify({
            'employee': asdict(employee),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving employee {employee_id}: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@api_v1.route('/stats', methods=['GET'])
def get_statistics():
    """Get employee statistics"""
    try:
        employees = data_manager.get_all_employees()
        
        # Calculate statistics
        stats = {
            'total_employees': len(employees),
            'departments': {},
            'recent_hires': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count by department
        for emp in employees:
            dept = emp.get('department', 'Unknown')
            stats['departments'][dept] = stats['departments'].get(dept, 0) + 1
        
        # Count recent hires (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
        for emp in employees:
            try:
                joining_date = datetime.strptime(emp.get('joining_date', ''), '%Y-%m-%d').date()
                if joining_date >= thirty_days_ago:
                    stats['recent_hires'] += 1
            except ValueError:
                continue
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

app.register_blueprint(api_v1, url_prefix="/api/v1")

if __name__ == '__main__':
    # Validate required environment variables
    if not Config.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable is required")
        exit(1)
    
    # Check if swagger.yaml exists
    if not os.path.exists('swagger.yaml'):
        logger.warning("swagger.yaml file not found. Swagger UI will not work properly.")
    
    logger.info("Starting Employee Management System...")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Data file: {Config.DATA_FILE}")
    logger.info(f"Swagger UI available at: http://localhost:5000/docs")
    
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)