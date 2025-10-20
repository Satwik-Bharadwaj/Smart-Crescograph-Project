# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
# Core Python libraries for file handling, data processing, and web framework
import os  # File system operations
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import plotly.graph_objs as go  # Interactive plotting
import plotly.utils  # Plotly utilities for JSON conversion
import json  # JSON data handling
from datetime import datetime, timedelta  # Date and time operations

# Machine Learning libraries
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.ensemble import RandomForestRegressor  # Random Forest ensemble model
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics

# Flask web framework and extensions
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy  # Database ORM
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user  # User authentication
from flask_wtf import FlaskForm  # Form handling
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField  # Form fields
from wtforms.validators import DataRequired, Length, EqualTo, Email  # Form validation
from werkzeug.utils import secure_filename  # Secure file uploads
from werkzeug.security import generate_password_hash, check_password_hash  # Password hashing
import io  # In-memory file operations
import uuid  # Unique identifier generation

# =============================================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================================
# Initialize the Flask web application
app = Flask(__name__)

# Configure Flask application settings
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'  # Secret key for session security (change in production)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory where uploaded CSV files are stored
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximum file upload size (16MB)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_growth.db'  # SQLite database file location
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable SQLAlchemy event tracking for performance

# =============================================================================
# EXTENSION INITIALIZATION
# =============================================================================
# Initialize database ORM (Object-Relational Mapping)
db = SQLAlchemy(app)

# Initialize user authentication system
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to login page if user not authenticated
login_manager.login_message = 'Please log in to access this page.'  # Message shown when login required

# Create uploads directory if it doesn't exist
# This ensures the directory exists before any file uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================
# Define allowed file extensions for security
ALLOWED_EXTENSIONS = {'csv'}  # Only CSV files are allowed for upload

# =============================================================================
# DATABASE MODELS (SQLAlchemy ORM)
# =============================================================================
# These classes define the database schema and relationships

class User(UserMixin, db.Model):
    """
    User model for authentication and user management.
    Inherits from UserMixin to work with Flask-Login.
    """
    __tablename__ = 'user'  # Database table name
    
    # Primary key - unique identifier for each user
    id = db.Column(db.Integer, primary_key=True)
    
    # User credentials and profile information
    username = db.Column(db.String(80), unique=True, nullable=False)  # Unique username
    email = db.Column(db.String(120), unique=True, nullable=False)  # Unique email address
    password_hash = db.Column(db.String(120), nullable=False)  # Hashed password for security
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Account creation timestamp
    
    # Relationship to user's uploads (one-to-many)
    uploads = db.relationship('Upload', backref='user', lazy=True)

class Upload(db.Model):
    """
    Upload model for storing CSV file upload information and metadata.
    Each upload belongs to a user and contains processed plant growth data.
    """
    __tablename__ = 'upload'  # Database table name
    
    # Primary key - unique identifier for each upload
    id = db.Column(db.Integer, primary_key=True)
    
    # File information
    filename = db.Column(db.String(120), nullable=False)  # Server-side filename (with user ID prefix)
    original_filename = db.Column(db.String(120), nullable=False)  # Original filename from user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to User
    
    # Upload metadata
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)  # When file was uploaded
    file_size = db.Column(db.Integer)  # File size in bytes
    
    # Data analysis results (computed during upload processing)
    total_measurements = db.Column(db.Integer)  # Number of data points in CSV
    date_range_start = db.Column(db.DateTime)  # First timestamp in data
    date_range_end = db.Column(db.DateTime)  # Last timestamp in data
    total_growth = db.Column(db.Float)  # Total growth in mm (last - first measurement)
    decrease_count = db.Column(db.Integer)  # Number of periods where growth decreased
    ml_accuracy = db.Column(db.Float)  # Machine learning model R² score
    is_active = db.Column(db.Boolean, default=True)  # Soft delete flag

class MergedDataset(db.Model):
    """
    MergedDataset model for storing information about combined datasets.
    Allows users to merge multiple CSV uploads for comparative analysis.
    """
    __tablename__ = 'merged_dataset'  # Database table name
    
    # Primary key - unique identifier for each merged dataset
    id = db.Column(db.Integer, primary_key=True)
    
    # Dataset information
    name = db.Column(db.String(120), nullable=False)  # User-defined name for merged dataset
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to User
    upload_ids = db.Column(db.Text)  # JSON string containing IDs of source uploads
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # When dataset was created
    description = db.Column(db.Text)  # Optional description of the merged dataset

# =============================================================================
# FLASK-WTF FORMS
# =============================================================================
# These classes define web forms with validation for user input

class LoginForm(FlaskForm):
    """
    Form for user login with username and password validation.
    """
    username = StringField('Username', validators=[DataRequired()])  # Required username field
    password = PasswordField('Password', validators=[DataRequired()])  # Required password field
    submit = SubmitField('Login')  # Submit button

class RegisterForm(FlaskForm):
    """
    Form for user registration with comprehensive validation.
    """
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])  # Username 4-20 chars
    email = StringField('Email', validators=[DataRequired(), Email()])  # Valid email address required
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])  # Password min 6 chars
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])  # Must match password
    submit = SubmitField('Register')  # Submit button

class UploadForm(FlaskForm):
    """
    Form for CSV file upload with optional description.
    """
    file = FileField('CSV File', validators=[DataRequired()])  # Required CSV file upload
    description = TextAreaField('Description (Optional)')  # Optional description text
    submit = SubmitField('Upload')  # Submit button

class MergeForm(FlaskForm):
    """
    Form for merging multiple datasets with name and description.
    """
    name = StringField('Dataset Name', validators=[DataRequired()])  # Required name for merged dataset
    description = TextAreaField('Description (Optional)')  # Optional description
    submit = SubmitField('Merge Datasets')  # Submit button

# =============================================================================
# FLASK-LOGIN USER LOADER
# =============================================================================
@login_manager.user_loader
def load_user(user_id):
    """
    Flask-Login callback function to load user from database by ID.
    This function is called by Flask-Login to get the current user.
    """
    return User.query.get(int(user_id))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def allowed_file(filename):
    """
    Security function to validate file extensions.
    
    Args:
        filename (str): The name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed (CSV), False otherwise
        
    Purpose:
        Prevents users from uploading potentially dangerous file types.
        Only CSV files are allowed for plant growth data analysis.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_csv_data(filepath):
    """
    Process and clean uploaded CSV data for plant growth analysis.
    
    This function performs comprehensive data validation, cleaning, and preprocessing
    to ensure the CSV data is ready for analysis and machine learning.
    
    Args:
        filepath (str): Path to the uploaded CSV file
        
    Returns:
        tuple: (df_cleaned, stats) where:
            - df_cleaned: Cleaned pandas DataFrame ready for analysis
            - stats: Dictionary containing data quality statistics
            
    Raises:
        Exception: If file cannot be processed or required columns are missing
        
    Data Processing Steps:
        1. Read CSV file using pandas
        2. Validate required columns exist (timestamp, distance_mm)
        3. Parse timestamp column to datetime format
        4. Convert distance_mm to numeric values
        5. Handle missing values by dropping incomplete rows
        6. Sort data chronologically
        7. Calculate summary statistics
    """
    try:
        # Step 1: Read CSV file using pandas
        # pandas automatically detects delimiters and handles various CSV formats
        df = pd.read_csv(filepath)
        
        # Step 2: Validate required columns exist
        # The application expects specific column names for plant growth data
        required_columns = ['timestamp', 'distance_mm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')
        
        # Step 3: Parse timestamp column as datetime
        # This converts string timestamps to pandas datetime objects for time-based analysis
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f'Error parsing timestamp column: {str(e)}')
        
        # Step 4: Ensure distance_mm is numeric
        # Convert to numeric, coercing invalid values to NaN for later removal
        try:
            df['distance_mm'] = pd.to_numeric(df['distance_mm'], errors='coerce')
        except Exception as e:
            raise ValueError(f'Error converting distance_mm to numeric: {str(e)}')
        
        # Step 5: Handle missing values
        initial_rows = len(df)  # Track original data size
        
        # Count missing values for reporting
        missing_timestamp = df['timestamp'].isna().sum()
        missing_distance = df['distance_mm'].isna().sum()
        
        # Drop rows with missing critical data
        # Both timestamp and distance_mm are required for meaningful analysis
        df_cleaned = df.dropna(subset=['timestamp', 'distance_mm'])
        final_rows = len(df_cleaned)
        dropped_rows = initial_rows - final_rows
        
        # Step 6: Sort by timestamp for chronological analysis
        # This ensures growth trends are analyzed in correct time order
        df_cleaned = df_cleaned.sort_values('timestamp')
        
        # Step 7: Calculate comprehensive summary statistics
        # These statistics provide insights into data quality and plant growth patterns
        stats = {
            'total_rows': initial_rows,  # Original number of data points
            'dropped_rows': dropped_rows,  # Number of rows removed due to missing data
            'final_rows': final_rows,  # Clean data points available for analysis
            'missing_timestamp': missing_timestamp,  # Count of missing timestamps
            'missing_distance': missing_distance,  # Count of missing distance values
            'min_distance': df_cleaned['distance_mm'].min(),  # Minimum plant height
            'max_distance': df_cleaned['distance_mm'].max(),  # Maximum plant height
            'mean_distance': df_cleaned['distance_mm'].mean(),  # Average plant height
            'variance_distance': df_cleaned['distance_mm'].var(),  # Variance in height measurements
            'std_distance': df_cleaned['distance_mm'].std(),  # Standard deviation of height
            'date_range': {  # Time span of the data
                'start': df_cleaned['timestamp'].min(),
                'end': df_cleaned['timestamp'].max()
            }
        }
        
        return df_cleaned, stats
        
    except Exception as e:
        raise Exception(f'Error processing CSV data: {str(e)}')

def create_growth_plot(df_cleaned, growth_stats=None, ml_analysis=None):
    """Create an interactive Plotly line graph for plant growth data with decrease highlighting and ML predictions."""
    try:
        # Create the main growth line
        trace = go.Scatter(
            x=df_cleaned['timestamp'],
            y=df_cleaned['distance_mm'],
            mode='lines+markers',
            name='Plant Growth',
            line=dict(
                color='#2E8B57',  # Sea green color
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=6,
                color='#32CD32',  # Lime green
                line=dict(width=2, color='#228B22')
            ),
            hovertemplate='<b>Date:</b> %{x}<br><b>Height:</b> %{y:.2f} mm<br><extra></extra>'
        )
        
        traces = [trace]
        
        # Add decrease highlighting if growth analysis is provided
        if growth_stats and growth_stats['growth_periods']:
            # Create decrease markers
            decrease_timestamps = [period['timestamp'] for period in growth_stats['growth_periods']]
            decrease_distances = [period['distance_mm'] for period in growth_stats['growth_periods']]
            decrease_amounts = [period['growth_difference'] for period in growth_stats['growth_periods']]
            
            decrease_trace = go.Scatter(
                x=decrease_timestamps,
                y=decrease_distances,
                mode='markers',
                name='Growth Decreases',
                marker=dict(
                    size=10,
                    color='#FF4444',  # Red color for decreases
                    symbol='triangle-down',
                    line=dict(width=2, color='#CC0000')
                ),
                hovertemplate='<b>Date:</b> %{x}<br><b>Height:</b> %{y:.2f} mm<br><b>Decrease:</b> %{customdata:.2f} mm<br><extra></extra>',
                customdata=decrease_amounts
            )
            traces.append(decrease_trace)
            
            # Add decrease annotations
            annotations = []
            for i, period in enumerate(growth_stats['growth_periods']):
                annotations.append(
                    dict(
                        x=period['timestamp'],
                        y=period['distance_mm'],
                        text=f"↓{abs(period['growth_difference']):.1f}mm",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='#FF4444',
                        ax=0,
                        ay=-30,
                        font=dict(color='#FF4444', size=10),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#FF4444',
                        borderwidth=1
                    )
                )
        else:
            annotations = []
        
        # Add ML prediction traces if available
        if ml_analysis:
            # Add predicted values trace
            pred_trace = go.Scatter(
                x=ml_analysis['predictions']['timestamps'],
                y=ml_analysis['predictions']['predicted'],
                mode='lines',
                name='ML Prediction',
                line=dict(
                    color='#FF6B35',  # Orange color
                    width=2,
                    dash='dash'
                ),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Height:</b> %{y:.2f} mm<br><extra></extra>'
            )
            traces.append(pred_trace)
            
            # Add future predictions trace
            future_trace = go.Scatter(
                x=ml_analysis['future_predictions']['timestamps'],
                y=ml_analysis['future_predictions']['predicted'],
                mode='lines+markers',
                name='Future Prediction',
                line=dict(
                    color='#8B5CF6',  # Purple color
                    width=2,
                    dash='dot'
                ),
                marker=dict(
                    size=4,
                    color='#8B5CF6'
                ),
                hovertemplate='<b>Date:</b> %{x}<br><b>Future Height:</b> %{y:.2f} mm<br><extra></extra>'
            )
            traces.append(future_trace)
        
        # Calculate growth rate (difference between consecutive measurements)
        df_with_growth = df_cleaned.copy()
        df_with_growth['growth_rate'] = df_with_growth['distance_mm'].diff()
        
        # Create layout
        layout = go.Layout(
            title=dict(
                text='🌱 Plant Growth Over Time',
                font=dict(size=24, color='#2E8B57'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text='Date & Time', font=dict(size=16, color='#333')),
                tickfont=dict(size=12),
                gridcolor='#E5E5E5',
                showgrid=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(
                title=dict(text='Distance (mm)', font=dict(size=16, color='#333')),
                tickfont=dict(size=12),
                gridcolor='#E5E5E5',
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E5E5E5',
                borderwidth=1
            ),
            margin=dict(l=60, r=30, t=80, b=60),
            height=500,
            annotations=annotations
        )
        
        # Create figure
        fig = go.Figure(data=traces, layout=layout)
        
        # Add growth rate as secondary y-axis if there are multiple data points
        if len(df_cleaned) > 1:
            # Create secondary trace for growth rate
            growth_trace = go.Scatter(
                x=df_with_growth['timestamp'][1:],  # Skip first point (no previous value)
                y=df_with_growth['growth_rate'][1:],
                mode='lines+markers',
                name='Growth Rate',
                yaxis='y2',
                line=dict(
                    color='#FF6347',  # Tomato color
                    width=2,
                    dash='dash'
                ),
                marker=dict(
                    size=4,
                    color='#FF4500'
                ),
                hovertemplate='<b>Date:</b> %{x}<br><b>Growth:</b> %{y:.2f} mm<br><extra></extra>'
            )
            
            fig.add_trace(growth_trace)
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title=dict(text="Growth Rate (mm)", font=dict(size=14, color='#FF6347')),
                    tickfont=dict(size=10, color='#FF6347'),
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
        
        # Configure interaction features
        fig.update_layout(
            dragmode='pan'  # Default to pan mode
        )
        
        # Convert to JSON for embedding in HTML
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
        
    except Exception as e:
        raise Exception(f'Error creating growth plot: {str(e)}')

def analyze_growth_patterns(df_cleaned):
    """Analyze growth patterns and identify decrease periods."""
    try:
        # Calculate growth differences between consecutive measurements
        df_analysis = df_cleaned.copy()
        df_analysis['growth_difference'] = df_analysis['distance_mm'].diff()
        df_analysis['growth_rate'] = df_analysis['growth_difference'] / df_analysis['distance_mm'].shift(1) * 100
        
        # Identify decrease periods (negative growth)
        decrease_periods = df_analysis[df_analysis['growth_difference'] < 0].copy()
        
        # Calculate statistics
        total_measurements = len(df_analysis)
        decrease_count = len(decrease_periods)
        decrease_percentage = (decrease_count / (total_measurements - 1)) * 100 if total_measurements > 1 else 0
        
        # Calculate growth statistics
        growth_stats = {
            'total_measurements': total_measurements,
            'decrease_count': decrease_count,
            'decrease_percentage': decrease_percentage,
            'max_growth': df_analysis['growth_difference'].max(),
            'min_growth': df_analysis['growth_difference'].min(),
            'avg_growth': df_analysis['growth_difference'].mean(),
            'total_growth': df_analysis['distance_mm'].iloc[-1] - df_analysis['distance_mm'].iloc[0],
            'growth_periods': []
        }
        
        # Process decrease periods for detailed analysis
        if not decrease_periods.empty:
            for idx, row in decrease_periods.iterrows():
                period_info = {
                    'timestamp': row['timestamp'],
                    'distance_mm': row['distance_mm'],
                    'growth_difference': row['growth_difference'],
                    'growth_rate_percent': row['growth_rate'],
                    'previous_distance': df_analysis.loc[idx - 1, 'distance_mm'] if idx > 0 else None
                }
                growth_stats['growth_periods'].append(period_info)
        
        # Identify consecutive decrease periods
        consecutive_decreases = []
        if not decrease_periods.empty:
            decrease_indices = decrease_periods.index.tolist()
            current_sequence = [decrease_indices[0]]
            
            for i in range(1, len(decrease_indices)):
                if decrease_indices[i] == decrease_indices[i-1] + 1:
                    current_sequence.append(decrease_indices[i])
                else:
                    if len(current_sequence) > 1:
                        consecutive_decreases.append(current_sequence)
                    current_sequence = [decrease_indices[i]]
            
            if len(current_sequence) > 1:
                consecutive_decreases.append(current_sequence)
        
        growth_stats['consecutive_decreases'] = consecutive_decreases
        
        return growth_stats, df_analysis
        
    except Exception as e:
        raise Exception(f'Error analyzing growth patterns: {str(e)}')

def create_ml_model(df_cleaned):
    """Create an enhanced ML model for growth prediction with improved accuracy."""
    try:
        if len(df_cleaned) < 3:
            raise Exception("Not enough data points for ML model training (minimum 3 required)")
        
        # Prepare enhanced features
        df_ml = df_cleaned.copy()
        
        # Convert timestamp to numeric features
        start_time = df_ml['timestamp'].min()
        df_ml['time_delta_seconds'] = (df_ml['timestamp'] - start_time).dt.total_seconds()
        df_ml['time_delta_hours'] = df_ml['time_delta_seconds'] / 3600
        df_ml['time_delta_days'] = df_ml['time_delta_hours'] / 24
        
        # Enhanced cyclical time features
        df_ml['hour_of_day'] = df_ml['timestamp'].dt.hour
        df_ml['day_of_week'] = df_ml['timestamp'].dt.dayofweek
        df_ml['day_of_month'] = df_ml['timestamp'].dt.day
        df_ml['month'] = df_ml['timestamp'].dt.month
        df_ml['quarter'] = df_ml['timestamp'].dt.quarter
        
        # Cyclical encoding for better time representation
        df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['hour_of_day'] / 24)
        df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['hour_of_day'] / 24)
        df_ml['day_sin'] = np.sin(2 * np.pi * df_ml['day_of_week'] / 7)
        df_ml['day_cos'] = np.cos(2 * np.pi * df_ml['day_of_week'] / 7)
        df_ml['month_sin'] = np.sin(2 * np.pi * df_ml['month'] / 12)
        df_ml['month_cos'] = np.cos(2 * np.pi * df_ml['month'] / 12)
        
        # Growth rate features
        df_ml['growth_rate'] = df_ml['distance_mm'].pct_change().fillna(0)
        df_ml['growth_acceleration'] = df_ml['growth_rate'].diff().fillna(0)
        df_ml['growth_jerk'] = df_ml['growth_acceleration'].diff().fillna(0)
        
        # Rolling statistics with multiple windows
        windows = [3, 5, min(7, len(df_ml) // 2)]
        for window in windows:
            if window <= len(df_ml):
                df_ml[f'rolling_mean_{window}'] = df_ml['distance_mm'].rolling(window=window, min_periods=1).mean()
                df_ml[f'rolling_std_{window}'] = df_ml['distance_mm'].rolling(window=window, min_periods=1).std().fillna(0)
                df_ml[f'rolling_min_{window}'] = df_ml['distance_mm'].rolling(window=window, min_periods=1).min()
                df_ml[f'rolling_max_{window}'] = df_ml['distance_mm'].rolling(window=window, min_periods=1).max()
        
        # Lag features
        for lag in [1, 2, 3]:
            if lag < len(df_ml):
                df_ml[f'distance_lag_{lag}'] = df_ml['distance_mm'].shift(lag).fillna(method='bfill')
                df_ml[f'growth_rate_lag_{lag}'] = df_ml['growth_rate'].shift(lag).fillna(0)
        
        # Polynomial features
        df_ml['time_delta_squared'] = df_ml['time_delta_seconds'] ** 2
        df_ml['time_delta_cubed'] = df_ml['time_delta_seconds'] ** 3
        df_ml['distance_squared'] = df_ml['distance_mm'] ** 2
        
        # Interaction features
        df_ml['time_growth_interaction'] = df_ml['time_delta_hours'] * df_ml['growth_rate']
        df_ml['hour_growth_interaction'] = df_ml['hour_of_day'] * df_ml['growth_rate']
        
        # Trend detection
        df_ml['is_increasing'] = (df_ml['growth_rate'] > 0).astype(int)
        df_ml['is_decreasing'] = (df_ml['growth_rate'] < 0).astype(int)
        
        # Create feature matrix with all engineered features
        feature_columns = [col for col in df_ml.columns if col not in ['timestamp', 'distance_mm', 'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'quarter']]
        
        X = df_ml[feature_columns].fillna(0).values
        y = df_ml['distance_mm'].values
        
        # Remove any infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data for training and validation
        if len(X) < 10:
            # For very small datasets, use all data
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            # Use time-based split
            split_idx = max(int(len(X) * 0.8), len(X) - 3)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced model selection
        from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=2, 
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Linear Regression': LinearRegression()
        }
        
        # Use simpler models for very small datasets
        if len(X) < 15:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Ridge Regression': Ridge(alpha=1.0),
                'Linear Regression': LinearRegression()
            }
        
        best_model = None
        best_score = -np.inf
        best_model_name = ""
        model_results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                model_results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                }
                
                # Select best model based on test R²
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Fallback to simple linear regression if all models fail
        if best_model is None:
            best_model = LinearRegression()
            best_model.fit(X_train_scaled, y_train)
            y_pred_test = best_model.predict(X_test_scaled)
            best_score = r2_score(y_test, y_pred_test)
            best_model_name = "Linear Regression (Fallback)"
        
        # Generate predictions for the entire dataset
        X_full_scaled = scaler.transform(X)
        y_pred_full = best_model.predict(X_full_scaled)
        
        # Create future predictions with uncertainty estimation
        last_time = df_ml['timestamp'].max()
        future_hours = np.linspace(0, (last_time - start_time).total_seconds() / 3600 * 0.2, 10)
        future_timestamps = [start_time + timedelta(hours=h) for h in future_hours]
        
        # Prepare future features
        future_df = pd.DataFrame({
            'timestamp': future_timestamps,
            'time_delta_seconds': future_hours * 3600,
            'time_delta_hours': future_hours,
            'time_delta_days': future_hours / 24,
            'time_delta_squared': (future_hours * 3600) ** 2,
            'time_delta_cubed': (future_hours * 3600) ** 3,
            'hour_of_day': [t.hour for t in future_timestamps],
            'day_of_week': [t.dayofweek for t in future_timestamps],
            'day_of_month': [t.day for t in future_timestamps],
            'month': [t.month for t in future_timestamps],
            'quarter': [t.quarter for t in future_timestamps]
        })
        
        # Add cyclical features for future data
        future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour_of_day'] / 24)
        future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour_of_day'] / 24)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        
        # Add other features for future data (using last known values)
        for col in feature_columns:
            if col not in future_df.columns:
                if 'rolling' in col or 'lag' in col or 'growth' in col:
                    future_df[col] = df_ml[col].iloc[-1] if col in df_ml.columns else 0
                else:
                    future_df[col] = 0
        
        X_future = future_df[feature_columns].fillna(0).values
        X_future_scaled = scaler.transform(X_future)
        y_future_pred = best_model.predict(X_future_scaled)
        
        # Calculate growth trend analysis
        actual_growth = df_ml['distance_mm'].iloc[-1] - df_ml['distance_mm'].iloc[0]
        predicted_growth = y_pred_full[-1] - y_pred_full[0]
        future_growth = y_future_pred[-1] - y_future_pred[0]
        
        # Calculate average growth rate
        time_span_hours = (df_ml['timestamp'].max() - df_ml['timestamp'].min()).total_seconds() / 3600
        avg_growth_rate = actual_growth / time_span_hours if time_span_hours > 0 else 0
        
        # Enhanced confidence calculation
        confidence_levels = {
            (0.95, 1.0): "Excellent",
            (0.85, 0.95): "Very High",
            (0.75, 0.85): "High", 
            (0.65, 0.75): "Good",
            (0.55, 0.65): "Moderate",
            (0.45, 0.55): "Low",
            (0.0, 0.45): "Very Low"
        }
        
        confidence = "Very Low"
        for (min_val, max_val), conf_name in confidence_levels.items():
            if min_val <= best_score < max_val:
                confidence = conf_name
                break
        
        # ML analysis results
        ml_analysis = {
            'best_model': best_model_name,
            'model_performance': model_results.get(best_model_name, {}),
            'all_model_scores': {name: results.get('test_r2', 0) for name, results in model_results.items()},
            'predictions': {
                'timestamps': df_ml['timestamp'].tolist(),
                'actual': df_ml['distance_mm'].tolist(),
                'predicted': y_pred_full.tolist()
            },
            'future_predictions': {
                'timestamps': future_timestamps,
                'predicted': y_future_pred.tolist()
            },
            'growth_analysis': {
                'actual_growth': actual_growth,
                'predicted_growth': predicted_growth,
                'future_growth': future_growth,
                'avg_growth_rate_mm_per_hour': avg_growth_rate,
                'model_accuracy_r2': best_score,
                'prediction_confidence': confidence,
                'model_type': best_model_name,
                'feature_importance': dict(zip(feature_columns, best_model.feature_importances_)) if hasattr(best_model, 'feature_importances_') else None
            }
        }
        
        return ml_analysis, scaler, feature_columns
        
    except Exception as e:
        raise Exception(f'Error creating ML model: {str(e)}')

def process_csv_data_from_df(df):
    """Process CSV data from a pandas DataFrame."""
    try:
        # Check if required columns exist
        required_columns = ['timestamp', 'distance_mm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')
        
        # Parse timestamp column as datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f'Error parsing timestamp column: {str(e)}')
        
        # Ensure distance_mm is numeric
        try:
            df['distance_mm'] = pd.to_numeric(df['distance_mm'], errors='coerce')
        except Exception as e:
            raise ValueError(f'Error converting distance_mm to numeric: {str(e)}')
        
        # Handle missing values
        initial_rows = len(df)
        
        # Check for missing values
        missing_timestamp = df['timestamp'].isna().sum()
        missing_distance = df['distance_mm'].isna().sum()
        
        # Drop rows with missing critical data
        df_cleaned = df.dropna(subset=['timestamp', 'distance_mm'])
        final_rows = len(df_cleaned)
        dropped_rows = initial_rows - final_rows
        
        # Sort by timestamp
        df_cleaned = df_cleaned.sort_values('timestamp')
        
        # Calculate summary statistics
        stats = {
            'total_rows': initial_rows,
            'dropped_rows': dropped_rows,
            'final_rows': final_rows,
            'missing_timestamp': missing_timestamp,
            'missing_distance': missing_distance,
            'min_distance': df_cleaned['distance_mm'].min(),
            'max_distance': df_cleaned['distance_mm'].max(),
            'mean_distance': df_cleaned['distance_mm'].mean(),
            'variance_distance': df_cleaned['distance_mm'].var(),
            'std_distance': df_cleaned['distance_mm'].std(),
            'date_range': {
                'start': df_cleaned['timestamp'].min(),
                'end': df_cleaned['timestamp'].max()
            }
        }
        
        return df_cleaned, stats
        
    except Exception as e:
        raise Exception(f'Error processing CSV data: {str(e)}')

def create_merged_plot(merged_data, growth_stats, ml_analysis):
    """Create a plot for merged datasets with different colors for each source."""
    try:
        traces = []
        colors = ['#2E8B57', '#FF6B35', '#8B5CF6', '#FF4444', '#4CAF50', '#FF9800']
        
        for i, df in enumerate(merged_data):
            color = colors[i % len(colors)]
            trace = go.Scatter(
                x=df['timestamp'],
                y=df['distance_mm'],
                mode='lines+markers',
                name=df['source_name'].iloc[0] if 'source_name' in df.columns else f'Dataset {i+1}',
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color)
            )
            traces.append(trace)
        
        # Add ML prediction if available
        if ml_analysis:
            combined_df = pd.concat(merged_data, ignore_index=True)
            pred_trace = go.Scatter(
                x=ml_analysis['predictions']['timestamps'],
                y=ml_analysis['predictions']['predicted'],
                mode='lines',
                name='ML Prediction',
                line=dict(color='#FF6B35', width=2, dash='dash')
            )
            traces.append(pred_trace)
        
        layout = go.Layout(
            title=dict(text='🌱 Merged Plant Growth Data', font=dict(size=24, color='#2E8B57'), x=0.5),
            xaxis=dict(title='Date & Time', type='date'),
            yaxis=dict(title='Distance (mm)'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        
        fig = go.Figure(data=traces, layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        raise Exception(f'Error creating merged plot: {str(e)}')

@app.route('/')
def home():
    """Home page - redirect to login if not authenticated."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if user already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'error')
            return render_template('register.html', form=form)
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered', 'error')
            return render_template('register.html', form=form)
        
        # Create new user
        user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with upload history."""
    uploads = Upload.query.filter_by(user_id=current_user.id, is_active=True).order_by(Upload.upload_date.desc()).all()
    merged_datasets = MergedDataset.query.filter_by(user_id=current_user.id).order_by(MergedDataset.created_at.desc()).all()
    return render_template('dashboard.html', uploads=uploads, merged_datasets=merged_datasets)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle CSV file upload with user authentication."""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        original_filename = file.filename
        filename = f"{current_user.id}_{uuid.uuid4().hex}_{secure_filename(original_filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process CSV data
            df_cleaned, stats = process_csv_data(filepath)
            growth_stats, df_analysis = analyze_growth_patterns(df_cleaned)
            
            # Try to create ML model
            try:
                ml_analysis, scaler, feature_columns = create_ml_model(df_cleaned)
                ml_accuracy = ml_analysis['growth_analysis']['model_accuracy_r2']
            except Exception as e:
                ml_accuracy = None
            
            # Save upload record to database
            upload_record = Upload(
                filename=filename,
                original_filename=original_filename,
                user_id=current_user.id,
                file_size=os.path.getsize(filepath),
                total_measurements=len(df_cleaned),
                date_range_start=df_cleaned['timestamp'].min(),
                date_range_end=df_cleaned['timestamp'].max(),
                total_growth=df_cleaned['distance_mm'].iloc[-1] - df_cleaned['distance_mm'].iloc[0],
                decrease_count=growth_stats['decrease_count'],
                ml_accuracy=ml_accuracy
            )
            db.session.add(upload_record)
            db.session.commit()
            
            flash('File uploaded and processed successfully!', 'success')
            return redirect(url_for('display_data', upload_id=upload_record.id))
            
        except Exception as e:
            flash(f'Error processing CSV file: {str(e)}', 'error')
            return redirect(url_for('dashboard'))
    
    flash('Invalid file type. Please upload a CSV file.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/display/<int:upload_id>')
@login_required
def display_data(upload_id):
    """Display the uploaded CSV data with statistics."""
    upload = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first()
    if not upload:
        flash('Upload not found', 'error')
        return redirect(url_for('dashboard'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Process CSV data
        df_cleaned, stats = process_csv_data(filepath)
        
        # Get basic info about the dataset
        total_rows = stats['total_rows']
        final_rows = stats['final_rows']
        columns = list(df_cleaned.columns)
        
        # Display first 10 rows
        preview_data = df_cleaned.head(10).to_html(classes='table table-striped', escape=False)
        
        # Analyze growth patterns
        growth_stats, df_analysis = analyze_growth_patterns(df_cleaned)
        
        # Create ML model for growth prediction
        try:
            ml_analysis, scaler, feature_columns = create_ml_model(df_cleaned)
        except Exception as e:
            print(f"ML model creation failed: {str(e)}")
            ml_analysis = None
        
        # Create interactive plot with growth analysis and ML predictions
        plot_json = create_growth_plot(df_cleaned, growth_stats, ml_analysis)
        
        return render_template('display.html', 
                             upload=upload,
                             total_rows=total_rows,
                             final_rows=final_rows,
                             columns=columns,
                             preview_data=preview_data,
                             stats=stats,
                             growth_stats=growth_stats,
                             ml_analysis=ml_analysis,
                             plot_json=plot_json,
                             df=df_cleaned)
    
    except Exception as e:
        flash(f'Error processing CSV file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/merge', methods=['GET', 'POST'])
@login_required
def merge_datasets():
    """Merge multiple datasets."""
    form = MergeForm()
    uploads = Upload.query.filter_by(user_id=current_user.id, is_active=True).all()
    
    if form.validate_on_submit():
        selected_uploads = request.form.getlist('uploads')
        if len(selected_uploads) < 2:
            flash('Please select at least 2 datasets to merge', 'error')
            return render_template('merge.html', form=form, uploads=uploads)
        
        try:
            # Load and merge datasets
            merged_data = []
            for upload_id in selected_uploads:
                upload = Upload.query.get(upload_id)
                if upload and upload.user_id == current_user.id:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
                    df, _ = process_csv_data(filepath)
                    df['source_upload_id'] = upload_id
                    df['source_name'] = upload.original_filename
                    merged_data.append(df)
            
            if not merged_data:
                flash('No valid datasets found', 'error')
                return render_template('merge.html', form=form, uploads=uploads)
            
            # Combine all datasets
            combined_df = pd.concat(merged_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp')
            
            # Save merged dataset
            merged_dataset = MergedDataset(
                name=form.name.data,
                user_id=current_user.id,
                upload_ids=json.dumps(selected_uploads),
                description=form.description.data
            )
            db.session.add(merged_dataset)
            db.session.commit()
            
            flash('Datasets merged successfully!', 'success')
            return redirect(url_for('display_merged', merge_id=merged_dataset.id))
            
        except Exception as e:
            flash(f'Error merging datasets: {str(e)}', 'error')
    
    return render_template('merge.html', form=form, uploads=uploads)

@app.route('/merged/<int:merge_id>')
@login_required
def display_merged(merge_id):
    """Display merged dataset analysis."""
    merged_dataset = MergedDataset.query.filter_by(id=merge_id, user_id=current_user.id).first()
    if not merged_dataset:
        flash('Merged dataset not found', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Load and combine datasets
        upload_ids = json.loads(merged_dataset.upload_ids)
        merged_data = []
        
        for upload_id in upload_ids:
            upload = Upload.query.get(upload_id)
            if upload and upload.user_id == current_user.id:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
                df, _ = process_csv_data(filepath)
                df['source_upload_id'] = upload_id
                df['source_name'] = upload.original_filename
                merged_data.append(df)
        
        if not merged_data:
            flash('No data found in merged dataset', 'error')
            return redirect(url_for('dashboard'))
        
        combined_df = pd.concat(merged_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        # Process merged data
        df_cleaned, stats = process_csv_data_from_df(combined_df)
        growth_stats, df_analysis = analyze_growth_patterns(df_cleaned)
        
        # Create ML model
        try:
            ml_analysis, scaler, feature_columns = create_ml_model(df_cleaned)
        except Exception as e:
            ml_analysis = None
        
        # Create plot with multiple datasets
        plot_json = create_merged_plot(merged_data, growth_stats, ml_analysis)
        
        return render_template('merged_display.html',
                             merged_dataset=merged_dataset,
                             stats=stats,
                             growth_stats=growth_stats,
                             ml_analysis=ml_analysis,
                             plot_json=plot_json,
                             source_data=merged_data)
    
    except Exception as e:
        flash(f'Error processing merged dataset: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/data/<filename>')
def get_data(filename):
    """API endpoint to get CSV data as JSON."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        df_cleaned, stats = process_csv_data(filepath)
        # Convert datetime to string for JSON serialization
        df_json = df_cleaned.copy()
        df_json['timestamp'] = df_json['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(df_json.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<int:upload_id>')
@login_required
def download_csv(upload_id):
    """Download processed CSV with growth analysis."""
    upload = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first()
    if not upload:
        flash('Upload not found', 'error')
        return redirect(url_for('dashboard'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Process CSV data
        df_cleaned, stats = process_csv_data(filepath)
        growth_stats, df_analysis = analyze_growth_patterns(df_cleaned)
        
        # Create enhanced dataframe with all analysis
        df_download = df_analysis.copy()
        df_download['growth_difference_mm'] = df_download['growth_difference']
        df_download['growth_rate_percent'] = df_download['growth_rate']
        df_download['is_decrease'] = df_download['growth_difference'] < 0
        
        # Reorder columns for better readability
        columns_order = ['timestamp', 'distance_mm', 'growth_difference_mm', 'growth_rate_percent', 'is_decrease']
        df_download = df_download[columns_order]
        
        # Create CSV in memory
        output = io.StringIO()
        df_download.to_csv(output, index=False)
        output.seek(0)
        
        # Create response
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=f'processed_{upload.original_filename}',
            mimetype='text/csv'
        )
        
    except Exception as e:
        flash(f'Error creating download file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/summary/<filename>')
def get_summary(filename):
    """API endpoint to get summary data for the home page."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Process CSV data
        df_cleaned, stats = process_csv_data(filepath)
        growth_stats, df_analysis = analyze_growth_patterns(df_cleaned)
        
        # Try to create ML model
        try:
            ml_analysis, scaler, feature_columns = create_ml_model(df_cleaned)
        except Exception as e:
            ml_analysis = None
        
        # Create summary data
        summary = {
            'filename': filename,
            'total_measurements': len(df_cleaned),
            'date_range': {
                'start': df_cleaned['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                'end': df_cleaned['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
                'duration_days': (df_cleaned['timestamp'].max() - df_cleaned['timestamp'].min()).days
            },
            'growth_stats': {
                'total_growth': df_cleaned['distance_mm'].iloc[-1] - df_cleaned['distance_mm'].iloc[0],
                'avg_growth_rate': growth_stats['avg_growth'],
                'decrease_count': growth_stats['decrease_count'],
                'decrease_percentage': growth_stats['decrease_percentage']
            },
            'ml_analysis': ml_analysis['growth_analysis'] if ml_analysis else None
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
