import os
import sqlite3
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import requests
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import hashlib
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Response, jsonify, send_from_directory, send_file
import timm
from torchvision import transforms
import warnings
import base64
from io import BytesIO, StringIO
import exifread
import folium
from geopy.geocoders import Nominatim
import pytz
from timezonefinder import TimezoneFinder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-this'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max for videos

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['REPORTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Model Configuration
MODEL_PATH = "flood_vit_model.pth"
CLASS_NAMES = ["Flood", "NonFlood"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenWeather API Configuration - REPLACE WITH YOUR ACTUAL API KEY
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY_HERE"  # Get from: https://openweathermap.org/api

# Email Configuration (Optional)
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver_email@gmail.com"

# ==================== DATABASE SETUP ====================
def init_database():
    """Initialize SQLite database for detection history"""
    conn = sqlite3.connect('flood_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        flood_prob REAL,
        nonflood_prob REAL,
        severity TEXT,
        location_name TEXT,
        latitude REAL,
        longitude REAL,
        weather_data TEXT,
        filename TEXT,
        processed_filename TEXT,
        gradcam_filename TEXT,
        alert_sent BOOLEAN DEFAULT 0,
        location_source TEXT DEFAULT 'unknown'
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS location_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ip_address TEXT,
        latitude REAL,
        longitude REAL,
        city TEXT,
        region TEXT,
        country TEXT,
        timezone TEXT,
        location_source TEXT DEFAULT 'unknown'
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

init_database()

def check_database_integrity():
    """Check if database tables exist and are accessible"""
    try:
        conn = sqlite3.connect('flood_detection.db')
        cursor = conn.cursor()
        
        # Check if detections table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        if not cursor.fetchone():
            print("⚠ Detections table missing, recreating...")
            init_database()
        
        # Check if location_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='location_history'")
        if not cursor.fetchone():
            print("⚠ Location history table missing, recreating...")
            init_database()
        
        conn.close()
        print("✓ Database integrity check passed")
    except Exception as e:
        print(f"✗ Database integrity check failed: {e}")
        # Try to recreate database
        try:
            if os.path.exists('flood_detection.db'):
                os.remove('flood_detection.db')
            init_database()
            print("✓ Database recreated successfully")
        except Exception as e2:
            print(f"✗ Failed to recreate database: {e2}")

check_database_integrity()

# ==================== MODEL LOADING ====================
def load_model():
    """Load and configure the Vision Transformer model"""
    try:
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠ Model file not found: {MODEL_PATH}")
            print("Using random weights for demonstration")
        
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        # Return a dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.randn(x.shape[0], 2)
        return DummyModel()

model = load_model()

# ==================== IMPROVED GRAD-CAM IMPLEMENTATION ====================
class ViTAttentionGradCAM:
    """Improved Grad-CAM implementation specifically for Vision Transformers"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Register hooks for the last attention layer
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations from attention layers"""
        def forward_hook(module, input, output):
            self.activations = output
            return output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return grad_input
        
        # For ViT models, we'll hook the last attention layer
        # Different ViT architectures have different structures
        if hasattr(self.model, 'blocks'):
            # Standard ViT architecture
            last_block = self.model.blocks[-1]
            if hasattr(last_block, 'attn'):
                # Hook the attention output
                last_block.attn.register_forward_hook(forward_hook)
                last_block.attn.register_backward_hook(backward_hook)
                print("✓ Grad-CAM hooks registered on attention layer")
            else:
                # Fallback to the last block
                last_block.register_forward_hook(forward_hook)
                last_block.register_backward_hook(backward_hook)
                print("⚠ Using block-level hooks for Grad-CAM")
        elif hasattr(self.model, 'transformer'):
            # Some other ViT variants
            self.model.transformer.register_forward_hook(forward_hook)
            self.model.transformer.register_backward_hook(backward_hook)
            print("⚠ Using transformer-level hooks for Grad-CAM")
        else:
            print("⚠ Could not find suitable layers for Grad-CAM hooks")
    
    def _compute_attention_weights(self, attention_maps, gradients):
        """Compute attention weights from gradients"""
        # Global average pooling of gradients across spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight the attention maps
        cam = (attention_maps * weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        return cam
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for ViT"""
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Create one-hot encoding for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Get gradients and activations
            if self.gradients is None or self.activations is None:
                print("⚠ Grad-CAM: No gradients or activations captured")
                return self._generate_fallback_cam(input_tensor.shape[2:])
            
            gradients = self.gradients.detach().cpu()
            activations = self.activations.detach().cpu()
            
            # Reshape for ViT (assuming [batch, num_patches, embed_dim])
            if len(activations.shape) == 3:
                # For ViT, we need to reshape from patches to spatial dimensions
                batch_size, num_patches, embed_dim = activations.shape
                
                # Calculate grid size (assuming square patches)
                grid_size = int(np.sqrt(num_patches))
                
                if grid_size * grid_size == num_patches:
                    # Reshape to spatial format [batch, embed_dim, grid_size, grid_size]
                    activations = activations.permute(0, 2, 1).reshape(
                        batch_size, embed_dim, grid_size, grid_size)
                    gradients = gradients.permute(0, 2, 1).reshape(
                        batch_size, embed_dim, grid_size, grid_size)
                    
                    # Compute CAM using weights
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (activations * weights).sum(dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # Resize to input image size
                    cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                                       mode='bilinear', align_corners=False)
                    cam = cam.squeeze().numpy()
                    
                    # Normalize
                    if cam.max() - cam.min() > 1e-8:
                        cam = (cam - cam.min()) / (cam.max() - cam.min())
                    else:
                        cam = np.zeros_like(cam)
                    
                    return cam
            
            # Fallback for other architectures
            return self._generate_fallback_cam(input_tensor.shape[2:])
            
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return self._generate_fallback_cam(input_tensor.shape[2:])
    
    def _generate_fallback_cam(self, target_size):
        """Generate a fallback CAM when primary method fails"""
        h, w = target_size
        cam = np.zeros((h, w), dtype=np.float32)
        
        # Create a simple centered gaussian as fallback
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        sigma = min(h, w) / 6
        cam = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        return cam

# Initialize Grad-CAM
gradcam = ViTAttentionGradCAM(model)

# ==================== TRANSFORMS ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==================== LOCATION SERVICES ====================
class LocationService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="flood_detection_app")
        self.tf = TimezoneFinder()
    
    def get_location_from_ip(self, ip_address=None):
        """Get location from IP address using free IP-API"""
        try:
            if not ip_address:
                # Try to get real IP from request headers
                if request:
                    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
                else:
                    ip_address = "auto"
            
            if ip_address == "auto" or ip_address == "127.0.0.1":
                # Use free IP geolocation service
                response = requests.get('http://ip-api.com/json/', timeout=5)
            else:
                response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
            
            data = response.json()
            
            if data['status'] == 'success':
                timezone_str = self.tf.timezone_at(lng=data['lon'], lat=data['lat'])
                location_name = self.get_location_name(data['lat'], data['lon'])
                return {
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'city': data['city'],
                    'region': data['regionName'],
                    'country': data['country'],
                    'country_code': data['countryCode'],
                    'timezone': timezone_str,
                    'isp': data.get('isp', ''),
                    'ip': data.get('query', ip_address),
                    'location_name': location_name,
                    'success': True,
                    'source': 'ip_api'
                }
        except Exception as e:
            print(f"Location detection error: {e}")
        
        # Return default location (Mumbai) if detection fails
        return {
            'latitude': 19.0760,
            'longitude': 72.8777,
            'city': 'Mumbai',
            'region': 'Maharashtra',
            'country': 'India',
            'country_code': 'IN',
            'timezone': 'Asia/Kolkata',
            'isp': 'Unknown',
            'ip': ip_address or 'Unknown',
            'location_name': 'Mumbai, Maharashtra, India',
            'success': False,
            'source': 'default'
        }
    
    def get_location_from_coords(self, lat, lon):
        """Get location information from coordinates"""
        try:
            timezone_str = self.tf.timezone_at(lng=lon, lat=lat)
            location_name = self.get_location_name(lat, lon)
            
            return {
                'latitude': lat,
                'longitude': lon,
                'city': location_name.split(',')[0] if location_name else 'Unknown',
                'region': 'Manual Input',
                'country': 'Unknown',
                'country_code': 'XX',
                'timezone': timezone_str or 'UTC',
                'isp': 'Manual Location',
                'ip': request.remote_addr if request else 'Manual',
                'location_name': location_name,
                'success': True,
                'source': 'manual'
            }
        except Exception as e:
            print(f"Location from coords error: {e}")
            return {
                'latitude': lat,
                'longitude': lon,
                'city': 'Unknown',
                'region': 'Unknown',
                'country': 'Unknown',
                'country_code': 'XX',
                'timezone': 'UTC',
                'isp': 'Manual',
                'ip': 'Manual',
                'location_name': f'{lat:.4f}, {lon:.4f}',
                'success': True,
                'source': 'manual'
            }
    
    def get_location_from_city(self, city_name):
        """Get coordinates from city name"""
        try:
            location = self.geolocator.geocode(city_name, timeout=10)
            if location:
                timezone_str = self.tf.timezone_at(lng=location.longitude, lat=location.latitude)
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'city': location.address.split(',')[0],
                    'region': 'Searched',
                    'country': 'Unknown',
                    'country_code': 'XX',
                    'timezone': timezone_str or 'UTC',
                    'isp': 'Search',
                    'ip': request.remote_addr if request else 'Search',
                    'location_name': location.address,
                    'success': True,
                    'source': 'search'
                }
        except Exception as e:
            print(f"City location error: {e}")
        
        return None
    
    def get_location_name(self, lat, lon):
        """Get human-readable location name from coordinates"""
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", timeout=10, language='en')
            return location.address if location else f"{lat:.4f}, {lon:.4f}"
        except:
            return f"{lat:.4f}, {lon:.4f}"
    
    def save_location_to_db(self, location_data, ip_address):
        """Save location data to database"""
        try:
            conn = sqlite3.connect('flood_detection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO location_history 
            (ip_address, latitude, longitude, city, region, country, timezone, location_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ip_address, location_data['latitude'], location_data['longitude'], 
                  location_data.get('city', 'Unknown'), location_data.get('region', 'Unknown'), 
                  location_data.get('country', 'Unknown'), location_data.get('timezone', 'UTC'),
                  location_data.get('source', 'unknown')))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database save error: {e}")
    
    def get_weather_alerts(self, lat, lon):
        """Get weather alerts for the location"""
        try:
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,daily&appid={OPENWEATHER_API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('alerts', [])
        except:
            pass
        return []

location_service = LocationService()

# ==================== WEATHER SERVICE ====================
class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_current_weather(self, lat, lon, location_name=None):
        """Get current weather data from OpenWeather API"""
        cache_key = f"{lat}_{lon}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            if self.api_key == "YOUR_OPENWEATHER_API_KEY_HERE":
                return self._get_mock_weather(lat, lon, location_name)
            
            # Get current weather
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get additional forecast data
                forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric&cnt=5"
                forecast_response = requests.get(forecast_url, timeout=10)
                forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
                
                weather_data = {
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'temp_min': data['main']['temp_min'],
                    'temp_max': data['main']['temp_max'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'sea_level': data['main'].get('sea_level'),
                    'grnd_level': data['main'].get('grnd_level'),
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'wind_gust': data['wind'].get('gust', 0),
                    'weather_main': data['weather'][0]['main'],
                    'weather_description': data['weather'][0]['description'],
                    'weather_icon': data['weather'][0]['icon'],
                    'cloudiness': data['clouds']['all'],
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'rain_1h': data.get('rain', {}).get('1h', 0),
                    'rain_3h': data.get('rain', {}).get('3h', 0),
                    'snow_1h': data.get('snow', {}).get('1h', 0),
                    'snow_3h': data.get('snow', {}).get('3h', 0),
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
                    'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
                    'timezone_offset': data['timezone'],
                    'location_name': location_name or data['name'],
                    'country': data['sys']['country']
                }
                
                # Add forecast if available
                if forecast_data:
                    weather_data['forecast'] = []
                    for item in forecast_data['list'][:3]:  # Next 3 forecasts
                        weather_data['forecast'].append({
                            'time': datetime.fromtimestamp(item['dt']).strftime('%H:%M'),
                            'temp': item['main']['temp'],
                            'weather': item['weather'][0]['description'],
                            'icon': item['weather'][0]['icon'],
                            'pop': item.get('pop', 0) * 100  # Probability of precipitation
                        })
                
                # Cache the result
                self.cache[cache_key] = (weather_data, current_time)
                return weather_data
                
        except Exception as e:
            print(f"Weather API error: {e}")
        
        # Return mock data if API fails
        return self._get_mock_weather(lat, lon, location_name)
    
    def _get_mock_weather(self, lat, lon, location_name=None):
        """Return mock weather data for testing"""
        return {
            'temperature': 28.5,
            'feels_like': 30.2,
            'temp_min': 26.0,
            'temp_max': 31.0,
            'humidity': 75,
            'pressure': 1013,
            'wind_speed': 12.5,
            'wind_direction': 180,
            'weather_main': 'Rain',
            'weather_description': 'moderate rain',
            'weather_icon': '10d',
            'cloudiness': 85,
            'visibility': 5,
            'rain_1h': 2.5,
            'sunrise': '06:15',
            'sunset': '18:45',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location_name': location_name or 'Mumbai',
            'country': 'IN',
            'forecast': [
                {'time': '15:00', 'temp': 29.0, 'weather': 'light rain', 'icon': '10d', 'pop': 60},
                {'time': '18:00', 'temp': 27.5, 'weather': 'overcast clouds', 'icon': '04d', 'pop': 40},
                {'time': '21:00', 'temp': 26.0, 'weather': 'scattered clouds', 'icon': '03n', 'pop': 20}
            ]
        }
    
    def get_flood_risk_assessment(self, weather_data, flood_prob):
        """Assess flood risk based on weather conditions and model prediction"""
        risk_factors = []
        risk_score = 0
        
        # Rainfall analysis
        if weather_data.get('rain_1h', 0) > 10:
            risk_factors.append(f"Heavy rainfall detected: {weather_data['rain_1h']}mm in last hour")
            risk_score += 30
        elif weather_data.get('rain_1h', 0) > 5:
            risk_factors.append(f"Moderate rainfall: {weather_data['rain_1h']}mm in last hour")
            risk_score += 15
        
        # Humidity analysis
        if weather_data['humidity'] > 85:
            risk_factors.append(f"Very high humidity: {weather_data['humidity']}%")
            risk_score += 10
        elif weather_data['humidity'] > 75:
            risk_factors.append(f"High humidity: {weather_data['humidity']}%")
            risk_score += 5
        
        # Pressure analysis (low pressure often means storms)
        if weather_data['pressure'] < 1000:
            risk_factors.append(f"Low atmospheric pressure: {weather_data['pressure']} hPa")
            risk_score += 20
        elif weather_data['pressure'] < 1010:
            risk_factors.append(f"Below normal pressure: {weather_data['pressure']} hPa")
            risk_score += 10
        
        # Wind analysis
        if weather_data['wind_speed'] > 15:
            risk_factors.append(f"Strong winds: {weather_data['wind_speed']} m/s")
            risk_score += 15
        
        # Cloudiness
        if weather_data['cloudiness'] > 80:
            risk_factors.append(f"Heavy cloud cover: {weather_data['cloudiness']}%")
            risk_score += 5
        
        # Visibility
        if weather_data['visibility'] < 2:
            risk_factors.append(f"Poor visibility: {weather_data['visibility']} km")
            risk_score += 10
        
        # Model prediction weight
        if flood_prob > 80:
            risk_factors.append(f"Very high flood probability from AI model: {flood_prob}%")
            risk_score += 40
        elif flood_prob > 60:
            risk_factors.append(f"High flood probability from AI model: {flood_prob}%")
            risk_score += 25
        elif flood_prob > 40:
            risk_factors.append(f"Moderate flood probability from AI model: {flood_prob}%")
            risk_score += 15
        
        # Weather description analysis
        weather_desc = weather_data['weather_description'].lower()
        if 'heavy' in weather_desc and 'rain' in weather_desc:
            risk_score += 25
        elif 'thunderstorm' in weather_desc:
            risk_score += 20
        elif 'rain' in weather_desc:
            risk_score += 15
        elif 'drizzle' in weather_desc:
            risk_score += 5
        
        # Determine overall risk level
        if risk_score >= 70:
            risk_level = "VERY HIGH"
            risk_color = "darkred"
        elif risk_score >= 50:
            risk_level = "HIGH"
            risk_color = "red"
        elif risk_score >= 30:
            risk_level = "MODERATE"
            risk_color = "orange"
        elif risk_score >= 15:
            risk_level = "LOW"
            risk_color = "yellow"
        else:
            risk_level = "VERY LOW"
            risk_color = "green"
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'total_factors': len(risk_factors)
        }

weather_service = WeatherService(OPENWEATHER_API_KEY)

# ==================== HELPER FUNCTIONS ====================
def get_severity_level(flood_prob):
    """Determine flood severity level based on probability"""
    if flood_prob < 30:
        return "SAFE", "success", "No flood threat detected"
    elif 30 <= flood_prob < 50:
        return "LOW", "info", "Low flood probability"
    elif 50 <= flood_prob < 70:
        return "MODERATE", "warning", "Moderate flood probability - Stay alert"
    elif 70 <= flood_prob < 85:
        return "HIGH", "danger", "High flood probability - Take precautions"
    else:  # flood_prob >= 85
        return "EXTREME", "dark", "Extreme flood probability - Immediate action required"

def generate_gradcam_visualization(image_tensor, original_image, prediction_index):
    """Generate Grad-CAM visualization and save it"""
    try:
        # Generate Grad-CAM heatmap
        cam = gradcam.generate_cam(image_tensor, prediction_index)
        
        if cam is None:
            return None
        
        # Convert original image to numpy
        if isinstance(original_image, Image.Image):
            original_np = np.array(original_image.resize((224, 224)))
        else:
            original_np = original_image
        
        # Handle grayscale images
        if len(original_np.shape) == 2:
            original_np = np.stack([original_np] * 3, axis=-1)
        
        # Normalize original image
        original_np = original_np.astype(np.float32) / 255.0
        
        # Ensure cam is 2D
        if len(cam.shape) == 3:
            cam = cam[0]
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        
        # Overlay heatmap on original image
        alpha = 0.5
        overlay = original_np * (1 - alpha) + heatmap * alpha
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        gradcam_img = Image.fromarray(overlay)
        
        # Add prediction text - FIXED: Use the correct prediction index
        draw = ImageDraw.Draw(gradcam_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Try to load default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Get the actual flood probability - FIXED: Calculate from model outputs
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            flood_prob = probs[0][0].item() * 100  # Class 0 is Flood
        
        # Add prediction overlay - FIXED: Show correct probability
        prediction_text = f"Flood Probability: {flood_prob:.1f}%"
        
        # Determine text color based on probability
        if flood_prob > 70:
            text_color = (255, 0, 0)  # Red for high probability
        elif flood_prob > 50:
            text_color = (255, 165, 0)  # Orange for moderate
        else:
            text_color = (0, 255, 0)  # Green for low
        
        # Draw semi-transparent background for text
        if font:
            # Estimate text size
            text_bbox = draw.textbbox((0, 0), prediction_text, font=font)
            text_width = text_bbox[2] - text_bbox[0] + 40
            text_height = text_bbox[3] - text_bbox[1] + 20
            
            # Draw rectangle
            draw.rectangle([10, 10, 10 + text_width, 10 + text_height], 
                          fill=(0, 0, 0, 128))
            
            draw.text((30, 20), prediction_text, font=font, fill=text_color)
        else:
            # Fallback without font
            draw.text((30, 20), prediction_text, fill=text_color)
        
        return gradcam_img
        
    except Exception as e:
        print(f"Grad-CAM visualization error: {e}")
        import traceback
        traceback.print_exc()
        return None
def extract_exif_location(image_path):
    """Extract GPS coordinates from image EXIF data"""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            # Extract latitude
            lat_deg = tags['GPS GPSLatitude'].values[0].num / tags['GPS GPSLatitude'].values[0].den
            lat_min = tags['GPS GPSLatitude'].values[1].num / tags['GPS GPSLatitude'].values[1].den
            lat_sec = tags['GPS GPSLatitude'].values[2].num / tags['GPS GPSLatitude'].values[2].den
            latitude = lat_deg + (lat_min / 60) + (lat_sec / 3600)
            
            # Extract longitude
            lon_deg = tags['GPS GPSLongitude'].values[0].num / tags['GPS GPSLongitude'].values[0].den
            lon_min = tags['GPS GPSLongitude'].values[1].num / tags['GPS GPSLongitude'].values[1].den
            lon_sec = tags['GPS GPSLongitude'].values[2].num / tags['GPS GPSLongitude'].values[2].den
            longitude = lon_deg + (lon_min / 60) + (lon_sec / 3600)
            
            # Check if South or West
            if tags['GPS GPSLatitudeRef'].values == 'S':
                latitude = -latitude
            if tags['GPS GPSLongitudeRef'].values == 'W':
                longitude = -longitude
            
            return latitude, longitude
    except:
        pass
    
    return None, None

def process_video_file(video_path, location_data):
    """Process video file frame by frame"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Process every nth frame for efficiency
        frame_skip = max(1, int(fps / 2))  # 2 frames per second
        
        predictions = []
        flood_frames = []
        gradcam_frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Convert frame to PIL Image
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Process frame
                result = process_image(img_pil)
                predictions.append(result)
                
                # Store flood frames for Grad-CAM
                if result['flood_prob'] > 50 and len(flood_frames) < 3:
                    flood_frames.append((frame_count, img_pil, result))
                
                # Generate Grad-CAM for key flood frames
                if len(gradcam_frames) < 2 and result['flood_prob'] > 60:
                    image_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        prediction_index = 0 if outputs[0][0] > outputs[0][1] else 1
                    
                    gradcam_img = generate_gradcam_visualization(image_tensor, img_pil, prediction_index)
                    if gradcam_img:
                        # Save Grad-CAM frame
                        gradcam_filename = f"gradcam_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(gradcam_frames)}.jpg"
                        gradcam_path = os.path.join(app.config['PROCESSED_FOLDER'], gradcam_filename)
                        gradcam_img.save(gradcam_path)
                        gradcam_frames.append(f"/static/processed/{gradcam_filename}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        if predictions:
            flood_probs = [p['flood_prob'] for p in predictions]
            avg_flood_prob = np.mean(flood_probs)
            max_flood_prob = np.max(flood_probs)
            min_flood_prob = np.min(flood_probs)
            
            # Count flood frames
            flood_frame_count = sum(1 for p in predictions if p['flood_prob'] > 50)
            flood_percentage = (flood_frame_count / len(predictions)) * 100
            
            severity, severity_color, message = get_severity_level(avg_flood_prob)
            
            return {
                'type': 'video',
                'avg_flood_prob': round(avg_flood_prob, 2),
                'max_flood_prob': round(max_flood_prob, 2),
                'min_flood_prob': round(min_flood_prob, 2),
                'flood_percentage': round(flood_percentage, 2),
                'total_frames': total_frames,
                'processed_frames': len(predictions),
                'flood_frames': flood_frame_count,
                'fps': fps,
                'duration': round(duration, 2),
                'severity': severity,
                'severity_color': severity_color,
                'message': message,
                'gradcam_images': gradcam_frames,
                'frame_predictions': predictions[:5]  # First 5 predictions
            }
        
    except Exception as e:
        print(f"Video processing error: {e}")
    
    return None

def log_detection(source, flood_prob, nonflood_prob, severity, location_data, weather_data, 
                  filename=None, processed_filename=None, gradcam_filename=None, alert_sent=False):
    """Log detection to database"""
    conn = sqlite3.connect('flood_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO detections 
    (source, flood_prob, nonflood_prob, severity, location_name, latitude, longitude, 
     weather_data, filename, processed_filename, gradcam_filename, alert_sent, location_source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (source, flood_prob, nonflood_prob, severity, 
          location_data.get('location_name', 'Unknown'),
          location_data.get('latitude'),
          location_data.get('longitude'),
          json.dumps(weather_data) if weather_data else None,
          filename, processed_filename, gradcam_filename, alert_sent,
          location_data.get('source', 'unknown')))
    
    detection_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return detection_id

def create_location_map(lat, lon, location_name, flood_prob):
    """Create an interactive map with location marker"""
    try:
        # Determine marker color based on flood probability
        if flood_prob >= 70:
            marker_color = 'red'
            icon_type = 'exclamation-triangle'
        elif flood_prob >= 50:
            marker_color = 'orange'
            icon_type = 'exclamation-circle'
        elif flood_prob >= 30:
            marker_color = 'blue'
            icon_type = 'info-sign'
        else:
            marker_color = 'green'
            icon_type = 'ok-sign'
        
        # Create map centered at location
        m = folium.Map(location=[lat, lon], zoom_start=12, tiles='OpenStreetMap')
        
        # Add marker with flood information
        popup_text = f"""
        <div style="font-family: Arial, sans-serif; width: 250px;">
            <h4 style="color: {marker_color}; margin-bottom: 10px;">{location_name}</h4>
            <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
            <p><strong>Flood Probability:</strong> <span style="color: {marker_color}; font-weight: bold;">{flood_prob}%</span></p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr style="margin: 10px 0;">
            <p style="font-size: 12px; color: #666;">FloodSense AI Detection System</p>
        </div>
        """
        
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip="Click for flood details",
            icon=folium.Icon(color=marker_color, icon=icon_type, prefix='fa')
        ).add_to(m)
        
        # Add circle to show area of interest
        folium.Circle(
            location=[lat, lon],
            radius=500,  # 500 meters
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.2,
            popup=f"500m radius around detection point"
        ).add_to(m)
        
        # Save map
        map_filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        map_path = os.path.join(app.config['PROCESSED_FOLDER'], map_filename)
        m.save(map_path)
        
        return f"/static/processed/{map_filename}"
    except Exception as e:
        print(f"Map creation error: {e}")
        return None

def process_image(image):
    """Process image and return prediction results"""
    try:
        if model is None:
            # Return dummy data for testing
            flood_prob = np.random.uniform(0, 100)
            severity, severity_color, message = get_severity_level(flood_prob)
            
            return {
                "prediction": "Flood" if flood_prob > 50 else "NonFlood",
                "flood_prob": round(flood_prob, 2),
                "nonflood_prob": round(100 - flood_prob, 2),
                "severity": severity,
                "severity_color": severity_color,
                "message": message,
                "is_emergency": flood_prob > 70,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Convert to tensor and predict
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            flood_prob = probs[0][0].item() * 100  # Class 0 is Flood
            nonflood_prob = probs[0][1].item() * 100  # Class 1 is NonFlood
        
        # Get severity level
        severity, severity_color, message = get_severity_level(flood_prob)
        
        # Generate Grad-CAM - FIXED: Use correct prediction index
        # Class 0 = Flood, Class 1 = NonFlood
        prediction_index = 0 if flood_prob > 50 else 1
        gradcam_img = generate_gradcam_visualization(image_tensor, image, prediction_index)
        
        # Save Grad-CAM image
        gradcam_filename = None
        if gradcam_img:
            gradcam_filename = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            gradcam_path = os.path.join(app.config['PROCESSED_FOLDER'], gradcam_filename)
            gradcam_img.save(gradcam_path)
        
        return {
            "prediction": "Flood" if flood_prob > 50 else "NonFlood",
            "flood_prob": round(flood_prob, 2),
            "nonflood_prob": round(nonflood_prob, 2),
            "severity": severity,
            "severity_color": severity_color,
            "message": message,
            "is_emergency": flood_prob > 70,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gradcam_filename": gradcam_filename
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return {
            "prediction": "Error",
            "flood_prob": 0,
            "nonflood_prob": 0,
            "severity": "ERROR",
            "severity_color": "gray",
            "message": f"Processing error: {str(e)}",
            "is_emergency": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gradcam_filename": None
        }
def generate_report(detection_id):
    """Generate PDF report for a detection"""
    try:
        # Get detection data
        conn = sqlite3.connect('flood_detection.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        detection = dict(row)
        
        # Parse weather data
        if detection['weather_data']:
            detection['weather_data'] = json.loads(detection['weather_data'])
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = styles['Title']
        title_style.alignment = 1
        story.append(Paragraph("FloodSense AI - Flood Detection Report", title_style))
        story.append(Spacer(1, 12))
        
        # Detection Information
        story.append(Paragraph(f"<b>Detection ID:</b> {detection['id']}", styles['Normal']))
        story.append(Paragraph(f"<b>Timestamp:</b> {detection['timestamp']}", styles['Normal']))
        story.append(Paragraph(f"<b>Source:</b> {detection['source']}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Flood Analysis
        story.append(Paragraph("<b>Flood Analysis:</b>", styles['Heading2']))
        flood_data = [
            ['Parameter', 'Value', 'Status'],
            ['Flood Probability', f"{detection['flood_prob']}%", detection['severity']],
            ['Non-Flood Probability', f"{detection['nonflood_prob']}%", ''],
            ['Severity Level', detection['severity'], ''],
            ['Alert Status', 'Yes' if detection['alert_sent'] else 'No', '']
        ]
        
        t = Table(flood_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        
        # Location Information
        story.append(Paragraph("<b>Location Information:</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Location:</b> {detection['location_name']}", styles['Normal']))
        story.append(Paragraph(f"<b>Coordinates:</b> {detection['latitude']}, {detection['longitude']}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Weather Information
        if detection.get('weather_data'):
            weather = detection['weather_data']
            story.append(Paragraph("<b>Weather Conditions:</b>", styles['Heading2']))
            weather_data = [
                ['Parameter', 'Value'],
                ['Temperature', f"{weather.get('temperature', 'N/A')}°C"],
                ['Feels Like', f"{weather.get('feels_like', 'N/A')}°C"],
                ['Humidity', f"{weather.get('humidity', 'N/A')}%"],
                ['Pressure', f"{weather.get('pressure', 'N/A')} hPa"],
                ['Wind Speed', f"{weather.get('wind_speed', 'N/A')} m/s"],
                ['Conditions', weather.get('weather_description', 'N/A')],
                ['Visibility', f"{weather.get('visibility', 'N/A')} km"],
                ['Rain (1h)', f"{weather.get('rain_1h', 0)} mm"]
            ]
            
            t2 = Table(weather_data, colWidths=[200, 200])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(t2)
        
        # Recommendations
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Recommendations:</b>", styles['Heading2']))
        
        if detection['severity'] in ['EXTREME', 'HIGH']:
            recommendations = [
                "1. IMMEDIATE ACTION REQUIRED: Move to higher ground",
                "2. Avoid walking or driving through flood waters",
                "3. Stay tuned to local weather alerts and warnings",
                "4. Prepare emergency kit with essentials",
                "5. Evacuate if advised by authorities"
            ]
        elif detection['severity'] == 'MODERATE':
            recommendations = [
                "1. Monitor weather conditions closely",
                "2. Prepare emergency supplies",
                "3. Stay informed about local flood warnings",
                "4. Avoid low-lying areas",
                "5. Have an evacuation plan ready"
            ]
        else:
            recommendations = [
                "1. Continue normal activities",
                "2. Stay aware of weather changes",
                "3. Keep emergency contacts handy",
                "4. Review flood safety procedures"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("FloodSense AI - Advanced Flood Detection System", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Save PDF
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Save to file
        report_filename = f"report_detection_{detection_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        with open(report_path, 'wb') as f:
            f.write(pdf_data)
        
        return report_path
        
    except Exception as e:
        print(f"Report generation error: {e}")
        return None

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Main page"""
    # Get user location
    ip_address = request.remote_addr
    location_data = location_service.get_location_from_ip(ip_address)
    location_service.save_location_to_db(location_data, ip_address)
    
    # Get weather data
    weather_data = weather_service.get_current_weather(
        location_data['latitude'], 
        location_data['longitude'],
        location_data['location_name']
    )
    
    # Get risk assessment
    risk_assessment = weather_service.get_flood_risk_assessment(weather_data, 0)
    
    return render_template('index.html', 
                         location=location_data,
                         weather=weather_data,
                         risk_assessment=risk_assessment)

@app.route('/detection/<int:detection_id>')
def view_detection(detection_id):
    """View detection details page"""
    try:
        conn = sqlite3.connect('flood_detection.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            # Return a proper error page with more info
            return render_template('404.html', 
                                 message=f"Detection #{detection_id} not found",
                                 detection_id=detection_id), 404
        
        detection = dict(row)
        
        # Parse weather data
        if detection['weather_data']:
            try:
                detection['weather_data'] = json.loads(detection['weather_data'])
            except:
                detection['weather_data'] = None
        
        # Add URLs
        if detection['filename']:
            detection['original_url'] = f"/static/uploads/{detection['filename']}"
        if detection['processed_filename']:
            detection['preview_url'] = f"/static/processed/{detection['processed_filename']}"
        if detection['gradcam_filename']:
            detection['gradcam_url'] = f"/static/processed/{detection['gradcam_filename']}"
        
        # Get risk assessment
        risk_assessment = None
        if detection.get('weather_data'):
            try:
                risk_assessment = weather_service.get_flood_risk_assessment(
                    detection['weather_data'], 
                    detection['flood_prob']
                )
            except:
                pass
        
        # Generate report URL if needed
        report_path = None
        try:
            report_path = generate_report(detection_id)
        except:
            pass
            
        if report_path:
            report_filename = os.path.basename(report_path)
            detection['report_url'] = f"/static/reports/{report_filename}"
        
        # Create location map
        map_url = None
        try:
            map_url = create_location_map(
                detection['latitude'],
                detection['longitude'],
                detection['location_name'],
                detection['flood_prob']
            )
        except:
            pass
        
        return render_template('view_detection.html',
                             detection=detection,
                             risk_assessment=risk_assessment,
                             map_url=map_url,
                             datetime=datetime)
    
    except Exception as e:
        print(f"Error viewing detection {detection_id}: {e}")
        return render_template('404.html', 
                             message=f"Error loading detection: {str(e)}",
                             detection_id=detection_id), 500

@app.route('/api/detect_location', methods=['GET', 'POST'])
def api_detect_location():
    """Detect user location - supports browser geolocation"""
    try:
        ip_address = request.remote_addr
        
        # Check for browser geolocation data
        browser_location = None
        if request.method == 'POST':
            data = request.json or {}
            if 'latitude' in data and 'longitude' in data:
                browser_location = {
                    'latitude': data['latitude'],
                    'longitude': data['longitude'],
                    'accuracy': data.get('accuracy')
                }
        
        # Get location data
        if browser_location:
            location_data = location_service.get_location_from_coords(
                browser_location['latitude'], 
                browser_location['longitude']
            )
        else:
            location_data = location_service.get_location_from_ip(ip_address)
        
        # Get weather data
        weather_data = weather_service.get_current_weather(
            location_data['latitude'], 
            location_data['longitude'],
            location_data['location_name']
        )
        
        # Get weather alerts
        alerts = location_service.get_weather_alerts(
            location_data['latitude'], 
            location_data['longitude']
        )
        
        # Get risk assessment
        risk_assessment = weather_service.get_flood_risk_assessment(weather_data, 0)
        
        # Save to database
        location_service.save_location_to_db(location_data, ip_address)
        
        response_data = {
            'location': location_data,
            'weather': weather_data,
            'alerts': alerts,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'success': True
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/get_weather', methods=['POST'])
def api_get_weather():
    """Get weather data for coordinates or city name"""
    try:
        data = request.json
        lat = data.get('latitude')
        lon = data.get('longitude')
        city_name = data.get('city_name')
        
        location_data = None
        
        if lat and lon:
            # Get location from coordinates
            location_data = location_service.get_location_from_coords(lat, lon)
        elif city_name:
            # Get location from city name
            location_data = location_service.get_location_from_city(city_name)
            if not location_data:
                return jsonify({"error": "City not found", "success": False}), 404
        else:
            return jsonify({"error": "Coordinates or city name required", "success": False}), 400
        
        if location_data:
            weather_data = weather_service.get_current_weather(
                location_data['latitude'], 
                location_data['longitude'],
                location_data['location_name']
            )
            
            # Get risk assessment with dummy flood probability
            risk_assessment = weather_service.get_flood_risk_assessment(weather_data, 0)
            
            response = {
                'location': location_data,
                'weather': weather_data,
                'risk_assessment': risk_assessment,
                'success': True
            }
            
            return jsonify(response)
        
        return jsonify({"error": "Location not found", "success": False}), 404
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Enhanced upload endpoint with all features"""
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided", "success": False}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected", "success": False}), 400
        
        # Determine file type
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Save original file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get location data from form
        location_name = request.form.get('location_name', '').strip()
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        location_data = None
        
        # Case 1: User provided location name and coordinates
        if latitude and longitude:
            lat, lon = float(latitude), float(longitude)
            if not location_name:
                location_name = location_service.get_location_name(lat, lon)
            location_data = location_service.get_location_from_coords(lat, lon)
            location_data['location_name'] = location_name
        
        # Case 2: User provided only location name
        elif location_name:
            # Try to geocode the city name
            loc_from_city = location_service.get_location_from_city(location_name)
            if loc_from_city:
                location_data = loc_from_city
        
        # Case 3: Auto-detect location from IP or EXIF
        if not location_data:
            ip_address = request.remote_addr
            location_data = location_service.get_location_from_ip(ip_address)
            
            # Try to extract from image EXIF
            if file_ext in ['jpg', 'jpeg', 'png']:
                exif_lat, exif_lon = extract_exif_location(filepath)
                if exif_lat and exif_lon:
                    location_data = location_service.get_location_from_coords(exif_lat, exif_lon)
        
        # Get weather data
        weather_data = weather_service.get_current_weather(
            location_data['latitude'], 
            location_data['longitude'],
            location_data['location_name']
        )
        
        # Process based on file type
        result = {}
        gradcam_filename = None
        processed_filename = None
        
        if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            # Image processing
            image = Image.open(filepath).convert("RGB")
            
            # Generate preview
            preview_size = (400, 300)
            preview_img = image.copy()
            preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)
            preview_filename = f"preview_{unique_filename}"
            preview_path = os.path.join(app.config['PROCESSED_FOLDER'], preview_filename)
            preview_img.save(preview_path)
            processed_filename = preview_filename
            
            # Process image for prediction
            result = process_image(image)
            gradcam_filename = result.get('gradcam_filename')
            
            # Add preview URL
            result['preview_url'] = f"/static/processed/{preview_filename}"
            
            # Add Grad-CAM URL if available
            if gradcam_filename:
                result['gradcam_url'] = f"/static/processed/{gradcam_filename}"
            
            result['type'] = 'image'
            
        elif file_ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            # Video processing
            video_result = process_video_file(filepath, location_data)
            if video_result:
                result = video_result
                # Use first Grad-CAM image as preview
                if result.get('gradcam_images'):
                    result['preview_url'] = result['gradcam_images'][0]
            else:
                return jsonify({"error": "Failed to process video", "success": False}), 500
        
        else:
            return jsonify({"error": "Unsupported file format", "success": False}), 400
        
        # Get flood risk assessment
        risk_assessment = weather_service.get_flood_risk_assessment(
            weather_data, 
            result.get('flood_prob', 0)
        )
        
        # Create location map
        map_url = create_location_map(
            location_data['latitude'], 
            location_data['longitude'],
            location_data['location_name'],
            result.get('flood_prob', 0)
        )
        
        # Log detection to database
        detection_id = log_detection(
            source="upload",
            flood_prob=result.get('flood_prob', 0),
            nonflood_prob=result.get('nonflood_prob', 0),
            severity=result.get('severity', 'UNKNOWN'),
            location_data=location_data,
            weather_data=weather_data,
            filename=unique_filename,
            processed_filename=processed_filename,
            gradcam_filename=gradcam_filename,
            alert_sent=result.get('is_emergency', False)
        )
        
        # Generate report
        report_path = generate_report(detection_id)
        report_url = None
        if report_path:
            report_filename = os.path.basename(report_path)
            report_url = f"/static/reports/{report_filename}"
        
        # Prepare response
        response_data = {
            "success": True,
            "detection_id": detection_id,
            "original_filename": filename,
            "uploaded_file_url": f"/static/uploads/{unique_filename}",
            "preview_url": result.get('preview_url'),
            "gradcam_url": result.get('gradcam_url'),
            "gradcam_images": result.get('gradcam_images', []),
            "prediction": result.get('prediction'),
            "flood_prob": result.get('flood_prob'),
            "nonflood_prob": result.get('nonflood_prob'),
            "severity": result.get('severity'),
            "severity_color": result.get('severity_color'),
            "message": result.get('message'),
            "is_emergency": result.get('is_emergency', False),
            "timestamp": result.get('timestamp'),
            "location": location_data,
            "weather": weather_data,
            "risk_assessment": risk_assessment,
            "map_url": map_url,
            "report_url": report_url,
            "file_type": result.get('type', 'unknown'),
            "video_info": result if result.get('type') == 'video' else None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/detection/<int:detection_id>', methods=['GET'])
def api_get_detection(detection_id):
    """Get detailed information for a specific detection"""
    conn = sqlite3.connect('flood_detection.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        detection = dict(row)
        
        # Parse weather data
        if detection['weather_data']:
            detection['weather_data'] = json.loads(detection['weather_data'])
        
        # Add URLs
        if detection['filename']:
            detection['original_url'] = f"/static/uploads/{detection['filename']}"
        if detection['processed_filename']:
            detection['preview_url'] = f"/static/processed/{detection['processed_filename']}"
        if detection['gradcam_filename']:
            detection['gradcam_url'] = f"/static/processed/{detection['gradcam_filename']}"
        
        # Generate report URL
        report_path = generate_report(detection_id)
        if report_path:
            report_filename = os.path.basename(report_path)
            detection['report_url'] = f"/static/reports/{report_filename}"
        
        return jsonify(detection)
    
    return jsonify({"error": "Detection not found", "detection_id": detection_id}), 404

@app.route('/api/download_report/<int:detection_id>', methods=['GET'])
def api_download_report(detection_id):
    """Download PDF report for a detection"""
    try:
        # Generate report
        report_path = generate_report(detection_id)
        
        if report_path and os.path.exists(report_path):
            return send_file(report_path, as_attachment=True)
        else:
            return jsonify({"error": "Report not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search_location', methods=['GET'])
def api_search_location():
    """Search for location by city name"""
    try:
        city_name = request.args.get('city', '')
        if not city_name:
            return jsonify({"error": "City name required", "success": False}), 400
        
        location_data = location_service.get_location_from_city(city_name)
        if location_data:
            # Get weather for this location
            weather_data = weather_service.get_current_weather(
                location_data['latitude'],
                location_data['longitude'],
                location_data['location_name']
            )
            
            risk_assessment = weather_service.get_flood_risk_assessment(weather_data, 0)
            
            return jsonify({
                'location': location_data,
                'weather': weather_data,
                'risk_assessment': risk_assessment,
                'success': True
            })
        else:
            return jsonify({"error": "Location not found", "success": False}), 404
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    if 'uploads' in filename:
        return send_from_directory('static/uploads', filename.split('/')[-1])
    elif 'processed' in filename:
        return send_from_directory('static/processed', filename.split('/')[-1])
    elif 'reports' in filename:
        return send_from_directory('static/reports', filename.split('/')[-1])
    return send_from_directory('static', filename)

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', 
                         message="Page not found",
                         detection_id=None), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('404.html', 
                         message="Internal server error",
                         detection_id=None), 500

# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 70)
    print("🌊 FloodSense AI - Advanced Flood Detection System")
    print("=" * 70)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Processed folder: {app.config['PROCESSED_FOLDER']}")
    print(f"📁 Reports folder: {app.config['REPORTS_FOLDER']}")
    print(f"🤖 Device: {DEVICE}")
    print(f"📍 Location Features: Browser Geolocation, Manual Input, EXIF Detection")
    print(f"🌤 Weather API: {'CONFIGURED' if OPENWEATHER_API_KEY != 'YOUR_OPENWEATHER_API_KEY_HERE' else 'NOT CONFIGURED - Using mock data'}")
    print(f"🎯 Grad-CAM: Improved ViT Attention-based")
    print(f"📊 PDF Reports: Enabled")
    print("=" * 70)
    print("🚀 Access the system at: http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)