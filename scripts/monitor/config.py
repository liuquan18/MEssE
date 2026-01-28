#!/usr/bin/env python3
"""
Configuration file for ICON ML Training Loss Monitor
"""

# Server Configuration
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5000        # Default port

# Update intervals (in seconds)
DATA_UPDATE_INTERVAL = 5      # How often to refresh data from disk
DASHBOARD_REFRESH_RATE = 5    # How often dashboard polls API

# Data settings
DEFAULT_HISTORY_LIMIT = 100   # Default number of data points to display
MAX_HISTORY_LIMIT = 1000      # Maximum history limit

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = 'dashboard.log'

# Data directory (auto-detected from username if None)
DATA_DIRECTORY = None  # Set to specific path or None for auto-detection

# TensorBoard settings
TENSORBOARD_ENABLED = True  # Try to load TensorBoard data if available

# Chart settings
CHART_COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444'
}

# Performance settings
MAX_LOG_ENTRIES = 50          # Maximum log entries in activity log
HISTOGRAM_BINS = 20           # Number of bins for distribution chart

# Security settings (for production deployment)
SECRET_KEY = None  # Set a secret key for production
ENABLE_CORS = True # Enable Cross-Origin Resource Sharing
