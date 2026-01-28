#!/usr/bin/env python3
"""
Real-time dashboard server for monitoring ICON-ML training losses.
This Flask application provides a web interface to visualize loss functions
in real-time during ICON climate model simulations.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import getpass
from pathlib import Path
import json
from datetime import datetime
import threading
import time

# Import our data aggregator
from data_aggregator import LossDataAggregator

app = Flask(__name__)
CORS(app)

# Global data aggregator
data_aggregator = None
update_thread = None
stop_event = threading.Event()


def initialize_aggregator():
    """Initialize the data aggregator with user-specific paths."""
    global data_aggregator
    user = getpass.getuser()
    log_dir = Path(f"/scratch/{user[0]}/{user}/icon_exercise_comin")
    data_aggregator = LossDataAggregator(log_dir)
    print(f"Monitoring directory: {log_dir}")


def background_update():
    """Background thread to periodically update data."""
    while not stop_event.is_set():
        if data_aggregator:
            data_aggregator.update_data()
        time.sleep(5)  # Update every 5 seconds


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get overall monitoring status."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    return jsonify({
        'status': 'running',
        'monitoring_dir': str(data_aggregator.log_dir),
        'last_update': datetime.now().isoformat(),
        'total_timesteps': len(data_aggregator.timesteps),
        'total_batches': data_aggregator.total_batches
    })


@app.route('/api/loss/current')
def get_current_loss():
    """Get the most recent loss values."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    current = data_aggregator.get_current_stats()
    return jsonify(current)


@app.route('/api/loss/history')
def get_loss_history():
    """Get historical loss data."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    limit = request.args.get('limit', type=int, default=100)
    history = data_aggregator.get_loss_history(limit=limit)
    return jsonify(history)


@app.route('/api/loss/statistics')
def get_statistics():
    """Get statistical summary of losses."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    stats = data_aggregator.get_statistics()
    return jsonify(stats)


@app.route('/api/loss/timesteps')
def get_timesteps():
    """Get all available timesteps."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    timesteps = data_aggregator.get_timesteps()
    return jsonify({'timesteps': timesteps})


@app.route('/api/loss/by_timestep/<timestamp>')
def get_loss_by_timestep(timestamp):
    """Get all losses for a specific timestep."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    data = data_aggregator.get_timestep_data(timestamp)
    if data is None:
        return jsonify({'error': 'Timestep not found'}), 404
    
    return jsonify(data)


@app.route('/api/tensorboard/available')
def check_tensorboard():
    """Check if TensorBoard data is available."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    return jsonify({
        'available': data_aggregator.has_tensorboard_data(),
        'runs_dir': str(data_aggregator.log_dir / 'runs')
    })


@app.route('/api/tensorboard/scalars')
def get_tensorboard_scalars():
    """Get scalar data from TensorBoard logs."""
    if not data_aggregator:
        return jsonify({'error': 'Data aggregator not initialized'}), 500
    
    tag = request.args.get('tag', 'Loss/batch')
    scalars = data_aggregator.get_tensorboard_scalars(tag)
    return jsonify(scalars)


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the dashboard server."""
    global update_thread
    
    # Initialize data aggregator
    initialize_aggregator()
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          ICON ML Training Loss Monitor Dashboard              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Dashboard URL: http://{host}:{port}                      ║
    ║  Monitoring:    {str(data_aggregator.log_dir)[:40]:<40}  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ICON ML Loss Monitor Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        start_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        stop_event.set()
        if update_thread:
            update_thread.join(timeout=2)
        print("Dashboard server stopped.")
