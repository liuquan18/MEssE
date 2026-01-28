#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEssE Training Monitor - Flask Backend
Real-time monitoring dashboard for Mini-batch GNN training
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os
import glob
import re
from datetime import datetime
import numpy as np

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
BASE_DIR = "/work/mh1498/m301257/work/MEssE"
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiment")
SCRATCH_DIR = "/scratch/m/m301257/icon_exercise_comin"

def get_latest_job():
    """Get the latest SLURM job output file"""
    slurm_files = glob.glob(os.path.join(EXPERIMENT_DIR, "slurm.*.out"))
    if not slurm_files:
        return None
    # Sort by modification time, get latest
    latest = max(slurm_files, key=os.path.getmtime)
    job_id = re.search(r'slurm\.(\d+)\.out', latest).group(1)
    return job_id, latest

def parse_slurm_output(filepath):
    """Parse SLURM output to extract training information"""
    info = {
        'job_id': '',
        'status': 'Unknown',
        'nodes': 0,
        'num_nodes': 0,
        'batch_size': 0,
        'num_batches': 0,
        'model_type': 'Unknown',
        'start_time': '',
        'elapsed_time': '',
        'completed_timesteps': 0,
        'current_batch': 0,
        'total_batches': 0
    }
    
    if not os.path.exists(filepath):
        return info
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            # Extract model type
            if 'Mini-batch GNN' in line:
                info['model_type'] = 'Mini-batch GNN'
            elif 'Initializing GNN' in line:
                info['model_type'] = 'GNN'
            elif 'Initializing MLP' in line:
                info['model_type'] = 'MLP'
            
            # Extract configuration
            if 'Training on' in line and 'nodes' in line:
                match = re.search(r'(\d+)\s+nodes', line)
                if match:
                    info['num_nodes'] = int(match.group(1))
            
            if 'Batch size:' in line:
                match = re.search(r'(\d+)\s+nodes/batch', line)
                if match:
                    info['batch_size'] = int(match.group(1))
            
            if 'Num batches:' in line:
                match = re.search(r'Num batches:\s+(\d+)', line)
                if match:
                    info['num_batches'] = int(match.group(1))
            
            # Count completed timesteps
            if 'Mini-batch GNN training completed' in line or 'GNN training completed' in line:
                info['completed_timesteps'] += 1
            
            # Current batch
            if 'Batch' in line and '/' in line:
                match = re.search(r'Batch\s+(\d+)/(\d+)', line)
                if match:
                    info['current_batch'] = int(match.group(1))
                    info['total_batches'] = int(match.group(2))
        
        info['status'] = 'Running' if info['completed_timesteps'] > 0 else 'Initializing'
        
    except Exception as e:
        print(f"Error parsing SLURM output: {e}")
    
    return info

def get_loss_data():
    """Get loss data from log files"""
    loss_files = glob.glob(os.path.join(SCRATCH_DIR, "log_*.txt"))
    
    if not loss_files:
        return {'timesteps': [], 'losses': [], 'batch_losses': []}
    
    # Sort by filename (timestamp)
    loss_files.sort()
    
    timesteps = []
    avg_losses = []
    all_batch_losses = []
    
    for i, filepath in enumerate(loss_files[-50:]):  # Last 50 timesteps
        try:
            with open(filepath, 'r') as f:
                losses = [float(line.strip()) for line in f if line.strip()]
            
            if losses:
                timesteps.append(i + 1)
                avg_losses.append(np.mean(losses))
                all_batch_losses.extend(losses)
        except:
            continue
    
    return {
        'timesteps': timesteps,
        'avg_losses': avg_losses,
        'batch_losses': all_batch_losses[-100:] if all_batch_losses else [],  # Last 100 batch losses
        'total_files': len(loss_files)
    }

def get_system_info():
    """Get system and dataset information"""
    return {
        'project': 'MEssE',
        'version': 'V.01',
        'model': 'ICON LAM',
        'plugin': 'ComIn',
        'framework': 'PyTorch 2.8.0',
        'strategy': 'Mini-batch GNN',
        'domain': 'DOM01',
        'grid': 'Icosahedral',
        'variables': ['RHI_MAX', 'QI_MAX'],
        'input_var': 'RHI_MAX (Relative Humidity over Ice)',
        'output_var': 'QI_MAX (Cloud Ice Content)',
        'resolution': 'R3B08',
        'simulation_start': '2021-07-14 00:00:00',
        'simulation_end': '2021-07-15 00:00:00',
        'output_interval': '5 minutes'
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for current training status"""
    job_info = get_latest_job()
    
    if not job_info:
        return jsonify({'error': 'No job found'})
    
    job_id, filepath = job_info
    training_info = parse_slurm_output(filepath)
    training_info['job_id'] = job_id
    
    return jsonify(training_info)

@app.route('/api/losses')
def api_losses():
    """API endpoint for loss data"""
    return jsonify(get_loss_data())

@app.route('/api/system')
def api_system():
    """API endpoint for system information"""
    return jsonify(get_system_info())

@app.route('/api/realtime')
def api_realtime():
    """API endpoint for real-time combined data"""
    job_info = get_latest_job()
    
    data = {
        'status': {},
        'losses': {},
        'system': get_system_info(),
        'timestamp': datetime.now().isoformat()
    }
    
    if job_info:
        job_id, filepath = job_info
        data['status'] = parse_slurm_output(filepath)
        data['status']['job_id'] = job_id
    
    data['losses'] = get_loss_data()
    
    return jsonify(data)

if __name__ == '__main__':
    # Run on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
