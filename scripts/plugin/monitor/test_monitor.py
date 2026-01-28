#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for MEssE Training Monitor
Verify all components are working correctly
"""

import os
import sys
import glob
import re

def test_directories():
    """Test if all required directories exist"""
    print("🔍 Testing Directories...")
    
    base_dir = "/work/mh1498/m301257/work/MEssE"
    experiment_dir = os.path.join(base_dir, "experiment")
    scratch_dir = "/scratch/m/m301257/icon_exercise_comin"
    
    checks = [
        ("Base directory", base_dir),
        ("Experiment directory", experiment_dir),
        ("Scratch directory", scratch_dir)
    ]
    
    all_ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    return all_ok

def test_job_files():
    """Test if SLURM job files exist"""
    print("\n🔍 Testing Job Files...")
    
    experiment_dir = "/work/mh1498/m301257/work/MEssE/experiment"
    slurm_files = glob.glob(os.path.join(experiment_dir, "slurm.*.out"))
    
    if slurm_files:
        latest = max(slurm_files, key=os.path.getmtime)
        job_id = re.search(r'slurm\.(\d+)\.out', latest).group(1)
        print(f"  ✅ Found SLURM files: {len(slurm_files)}")
        print(f"  ✅ Latest job: {job_id}")
        print(f"  ✅ File: {latest}")
        return True
    else:
        print("  ❌ No SLURM files found")
        return False

def test_loss_files():
    """Test if loss log files exist"""
    print("\n🔍 Testing Loss Files...")
    
    scratch_dir = "/scratch/m/m301257/icon_exercise_comin"
    loss_files = glob.glob(os.path.join(scratch_dir, "log_*.txt"))
    
    if loss_files:
        print(f"  ✅ Found {len(loss_files)} loss log files")
        # Show a sample
        sample = loss_files[0]
        try:
            with open(sample, 'r') as f:
                losses = [float(line.strip()) for line in f if line.strip()]
            print(f"  ✅ Sample file has {len(losses)} loss values")
            return True
        except Exception as e:
            print(f"  ⚠️  Could not read loss file: {e}")
            return False
    else:
        print("  ❌ No loss files found")
        return False

def test_flask():
    """Test if Flask is installed"""
    print("\n🔍 Testing Flask...")
    
    try:
        import flask
        print(f"  ✅ Flask version: {flask.__version__}")
        return True
    except ImportError:
        print("  ❌ Flask not installed")
        print("     Install with: pip install flask")
        return False

def test_monitor_files():
    """Test if monitor files exist"""
    print("\n🔍 Testing Monitor Files...")
    
    monitor_dir = "/work/mh1498/m301257/work/MEssE/scripts/plugin/monitor"
    
    files = [
        ("Flask app", "app.py"),
        ("Dashboard template", "templates/dashboard.html"),
        ("Start script", "start_monitor.sh"),
        ("README", "README.md")
    ]
    
    all_ok = True
    for name, filename in files:
        path = os.path.join(monitor_dir, filename)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {filename}")
        if not exists:
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("🧪 MEssE Training Monitor - System Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Directories", test_directories()))
    results.append(("Job Files", test_job_files()))
    results.append(("Loss Files", test_loss_files()))
    results.append(("Flask", test_flask()))
    results.append(("Monitor Files", test_monitor_files()))
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Monitor is ready to use.")
        print("\nNext steps:")
        print("  1. Start the monitor: ./start_monitor.sh")
        print("  2. Set up SSH forwarding (see QUICKSTART.sh)")
        print("  3. Open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
