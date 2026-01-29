#!/usr/bin/env python
"""
Test script to verify the monitoring system works correctly
"""

import json
import os
import sys
import getpass


def test_status_file():
    """Test that the status file exists and is valid JSON"""
    user = getpass.getuser()
    status_file = f"/scratch/{user[0]}/{user}/icon_exercise_comin/monitor_status.json"

    print(f"Testing status file: {status_file}")

    if not os.path.exists(status_file):
        print("❌ Status file does not exist")
        return False

    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        print("✅ Status file is valid JSON")
    except Exception as e:
        print(f"❌ Error reading status file: {e}")
        return False

    # Check required fields
    required_sim_fields = [
        "start_time",
        "current_time",
        "elapsed_time",
        "n_domains",
        "total_points",
        "output_count",
    ]
    required_train_fields = [
        "model_type",
        "current_loss",
        "total_batches",
        "batches_per_timestep",
        "learning_rate",
        "avg_loss",
        "min_loss",
        "max_loss",
    ]

    missing = []

    if "simulation" not in data:
        missing.append("simulation")
    else:
        for field in required_sim_fields:
            if field not in data["simulation"]:
                missing.append(f"simulation.{field}")

    if "training" not in data:
        missing.append("training")
    else:
        for field in required_train_fields:
            if field not in data["training"]:
                missing.append(f"training.{field}")

    if missing:
        print(f"❌ Missing fields: {', '.join(missing)}")
        return False

    print("✅ All required fields present")

    # Print summary
    print("\n" + "=" * 60)
    print("MONITORING STATUS SUMMARY")
    print("=" * 60)
    print(f"Model Type: {data['training']['model_type']}")
    print(f"Current Time: {data['simulation']['current_time']}")
    print(f"Elapsed Time: {data['simulation']['elapsed_time']}")
    print(f"Total Points: {data['simulation']['total_points']}")
    print(f"Output Count: {data['simulation']['output_count']}")
    print(f"Current Loss: {data['training']['current_loss']:.6e}")
    print(f"Total Batches: {data['training']['total_batches']}")
    print(f"Batches/Timestep: {data['training']['batches_per_timestep']}")
    print(f"Learning Rate: {data['training']['learning_rate']}")
    print(f"Avg Loss: {data['training']['avg_loss']:.6e}")
    print(f"Min Loss: {data['training']['min_loss']:.6e}")
    print(f"Max Loss: {data['training']['max_loss']:.6e}")
    print("=" * 60)

    return True


def test_log_files():
    """Test that log files exist"""
    user = getpass.getuser()
    log_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

    print(f"\nTesting log directory: {log_dir}")

    if not os.path.exists(log_dir):
        print("❌ Log directory does not exist")
        return False

    import glob as glob_module

    log_files = glob_module.glob(os.path.join(log_dir, "log_*.txt"))

    if not log_files:
        print("❌ No log files found")
        return False

    print(f"✅ Found {len(log_files)} log files")

    # Check latest log file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Latest log: {os.path.basename(latest_log)}")

    return True


def test_checkpoint_files():
    """Test that checkpoint files exist"""
    user = getpass.getuser()
    checkpoint_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

    print(f"\nTesting checkpoint directory: {checkpoint_dir}")

    import glob as glob_module

    checkpoint_files = glob_module.glob(os.path.join(checkpoint_dir, "net_*.pth"))

    if not checkpoint_files:
        print("⚠️  No checkpoint files found (may be OK if simulation just started)")
        return True

    print(f"✅ Found {len(checkpoint_files)} checkpoint files")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING SYSTEM TEST")
    print("=" * 60)

    results = []

    results.append(("Status File", test_status_file()))
    results.append(("Log Files", test_log_files()))
    results.append(("Checkpoint Files", test_checkpoint_files()))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✅ All tests passed! Monitoring system is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
