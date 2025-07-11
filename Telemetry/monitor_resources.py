#!/usr/bin/env python3
"""
Resource monitoring script to check memory and GPU usage during training.
"""

import psutil
import time
import subprocess
import os
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in GB."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'used_gb': memory.used / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent': memory.percent
    }

def get_gpu_info():
    """Get GPU information using system_profiler."""
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting GPU info: {e}"

def get_process_memory():
    """Get memory usage of Python processes."""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                memory_mb = proc.info['memory_info'].rss / (1024**2)
                python_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': memory_mb
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

def monitor_resources():
    """Monitor system resources in real-time."""
    print("=== Resource Monitoring for Training ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Get memory usage
            mem = get_memory_usage()
            
            # Get Python process memory
            python_procs = get_process_memory()
            
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"=== Resource Monitor - {datetime.now().strftime('%H:%M:%S')} ===")
            print()
            
            # System memory
            print("📊 SYSTEM MEMORY:")
            print(f"   Total: {mem['total_gb']:.1f} GB")
            print(f"   Used: {mem['used_gb']:.1f} GB ({mem['percent']:.1f}%)")
            print(f"   Available: {mem['available_gb']:.1f} GB")
            print(f"   Free: {mem['available_gb']:.1f} GB")
            print()
            
            # Python processes
            print("🐍 PYTHON PROCESSES:")
            total_python_memory = 0
            for proc in python_procs:
                print(f"   PID {proc['pid']}: {proc['name']} - {proc['memory_mb']:.1f} MB")
                total_python_memory += proc['memory_mb']
            print(f"   Total Python memory: {total_python_memory:.1f} MB ({total_python_memory/1024:.2f} GB)")
            print()
            
            # Recommendations
            available_gb = mem['available_gb']
            print("💡 BATCH SIZE RECOMMENDATIONS:")
            
            if available_gb > 8:
                print(f"   ✅ Can safely increase to batch size 1024+ (8+ GB available)")
                print(f"   🚀 Try: batch_size=1024, lr=6e-5")
            elif available_gb > 4:
                print(f"   ✅ Can increase to batch size 512 (4+ GB available)")
                print(f"   🚀 Try: batch_size=512, lr=3e-5")
            elif available_gb > 2:
                print(f"   ⚠️  Can try batch size 384 (2+ GB available)")
                print(f"   🚀 Try: batch_size=384, lr=2.25e-5")
            else:
                print(f"   ❌ Limited memory available ({available_gb:.1f} GB)")
                print(f"   💡 Keep current batch size or reduce")
            
            print()
            print("Press Ctrl+C to stop monitoring...")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Final recommendations:")
        mem = get_memory_usage()
        if mem['available_gb'] > 8:
            print("✅ You have plenty of memory! Try batch_size=1024")
        elif mem['available_gb'] > 4:
            print("✅ Good memory available! Try batch_size=512")
        else:
            print("⚠️  Limited memory available. Keep current batch size.")

if __name__ == "__main__":
    monitor_resources() 