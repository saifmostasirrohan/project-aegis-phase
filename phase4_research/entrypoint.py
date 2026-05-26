import subprocess
import sys
import time
import os

print("Starting Aegis FastAPI Backend...")
backend_process = subprocess.Popen([
    sys.executable, "-m", "uvicorn", "capstone.api:app",
    "--host", "127.0.0.1",
    "--port", "8000"
])

print("Waiting for backend to initialize...")
time.sleep(3)

print("Starting Aegis Streamlit Frontend...")
frontend_process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "capstone/app.py",
    "--server.port", "7860",
    "--server.address", "0.0.0.0",
    "--browser.gatherUsageStats", "false"
])

try:
    while True:
        backend_exit = backend_process.poll()
        frontend_exit = frontend_process.poll()
        
        if backend_exit is not None:
            print(f"Backend process exited with code {backend_exit}")
            sys.exit(backend_exit)
        if frontend_exit is not None:
            print(f"Frontend process exited with code {frontend_exit}")
            sys.exit(frontend_exit)
            
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating processes...")
    backend_process.terminate()
    frontend_process.terminate()
