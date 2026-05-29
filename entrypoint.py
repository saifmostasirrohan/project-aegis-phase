import os
import subprocess
import sys
import time

# Pre-populate fallback mock API keys for validation during import/startup.
# These will be overridden by any real secrets configured in Hugging Face Space settings.
if not os.environ.get("GROQ_API_KEY"):
    print("GROQ_API_KEY not found in environment. Injecting mock key for startup validation.")
    os.environ["GROQ_API_KEY"] = "gsk_mock_key_for_startup_validation_only_12345"
if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not found in environment. Injecting mock key for startup validation.")
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-startup-validation-only-12345"

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

# Mock comment to trigger rebuild on HF Space
