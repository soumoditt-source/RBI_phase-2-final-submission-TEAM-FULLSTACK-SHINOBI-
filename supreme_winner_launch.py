import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    banner = r"""
    🔍 MuleHunter.AI | SUPREME WINNER EDITION
    =========================================
    National Fraud Prevention Challenge 2026
    Team: FullStackShinobi
    """
    print(banner)

def check_dependencies():
    print("[1/4] Verifying Shinobi-Cortex Dependencies...")
    required = ["streamlit", "pandas", "polars", "pyarrow", "rustworkx", "plotly", "scikit-learn", "lightgbm", "xgboost"]
    missing = []
    
    for pkg in required:
        try:
            # Map common pip names to import names if different
            import_name = pkg
            if pkg == "scikit-learn": import_name = "sklearn"
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        print(" ✅ All Dependencies Validated.")
    else:
        print(f" ⚠️ Missing: {', '.join(missing)}")
        print(" [!] Initiating Auto-Provisioning (Shinobi-Heal)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print(" ✅ Environment Provisioned Successfully.")
        except Exception as e:
            print(f" ❌ Auto-install failed: {e}")
            print(" Please run: pip install -r requirements.txt manually.")
            sys.exit(1)

def check_forensic_artifacts():
    print("[2/4] Authenticating Forensic Hot-Caches...")
    results_dir = Path("results")
    mandatory = ["account_violations.parquet", "money_chain.json", "shap_importance.csv"]
    
    for m in mandatory:
        p = results_dir / m
        if p.exists():
            size_mb = p.stat().st_size / (1024*1024)
            print(f" ✅ {m} ({size_mb:.2f} MB) - VERIFIED.")
        else:
            print(f" ⚠️ Warning: {m} is missing. Some tabs may be empty.")

def check_model_integrity():
    print("[3/4] Validating ML Model Ensemble Weights...")
    # Checking for any .pkl files in results
    models = list(Path("results").glob("*.pkl"))
    if models:
        print(f" ✅ {len(models)} Model Artifacts Detected.")
    else:
        print(" ⚠️ No serialized models found in /results. Running in rules-only mode.")

def launch_mission_control():
    print("[4/4] Launching MuleHunter.AI Mission Control...")
    print(" 🚀 Portal will open in your default browser shortly.")
    time.sleep(2)
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "app/dashboard.py", 
        "--server.port", "8509", 
        "--theme.base", "dark"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n [!] Shinobi-Cortex Offline. System Shutdown.")

if __name__ == "__main__":
    print_banner()
    check_dependencies()
    check_forensic_artifacts()
    check_model_integrity()
    launch_mission_control()
