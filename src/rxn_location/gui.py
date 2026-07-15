import os
import sys
import subprocess

def run():
    """Entry point for the rxn-location-gui command."""
    # Find the app.py file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    # Run streamlit with the app
    sys.exit(subprocess.call(["streamlit", "run", app_path]))
