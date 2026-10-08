"""Reproduce every result and figure: python run_all.py (about 20 minutes on a laptop)."""
import subprocess
import sys

for script in ["run_message.py", "run_journey.py", "run_external.py", "injection_model.py",
               "run_sensitivity.py", "run_ai_variants.py", "make_figures.py"]:
    print(f"== {script}", flush=True)
    subprocess.run([sys.executable, script], check=True)
