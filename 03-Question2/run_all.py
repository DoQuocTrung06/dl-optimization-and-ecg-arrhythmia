import os
import subprocess
import sys

if "--fast" in sys.argv:
    os.environ["FAST"] = "1"

for s in ["01_eda.py", "02_ml_baselines.py", "03_deep_learning.py", "04_improvements.py", "05_analysis.py"]:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, s], check=True)
