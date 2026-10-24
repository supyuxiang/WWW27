'''Entry point for the AVeriTeC/FEVER/SciFact GProofT adaptation.'''

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'www27'
sys.path.insert(0,str(VENDOR))
runpy.run_path(str(VENDOR / 'baselines' / 'run_gprooft_offline.py'), run_name='__main__')
