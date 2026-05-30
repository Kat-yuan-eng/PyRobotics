import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from Localization.ekf_localizer import run_ekf_demo
from Localization.pf_localizer import run_pf_demo
from Localization.fusion_localizer import run_fusion_demo

rmse_ekf, dr_rmse = run_ekf_demo()
rmse_pf = run_pf_demo()
_, _, rmse_fused = run_fusion_demo()

improvement = (1 - rmse_fused / dr_rmse) * 100
target = dr_rmse * 0.8

print()
print("=== RMSE Summary ===")
print(f"Dead Reckoning: {dr_rmse:.3f}m")
print(f"EKF:            {rmse_ekf:.3f}m  ({(1-rmse_ekf/dr_rmse)*100:+.1f}%)")
print(f"PF:             {rmse_pf:.3f}m  ({(1-rmse_pf/dr_rmse)*100:+.1f}%)")
print(f"Fused:          {rmse_fused:.3f}m  ({improvement:+.1f}%)")
print(f"Target: RMSE < {target:.3f}m (20% reduction)")
achieved = "YES" if rmse_fused < target else "NO"
print(f"Achieved: {achieved} (Fused RMSE={rmse_fused:.3f}m)")
