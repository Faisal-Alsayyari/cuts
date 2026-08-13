"""Benchmark all 6 systems (A, B, C, D, E, F) on a single video."""

from cuts.config import CutsConfig, AutoShotConfig, OmniShotCutConfig
from cuts.pipeline import run_system
import sys
import time

# Paths to the downloaded models
AUTOSHOT_REPO = r"C:\Users\sayya\Desktop\cuts\AutoShot"
AUTOSHOT_CKPT = r"C:\Users\sayya\Desktop\cuts\AutoShot\supernet_best_f1.pth"

OMNISHOTCUT_REPO = r"C:\Users\sayya\Desktop\cuts\OmniShotCut"
OMNISHOTCUT_CKPT = r"C:\Users\sayya\Desktop\cuts\checkpoints\OmniShotCut_ckpt.pth"

if len(sys.argv) < 2:
    print("Usage: python benchmark_all.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

# Configure with all models
cfg = CutsConfig()
cfg.autoshot = AutoShotConfig(
    repo_path=AUTOSHOT_REPO,
    checkpoint_path=AUTOSHOT_CKPT,
)
cfg.omnishotcut = OmniShotCutConfig(
    repo_path=OMNISHOTCUT_REPO,
    checkpoint_path=OMNISHOTCUT_CKPT,
)

SYSTEMS = {
    "A": "PySceneDetect ContentDetector",
    "B": "PySceneDetect AdaptiveDetector",
    "C": "TransNetV2",
    "D": "Full Hybrid Pipeline (PyScene + TransNetV2 + Refinement)",
    "E": "AutoShot (NAS-optimised 3D ConvNet + Transformer)",
    "F": "OmniShotCut (Shot-Query Transformer)",
}

print(f"Benchmarking on: {video_path}\n")
print("=" * 80)

results = {}
for system in ["A", "B", "C", "D", "E", "F"]:
    print(f"\nRunning System {system}: {SYSTEMS[system]}...")
    try:
        result = run_system(system, video_path, cfg)
        results[system] = result
        print(f"  ✓ Boundaries: {len(result.boundaries)}, "
              f"Shots: {len(result.shots)}, "
              f"Runtime: {result.runtime_sec:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "=" * 80)
print("Summary Table:")
print("-" * 80)
print(f"{'System':<10} {'Description':<40} {'Boundaries':<12} {'Runtime (s)':<12}")
print("-" * 80)

for system in ["A", "B", "C", "D", "E", "F"]:
    if system in results:
        r = results[system]
        desc = SYSTEMS[system][:37] + "..." if len(SYSTEMS[system]) > 40 else SYSTEMS[system]
        print(f"{system:<10} {desc:<40} {len(r.boundaries):<12} {r.runtime_sec:<12.2f}")
    else:
        print(f"{system:<10} {SYSTEMS[system][:40]:<40} {'FAILED':<12} {'-':<12}")

print("-" * 80)
