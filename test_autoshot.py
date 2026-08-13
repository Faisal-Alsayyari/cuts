"""Quick test of AutoShot (System E) integration."""

from cuts.config import CutsConfig, AutoShotConfig
from cuts.pipeline import run_system
import sys

# Adjust these paths if your AutoShot directory is elsewhere
AUTOSHOT_REPO = r"C:\Users\sayya\Desktop\cuts\AutoShot"
AUTOSHOT_CKPT = r"C:\Users\sayya\Desktop\cuts\AutoShot\supernet_best_f1.pth"

if len(sys.argv) < 2:
    print("Usage: python test_autoshot.py <video_path> [system]")
    print("  system: E (AutoShot) or C (TransNetV2) for comparison")
    sys.exit(1)

video_path = sys.argv[1]
system = sys.argv[2] if len(sys.argv) > 2 else "E"

# Configure
cfg = CutsConfig()
cfg.autoshot = AutoShotConfig(
    repo_path=AUTOSHOT_REPO,
    checkpoint_path=AUTOSHOT_CKPT,
)

print(f"Running System {system} on {video_path}...")
result = run_system(system, video_path, cfg)

print(f"\n=== System {system} Results ===")
print(f"Boundaries found: {len(result.boundaries)}")
print(f"Shots: {len(result.shots)}")
print(f"Runtime: {result.runtime_sec:.2f}s")
print(f"\nFirst 5 boundaries:")
for i, b in enumerate(result.boundaries[:5]):
    print(f"  {i+1}. Frame {b.frame_idx} (time {b.time_sec:.2f}s) - "
          f"type: {b.type}, gradual: {b.is_gradual}")
