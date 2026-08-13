"""Quick test of OmniShotCut (System F) integration."""

from cuts.config import CutsConfig, OmniShotCutConfig
from cuts.pipeline import run_system
import sys

# Adjust these paths if your OmniShotCut directory is elsewhere
OMNISHOTCUT_REPO = r"C:\Users\sayya\Desktop\cuts\OmniShotCut"
OMNISHOTCUT_CKPT = r"C:\Users\sayya\Desktop\cuts\checkpoints\OmniShotCut_ckpt.pth"

if len(sys.argv) < 2:
    print("Usage: python test_omnishotcut.py <video_path> [system]")
    print("  system: F (OmniShotCut) or C (TransNetV2) for comparison")
    sys.exit(1)

video_path = sys.argv[1]
system = sys.argv[2] if len(sys.argv) > 2 else "F"

# Configure
cfg = CutsConfig()
cfg.omnishotcut = OmniShotCutConfig(
    repo_path=OMNISHOTCUT_REPO,
    checkpoint_path=OMNISHOTCUT_CKPT,
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
