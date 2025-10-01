from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plotting import plot_channel


if __name__ == "__main__":
    plot_channel(Path("data2/(1).csv"), channel_idx=3, kind="B1")
