"""Print observation-count statistics for the APN P12 test split."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronolm.tools.p12_obs_count import main  # noqa: E402


if __name__ == "__main__":
    main()
