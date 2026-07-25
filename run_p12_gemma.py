"""Repository-root entry point for the Gemma P12 experiment."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronolm.experiments.p12_gemma import main  # noqa: E402


if __name__ == "__main__":
    main()
