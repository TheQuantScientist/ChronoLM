"""Repository-root entry point for the OpenAI GPT-5 mini P12 experiment."""

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


load_dotenv(REPO_ROOT / ".env")

os.environ.setdefault("CHRONOLM_API_PROVIDER", "openai")
os.environ.setdefault("CHRONOLM_API_URL", "https://api.openai.com/v1/chat/completions")
os.environ.setdefault("CHRONOLM_MODEL_ID", "gpt-5-mini")
os.environ.setdefault("CHRONOLM_RUN_NAME", "gpt5_mini")
os.environ.setdefault("CHRONOLM_PROMPT_STYLE", "gpt_clinical_mse")
os.environ.setdefault("CHRONOLM_REASONING_EFFORT", "minimal")
os.environ.setdefault("CHRONOLM_VERBOSITY", "low")
os.environ.setdefault("CHRONOLM_INCLUDE_TEMPERATURE", "false")
os.environ.setdefault("CHRONOLM_MAX_TOKENS", "64")

if "CHRONOLM_API_KEY" not in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY before running run_p12_gpt5_mini.py. "
            "Do not hard-code API keys in the repository."
        )
    os.environ["CHRONOLM_API_KEY"] = openai_api_key

from chronolm.experiments.p12_llm import main  # noqa: E402


if __name__ == "__main__":
    main()
