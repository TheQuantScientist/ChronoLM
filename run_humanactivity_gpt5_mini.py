"""Repository-root entry point for the OpenAI GPT-5 mini HumanActivity experiment."""

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronolm.env import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

os.environ.setdefault("CHRONOLM_DATASET", "HumanActivity")
os.environ.setdefault("CHRONOLM_API_PROVIDER", "openai")
os.environ.setdefault("CHRONOLM_API_URL", "https://api.openai.com/v1/chat/completions")
os.environ.setdefault("CHRONOLM_MODEL_ID", "gpt-5-mini")
os.environ.setdefault("CHRONOLM_RUN_NAME", "gpt5_mini_humanactivity_anchor_blend_mse")
os.environ.setdefault("CHRONOLM_PROMPT_STYLE", "gpt_activity_anchor_blend_mse")
os.environ.setdefault("CHRONOLM_ANCHOR_BLEND_WEIGHT", "0.80")
os.environ.setdefault("CHRONOLM_REASONING_EFFORT", "minimal")
os.environ.setdefault("CHRONOLM_VERBOSITY", "low")
os.environ.setdefault("CHRONOLM_INCLUDE_TEMPERATURE", "false")
os.environ.setdefault("CHRONOLM_MAX_TOKENS", "64")

if "CHRONOLM_API_KEY" not in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY before running run_humanactivity_gpt5_mini.py. "
            "Do not hard-code API keys in the repository."
        )
    os.environ["CHRONOLM_API_KEY"] = openai_api_key

from chronolm.experiments.apn_llm import main  # noqa: E402


if __name__ == "__main__":
    main()
