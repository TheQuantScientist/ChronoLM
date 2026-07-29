"""Repository-root entry point for the OpenAI GPT-5 mini P12 experiment."""

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronolm.env import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

os.environ["CHRONOLM_DATASET"] = "P12"
os.environ["CHRONOLM_API_PROVIDER"] = "openrouter"
os.environ["CHRONOLM_API_URL"] = "https://openrouter.ai/api/v1/chat/completions"
os.environ["CHRONOLM_MODEL_ID"] = "openai/gpt-5-mini"
os.environ["CHRONOLM_RUN_NAME"] = "gpt5_mini_p12_calibrated_anchor_mse"
os.environ["CHRONOLM_PROMPT_STYLE"] = "gpt_clinical_calibrated_anchor_mse"
os.environ["CHRONOLM_ANCHOR_BLEND_WEIGHT"] = "0.95"
os.environ["CHRONOLM_REASONING_EFFORT"] = "minimal"
os.environ["CHRONOLM_INCLUDE_TEMPERATURE"] = "false"
os.environ["CHRONOLM_MAX_TOKENS"] = "64"

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if openrouter_api_key:
    os.environ["CHRONOLM_API_KEY"] = openrouter_api_key
elif "CHRONOLM_API_KEY" not in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY before running run_p12_gpt5_mini.py. "
            "Do not hard-code API keys in the repository."
        )
    os.environ["CHRONOLM_API_KEY"] = openai_api_key

from chronolm.experiments.apn_llm import main  # noqa: E402


if __name__ == "__main__":
    main()
