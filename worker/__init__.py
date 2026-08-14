"""kakapo AI 워커."""

from pathlib import Path

from dotenv import load_dotenv

# 프로세스 어디서 import 되든 .env 를 한 번만 읽는다.
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
