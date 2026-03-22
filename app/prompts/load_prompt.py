from pathlib import Path

BASE = Path(__file__).parent

def load_prompt(name: str) -> str:
    return (BASE / name ).read_text(encoding="utf-8").strip()