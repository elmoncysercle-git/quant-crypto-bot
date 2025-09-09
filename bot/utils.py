import os, json, logging, yaml, time, math, pathlib
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_state(path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps({"last_rebalance": None, "portfolio": {}, "equity_history": []}, indent=2))

def load_state(path: str) -> Dict[str, Any]:
    ensure_state(path)
    return json.loads(pathlib.Path(path).read_text())

def save_state(path: str, state: Dict[str, Any]):
    pathlib.Path(path).write_text(json.dumps(state, indent=2))

def setup_logging(level="INFO"):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=getattr(logging, level.upper(), "INFO")
    )
    return logging.getLogger("bot")

def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def bp(n: float) -> float:
    return 10000*n
