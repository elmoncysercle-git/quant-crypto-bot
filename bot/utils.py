import os, json, logging, yaml, pathlib

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_state(path: str):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps({"last_plan": None, "equity_history": []}, indent=2))

def load_state(path: str):
    ensure_state(path)
    return json.loads(pathlib.Path(path).read_text())

def save_state(path: str, state):
    pathlib.Path(path).write_text(json.dumps(state, indent=2))

def setup_logging(level="INFO"):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=getattr(logging, level.upper(), "INFO"))
    return logging.getLogger("bot")

def env(name, default=""):
    return os.environ.get(name, default)

