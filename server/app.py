"""Read-only dashboard API. It only READS the database. The one thing it can write is a 'collect' command
(start/stop a labelled recording). There is no endpoint that can switch the fan.
Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000"""
import asyncio, json, pathlib, re, time
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect
from common import config
from common.db import DB
from common.schema import LABEL_NAMES, SCHEMA_VERSION

cfg = config.load()
db = DB(config.path(cfg, cfg["db_path"]))
HERE = pathlib.Path(__file__).parent
REPORT_DIR = config.path(cfg, "ml/reports/latest")
STALE_S = 5.0

app = FastAPI(title="Efficio Apex dashboard API")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

def _live():
    r = db.latest_row()
    if not r:
        return {"row": None, "stale": True, "age_s": None}
    row, written = r
    age = time.time() - written
    return {"row": row, "stale": age > STALE_S, "age_s": round(age, 1)}

@app.get("/")
def index():
    return FileResponse(HERE / "static" / "debug.html")     # replace with the real UI later

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(_live())
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, Exception):
        pass

@app.get("/api/live")
def live():
    return _live()

@app.get("/api/history")
def history(minutes: float = 10, limit: int = 3000):
    return db.history(min(max(minutes, 0.1), 120), min(max(limit, 1), 5000))

@app.get("/api/events")
def events(limit: int = 100):
    return db.events(min(max(limit, 1), 500))

@app.get("/api/runs")
def runs(limit: int = 50):
    return db.runs(min(max(limit, 1), 200))

@app.get("/api/ml/report")
def ml_report():
    out = {"available": False}
    for name in ("metrics", "confusion_matrix", "feature_importance", "dataset_summary", "model_info"):
        p = REPORT_DIR / f"{name}.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text()); out["available"] = True
            except Exception:
                out[name] = {"error": "unreadable"}
    return out

@app.get("/api/health")
def health():
    lv = _live()
    return {"schema": SCHEMA_VERSION, "db": db.counts(), "last_row_age_s": lv["age_s"], "stale": lv["stale"],
            "mode": lv["row"]["mode"] if lv["row"] else None, "source": cfg["source"], "server_time": time.time()}

@app.get("/api/config")
def get_config():
    return {k: cfg[k] for k in ("source", "demo_timers", "timers", "thresholds", "clear_ratio", "tick_s")}

class Collect(BaseModel):
    label: str
    note: str = ""

@app.post("/api/collect/start")
def collect_start(b: Collect):
    if b.label not in LABEL_NAMES:
        raise HTTPException(400, f"label must be one of {LABEL_NAMES}")
    db.add_command("collect_start", {"label": b.label, "note": b.note[:80]})
    return {"queued": True}

@app.post("/api/collect/stop")
def collect_stop():
    db.add_command("collect_stop", {})
    return {"queued": True}
