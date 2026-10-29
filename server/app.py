"""Read-only dashboard API and WebSocket server.
It only READS the database. The one write endpoint queues a labelled recording command.
There is no API endpoint that can manually switch the fan relay.
Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000"""
import asyncio, json, pathlib, time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect
from common import config
from common.db import DB
from common.logger import get_logger
from common.schema import LABEL_NAMES, SCHEMA_VERSION
from ml.inference.predictor import AirQualityPredictor
from ml.inference.status import get_system_ml_status, get_model_health, get_training_readiness

logger = get_logger("server.app")

cfg = config.load()
db = DB(config.path(cfg, cfg["db_path"]))
HERE = pathlib.Path(__file__).parent
REPORT_DIR = config.path(cfg, "ml/reports/latest")
STALE_S = 5.0

app = FastAPI(title="Efficio Apex Dashboard API")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

_active_predictor: Optional[AirQualityPredictor] = None

def _get_active_predictor() -> AirQualityPredictor:
    global _active_predictor
    if _active_predictor is None:
        mp = cfg.get("ml", {}).get("model_path")
        p = config.path(cfg, str(mp)) if mp else None
        _active_predictor = AirQualityPredictor(p) if (p and p.exists()) else AirQualityPredictor()
    return _active_predictor


def _live():
    r = db.latest_row()
    if not r:
        return {"row": None, "stale": True, "age_s": None}
    row, written = r
    age = time.time() - written
    return {"row": row, "stale": age > STALE_S, "age_s": round(age, 1)}


@app.get("/")
def index():
    """Serve the primary dashboard UI."""
    index_file = HERE / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return FileResponse(HERE / "static" / "debug.html")


@app.get("/debug")
def debug():
    """Serve the minimal debug UI."""
    return FileResponse(HERE / "static" / "debug.html")


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
                out[name] = json.loads(p.read_text(encoding="utf-8"))
                out["available"] = True
            except Exception:
                out[name] = {"error": "unreadable"}
    return out


@app.get("/api/ml/status")
def ml_status():
    """Consolidated ML status, model health, training readiness, and safety telemetry."""
    lv = _live()
    return get_system_ml_status(cfg, latest_row=lv.get("row"))


@app.post("/api/ml/reload")
def ml_reload():
    """Safely validate and reload the configured ML model artifact.
    
    Validates the new artifact before activating it. Preserves the currently active
    model if the replacement artifact is invalid or incompatible.
    """
    global _active_predictor
    mp = cfg.get("ml", {}).get("model_path")
    if not mp:
        raise HTTPException(status_code=400, detail="No model_path configured in config.yaml")

    target_path = config.path(cfg, str(mp))
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found at '{target_path.name}'")

    # Validate before activating
    candidate = AirQualityPredictor(target_path)
    if not candidate.is_loaded:
        raise HTTPException(
            status_code=400,
            detail=f"Model artifact validation failed: {candidate.last_error or 'Unknown loading error'}",
        )

    # Atomic swap
    _active_predictor = candidate
    logger.info(
        "Successfully reloaded model artifact from %s (id: %s, features: %d)",
        target_path.name,
        candidate.model_id,
        candidate.feature_count,
    )

    return {
        "status": "success",
        "reloaded": True,
        "model_id": candidate.model_id,
        "feature_count": candidate.feature_count,
        "contract": "CANONICAL_32" if candidate.feature_count == 32 else ("LEGACY_26" if candidate.feature_count == 26 else "CUSTOM"),
        "classes": list(candidate.class_names),
        "source": candidate.model_source,
    }


@app.get("/api/health")
def health():
    lv = _live()
    return {
        "schema": SCHEMA_VERSION,
        "db": db.counts(),
        "last_row_age_s": lv["age_s"],
        "stale": lv["stale"],
        "mode": lv["row"]["mode"] if lv["row"] else None,
        "source": cfg["source"],
        "hardware_enabled": cfg.get("hardware_enabled", False),
        "fan_control_enabled": cfg.get("fan_control_enabled", False),
        "server_time": time.time(),
        "ml_status": lv["row"]["ml"]["status"] if (lv["row"] and lv["row"].get("ml")) else "MODEL_UNAVAILABLE",
    }


@app.get("/api/config")
def get_config():
    return {
        k: cfg[k]
        for k in ("source", "runtime_mode", "demo_timers", "timers", "thresholds", "clear_ratio", "tick_s", "hardware_enabled", "fan_control_enabled")
        if k in cfg
    }


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
