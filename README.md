# Efficio Apex - dashboard backend

Two processes that talk only through a SQLite file (`data/app.db`):

| Process | Command | Job |
|---|---|---|
| **control** | `python -m control.main` | reads the source, runs ML + state machine, drives the relay, writes 1 row/second |
| **server** | `uvicorn server.app:app --host 0.0.0.0 --port 8000` | read-only API + WebSocket for the dashboard |

The server can crash or be closed and the fan logic keeps running. The relay is forced OFF on exit.
The server has **no endpoint that can switch the fan** (the only write it can do is queue "start/stop recording").

## Quick start (fake data, any laptop)
```
pip install -r requirements.txt
python -m control.main          # terminal 1
uvicorn server.app:app --port 8000   # terminal 2   -> open http://localhost:8000
pytest -q tests
```
Set `speed: 20` in `config.yaml` to watch a whole steam/cooking/foul/fault cycle in about 25 seconds.

## Data sources (`source:` in config.yaml)
| source | badge | use |
|---|---|---|
| `fake` | SIMULATED | build and polish the UI. `fake.scenario`: cycle, clean, steam, cooking, foul, fault |
| `replay` | REPLAY | replay a dataset CSV. Map your column names under `replay.columns` |
| `serial` | LIVE | **ESP32 (or any board) -> Pi over USB**, same idea as the LoRa repo |
| `live` | LIVE | Pi reads the sensors itself: fill in `hardware/sensors.py` |

**Serial format** (one line per second, `nan` for a failed sensor):
`mq135,mq2,humidity,temp_c,pir,override` e.g. `1.52,1.01,56.0,30.1,0,0`
The Pi sends `F1` / `F0` back over the same port to switch the relay. A silent link counts as a sensor fault.
Pi port is usually `/dev/ttyUSB0` or `/dev/ttyACM0`.

## Row format (one per second; the only thing the dashboard needs)
`ts, mode, sensors{mq135,mq2,humidity,temp_c,pir}, baseline{}, deltas{}, health{}, ml{label,confidence,probs,source},
state, fan_on, relay_on, fault_led, override, timers{warmup_left,run_on_left,cap_left}, demo_timers, collecting{}, truth?`
`ml.source` is `rules-placeholder` until `ml/model/model.joblib` + `ml/features.py` exist.
Contract lives in `common/schema.py` (bump `SCHEMA_VERSION` on any change).

## API
`WS /ws/live` (row + stale flag every second) | `GET /api/live` | `/api/history?minutes=10` | `/api/events` | `/api/runs` |
`/api/ml/report` | `/api/health` | `/api/config` | `POST /api/collect/start {label,note}` | `POST /api/collect/stop`

`/api/ml/report` reads `ml/reports/latest/` : `metrics.json`, `confusion_matrix.json`, `feature_importance.json`,
`dataset_summary.json`, `model_info.json`. Written by the ML person's training script; nothing here is made up.

## Plugging in the others' work
* **ML person:** ship `ml/features.py` (`build_features(history) -> list`, history = recent rows) and `ml/model/model.joblib`
  (scikit-learn, `predict_proba`). Control imports the same `features.py`. If loading fails it falls back to the placeholder.
* **Hardware person:** either stream the serial line above, or fill in `hardware/sensors.py`
  (`read_sensors`, `read_override`, `set_fan`; relay must default OFF).
* **Recording training data:** `POST /api/collect/start` -> labelled CSV in `data/runs/` with the columns in `common/schema.py`.

## Using the LoRa dashboard UI
Copy its `static/` look into `server/static/` and point the JS at `/ws/live` and `/api/*`
(rows above replace its per-node dictionary). Keep a credit to the original repo in your README and report,
and ask the author about reuse (no license file was visible).

## Before the real demo
Set `demo_timers: false` for real timings (the dashboard shows "shortened demo timers" whenever it is true).
Run control and server as `systemd` services so they start on boot in auto mode.
Mains switching needs a fused, enclosed build checked by someone experienced; use a low-voltage fan until then.

## Dashboard UI
The UI lives in `server/static/` (the folder the server already serves at `/static`):

| File | What it is |
|---|---|
| `index.html` | page layout: sidebar (bottom tab bar on phones), top bar, 7 tabs |
| `style.css` | all colours and fonts are variables at the top (`:root`), edit them there |
| `app.js` | talks to `/ws/live` and `/api/*` with same-origin addresses, so it works from any device |
| `vendor/` | Bootstrap 5.3.3, Chart.js 4.4.4, Bootstrap Icons 1.11.3 (copied in, **no internet needed**) |

Open `http://<pi-address>:8000/static/index.html` (add `#collect`, `#ml`, ... to jump to a tab).
To make `/` open it too, change the `index()` route in `server/app.py` to return `HERE / "static" / "index.html"`.

State colours: green = IDLE, amber = RUNNING, blue = RUN-ON, red = FAULT (grey = WARMUP, lilac = MANUAL).
Mode badge: SIMULATED, REPLAY or LIVE, straight from the backend row.

## Credits
* Layout idea (sidebar for the tabs, top bar, stat cards and chart cards): **Superbyte UI** by superbyte23, <https://github.com/superbyte23/superbyte-ui> (MIT licence per its README).
* Small set of CSS colour variables and a monospace font for the numbers: **fastapi-lora-dashboard** by Ishaan-Gill, <https://github.com/Ishaan-Gill/fastapi-lora-dashboard>.
  No licence file was visible, so only the idea is reused and no code was copied; ask the author before reusing more.
* **Bootstrap** 5.3.3 (MIT, <https://getbootstrap.com>), **Chart.js** 4.4.4 (MIT, <https://www.chartjs.org>) and **Bootstrap Icons** 1.11.3 (MIT, <https://icons.getbootstrap.com>).
  Their licence files are in `server/static/vendor/`.
