from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
import asyncio
import serial
import threading
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

sensor_data = {
    "node1": {
        "temp": 0,
        "humidity": 0,
        "light": 0,
        "rssi": 0,
        "pdr": 0,
        "snr": 0,
        "packet_loss": 0,
        "latency": 0
    },

    "node2": {
        "temp": 0,
        "humidity": 0,
        "light": 0,
        "rssi": 0,
        "pdr": 0,
        "snr": 0,
        "packet_loss": 0,
        "latency": 0
    },

    "node3": {
        "temp": 0,
        "humidity": 0,
        "light": 0,
        "rssi": 0,
        "pdr": 0,
        "snr": 0,
        "packet_loss": 0,
        "latency": 0
    }
}

last_seen = {
    "node1": 0,
    "node2": 0,
    "node3": 0
}

def get_node_status():
    status = {}

    for node in last_seen:
        if time.time() - last_seen[node] <10:
            status[node] = "ONLINE"
        else:
            status[node] = "OFFLINE"
    return status

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "nodes": sensor_data
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "nodes": sensor_data,
                "status": get_node_status()
            })
            await asyncio.sleep(2)
    except (WebSocketDisconnect, Exception):
        print("WebSocket Closed")

def update_sensor_data(
    node,
    temp,
    humidity,
    light,
    rssi,
    snr,
    pdr,
    packet_loss,
    latency
):
    sensor_data[node]["temp"] = temp
    sensor_data[node]["humidity"] = humidity
    sensor_data[node]["light"] = light
    sensor_data[node]["rssi"] = rssi
    sensor_data[node]["snr"] = snr
    sensor_data[node]["pdr"] = pdr
    sensor_data[node]["packet_loss"] = packet_loss
    sensor_data[node]["latency"] = latency
    last_seen[node] = time.time()

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

def connect_serial():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
            print("ESP32 connected")
            return ser
        except serial.SerialException:
            print("Waiting for ESP32...")
            time.sleep(2)

def read_serial():
    global ser

    while True:
        try:
            line = ser.readline().decode().strip()
            parts = line.split(",")

            if len(parts) != 9:
                continue

            node = "node" + parts[0]

            if node not in sensor_data:
                continue

            update_sensor_data(
                node,
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                int(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
                float(parts[8])
            )
        except serial.SerialException as e:
            print("ESP32 disconnected:", e)
            ser.close()
            ser = connect_serial()

        except Exception as e:
            print("Serial Error:", e)

# UI DEVELOPMENT MODE: ENABLE_SERIAL = False
# HARDWARE MODE: ENABLE_SERIAL = True
ENABLE_SERIAL = True

if ENABLE_SERIAL:
    serial_thread = threading.Thread(
        target=read_serial,
        daemon=True
    )
    serial_thread.start()
