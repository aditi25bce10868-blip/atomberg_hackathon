"""Single source of truth for the data contract. Bump SCHEMA_VERSION on any change."""
SCHEMA_VERSION = 1
SENSORS = ["mq135", "mq2", "humidity", "temp_c", "pir"]
CRITICAL = ("mq135", "mq2", "humidity", "temp_c")     # a fault here stops automation
LABELS = {0: "clean", 1: "steam", 2: "cooking", 3: "foul"}
LABEL_NAMES = list(LABELS.values())
LABEL_CODES = {v: k for k, v in LABELS.items()}
CSV_COLUMNS = ["run_id", "ts", "mq135", "mq2", "humidity", "temp_c", "pir", "label"]
STATES = ["WARMUP", "IDLE", "RUNNING", "RUN-ON", "FAULT", "MANUAL"]
