import sqlite3, json, time, pathlib

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, written_at REAL, mode TEXT, json TEXT);
CREATE INDEX IF NOT EXISTS rows_ts ON rows(ts);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, kind TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, json TEXT);
CREATE TABLE IF NOT EXISTS commands(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, cmd TEXT, payload TEXT, done INTEGER DEFAULT 0);
"""

class DB:
    def __init__(self, path):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(str(p), timeout=5, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.executescript(SCHEMA)
        self.c.commit()

    def insert_row(self, row):
        self.c.execute("INSERT INTO rows(ts,written_at,mode,json) VALUES(?,?,?,?)",
                       (row["ts"], time.time(), row["mode"], json.dumps(row)))
        self.c.commit()

    def latest_row(self):
        r = self.c.execute("SELECT json, written_at FROM rows ORDER BY id DESC LIMIT 1").fetchone()
        return (json.loads(r["json"]), r["written_at"]) if r else None

    def history(self, minutes=10, limit=5000):
        r = self.c.execute("SELECT MAX(ts) m FROM rows").fetchone()
        if r["m"] is None:
            return []
        cut = r["m"] - minutes * 60
        rows = self.c.execute("SELECT json FROM (SELECT id,json FROM rows WHERE ts>=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                              (cut, limit)).fetchall()
        return [json.loads(x["json"]) for x in rows]

    def add_event(self, ts, level, kind, msg):
        self.c.execute("INSERT INTO events(ts,level,kind,msg) VALUES(?,?,?,?)", (ts, level, kind, msg))
        self.c.commit()

    def events(self, limit=100):
        rs = self.c.execute("SELECT ts,level,kind,msg FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(x) for x in rs]

    def add_run(self, run):
        self.c.execute("INSERT INTO runs(json) VALUES(?)", (json.dumps(run),))
        self.c.commit()

    def runs(self, limit=50):
        rs = self.c.execute("SELECT json FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(x["json"]) for x in rs]

    def add_command(self, cmd, payload):
        self.c.execute("INSERT INTO commands(ts,cmd,payload) VALUES(?,?,?)", (time.time(), cmd, json.dumps(payload)))
        self.c.commit()

    def poll_commands(self):
        rs = self.c.execute("SELECT id,cmd,payload FROM commands WHERE done=0 ORDER BY id").fetchall()
        for r in rs:
            self.c.execute("UPDATE commands SET done=1 WHERE id=?", (r["id"],))
        self.c.commit()
        return [(r["cmd"], json.loads(r["payload"])) for r in rs]

    def prune(self, hours):
        self.c.execute("DELETE FROM rows WHERE written_at < ?", (time.time() - hours * 3600,))
        self.c.commit()

    def counts(self):
        return {t: self.c.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"] for t in ("rows", "events", "runs")}
