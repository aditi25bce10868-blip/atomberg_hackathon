"""Classifier wrapper. Uses the trained model if present, otherwise a clearly-labelled rules placeholder."""
import importlib.util, pathlib
from common import config
from common.schema import LABELS, LABEL_NAMES

class Classifier:
    def __init__(self, cfg):
        self.cfg, self.model, self.feat, self.names = cfg, None, None, LABEL_NAMES
        self.source = "rules-placeholder"
        mp = config.path(cfg, cfg["ml"]["model_path"])
        fp = mp.parent.parent / "features.py"
        try:
            if mp.exists() and fp.exists():
                import joblib
                spec = importlib.util.spec_from_file_location("features", fp)
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                self.model, self.feat = joblib.load(mp), mod.build_features
                self.names = [LABELS.get(c, str(c)) if isinstance(c, int) else str(c) for c in self.model.classes_]
                self.source = "model:" + mp.name
        except Exception as e:                      # never let ML break control
            print("[ml] model not loaded, using placeholder:", e)
            self.model = None

    def predict(self, sensors, deltas, thr, history):
        if self.model is not None:
            try:
                p = self.model.predict_proba([self.feat(history)])[0]
                probs = {n: float(x) for n, x in zip(self.names, p)}
                lab = max(probs, key=probs.get)
                return {"label": lab, "confidence": probs[lab], "probs": probs, "source": self.source}
            except Exception as e:
                print("[ml] predict failed, using placeholder:", e)
        r = {k: max(0.0, deltas[k] / thr[k]) for k in thr}
        steam = max(0.0, r["humidity"] - 0.5 * max(r["mq2"], r["temp_c"]))
        cook = (r["temp_c"] + r["mq2"]) / 2
        foul = max(0.0, r["mq135"] - 0.5 * max(r["humidity"], r["temp_c"], r["mq2"]))
        s = {"clean": max(0.2, 1.0 - max(steam, cook, foul)), "steam": steam, "cooking": cook, "foul": foul}
        tot = sum(s.values()) or 1.0
        probs = {k: v / tot for k, v in s.items()}
        lab = max(probs, key=probs.get)
        return {"label": lab, "confidence": probs[lab], "probs": probs, "source": "rules-placeholder"}
