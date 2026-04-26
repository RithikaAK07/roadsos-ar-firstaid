import json
import time
import threading
from dataclasses import dataclass
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


CONFIG = {
    "twilio_sid":         "YOUR_TWILIO_SID",
    "twilio_token":       "YOUR_TWILIO_TOKEN",
    "twilio_from":        "+1XXXXXXXXXX",
    "firebase_url":       "https://your-project.firebaseio.com",
    "emergency_contacts": [],
    "trauma_center_api":  "https://api.tnhospitals.gov.in/nearby",
}


@dataclass
class AlertPayload:
    latitude:     float
    longitude:    float
    injury_type:  str
    triage_level: str
    severity:     int
    timestamp:    str = ""
    incident_id:  str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.incident_id:
            self.incident_id = f"ROADSOS-{int(time.time())}"


class EmergencyDispatcher:

    def __init__(self, config=None):
        self.config = config or CONFIG
        self._dispatched = False

    def dispatch_all(self, payload: AlertPayload):
        if self._dispatched:
            return
        self._dispatched = True

        threads = [
            threading.Thread(target=self._alert_112,      args=(payload,), daemon=True),
            threading.Thread(target=self._alert_trauma,   args=(payload,), daemon=True),
            threading.Thread(target=self._sms_contacts,   args=(payload,), daemon=True),
            threading.Thread(target=self._firebase_push,  args=(payload,), daemon=True),
        ]
        for t in threads:
            t.start()
        print(f"[Alert] {len(threads)} channels dispatched for {payload.incident_id}")

    def _alert_112(self, p: AlertPayload):
        data = {
            "type":      "road_accident",
            "subtype":   p.injury_type,
            "latitude":  p.latitude,
            "longitude": p.longitude,
            "severity":  p.severity,
            "timestamp": p.timestamp,
            "source":    "RoadSoS",
        }
        self._post_safe("https://112india.gov.in/api/alert", data, "112 India")

    def _alert_trauma(self, p: AlertPayload):
        data = {
            "lat":         p.latitude,
            "lon":         p.longitude,
            "injury":      p.injury_type,
            "triage":      p.triage_level,
            "incident_id": p.incident_id,
        }
        self._post_safe(self.config["trauma_center_api"], data, "Trauma Center")

    def _sms_contacts(self, p: AlertPayload):
        message = (
            f"ROADSOS ALERT\n"
            f"Accident detected.\n"
            f"Injury: {p.injury_type.replace('_', ' ').title()}\n"
            f"Triage: {p.triage_level}\n"
            f"Location: https://maps.google.com/?q={p.latitude},{p.longitude}\n"
            f"ID: {p.incident_id}"
        )
        for number in self.config.get("emergency_contacts", []):
            print(f"[Alert] SMS -> {number}")

    def _firebase_push(self, p: AlertPayload):
        data = {
            "incident_id": p.incident_id,
            "injury":      p.injury_type,
            "triage":      p.triage_level,
            "lat":         p.latitude,
            "lon":         p.longitude,
            "timestamp":   p.timestamp,
        }
        url = f"{self.config['firebase_url']}/incidents/{p.incident_id}.json"
        self._post_safe(url, data, "Firebase")

    def _post_safe(self, url, data, service):
        if not REQUESTS_AVAILABLE:
            print(f"[Alert] {service}: requests not installed")
            return
        try:
            r = requests.post(url, json=data, timeout=5)
            print(f"[Alert] {service}: HTTP {r.status_code}")
        except Exception as e:
            print(f"[Alert] {service}: failed ({e})")


if __name__ == "__main__":
    dispatcher = EmergencyDispatcher()
    payload = AlertPayload(
        latitude=13.0827,
        longitude=80.2707,
        injury_type="cardiac_arrest",
        triage_level="CRITICAL — Act NOW",
        severity=9,
    )
    dispatcher.dispatch_all(payload)
    time.sleep(2)
    print("[Alert] Test complete.")
