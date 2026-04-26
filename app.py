import cv2
import numpy as np
import pyttsx3
import time
import json
import threading
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import deque
from datetime import datetime


class InjuryType(Enum):
    CARDIAC_ARREST   = "cardiac_arrest"
    SEVERE_BLEEDING  = "severe_bleeding"
    BURNS            = "burns"
    SPINAL_SUSPECTED = "spinal_suspected"
    UNCONSCIOUS      = "unconscious"
    CHOKING          = "choking"
    UNKNOWN          = "unknown"


class TriageLevel(Enum):
    RED    = (0, 0, 255,   "CRITICAL — Act NOW",    1)
    YELLOW = (0, 165, 255, "URGENT — Act in 30s",   2)
    GREEN  = (0, 200, 0,   "STABLE — Monitor",      3)
    BLACK  = (80, 80, 80,  "ASSESS — Unresponsive", 0)

    def __new__(cls, r, g, b, label, priority):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.color = (b, g, r)
        obj.label = label
        obj.priority = priority
        return obj


@dataclass
class DetectionResult:
    injury_type: InjuryType      = InjuryType.UNKNOWN
    confidence:  float           = 0.0
    bbox:        Optional[tuple] = None
    triage:      TriageLevel     = TriageLevel.BLACK
    severity:    int             = 0


@dataclass
class ResponderState:
    is_panicked:        bool  = False
    tremor_score:       float = 0.0
    current_step:       int   = 0
    step_confirmed:     bool  = False
    session_start:      float = field(default_factory=time.time)
    compressions_done:  int   = 0
    last_cpr_beat:      float = 0.0
    voice_spoken_steps: set   = field(default_factory=set)


PROTOCOLS: dict[InjuryType, list[dict]] = {
    InjuryType.CARDIAC_ARREST: [
        {"step": 1, "title": "CHECK SAFETY",        "detail": "Ensure the scene is safe. Tap shoulders firmly. Shout: Are you okay?"},
        {"step": 2, "title": "CALL FOR HELP",        "detail": "Call 112 now. Ask a bystander to find an AED immediately."},
        {"step": 3, "title": "OPEN AIRWAY",          "detail": "Tilt head back. Lift chin. Look, listen and feel for breathing for 10 seconds."},
        {"step": 4, "title": "BEGIN COMPRESSIONS",   "detail": "Heel of hand on center of chest. 30 compressions at 100 to 120 per minute. Push 5 to 6 cm deep."},
        {"step": 5, "title": "GIVE RESCUE BREATHS",  "detail": "Pinch nose. Seal mouth. Give 2 breaths, 1 second each. Watch chest rise."},
        {"step": 6, "title": "CONTINUE CPR",         "detail": "Repeat 30 compressions then 2 breaths. Do not stop until EMS arrives or AED is ready."},
    ],
    InjuryType.SEVERE_BLEEDING: [
        {"step": 1, "title": "PROTECT YOURSELF",     "detail": "Use gloves or a plastic bag. Avoid direct contact with blood."},
        {"step": 2, "title": "APPLY PRESSURE",       "detail": "Press hard on wound using cloth. Do not remove it. Add more on top if soaked."},
        {"step": 3, "title": "ELEVATE LIMB",         "detail": "If arm or leg: raise it above the level of the heart."},
        {"step": 4, "title": "TOURNIQUET",           "detail": "If bleeding is uncontrollable: apply tourniquet 5 cm above wound. Note time of application."},
        {"step": 5, "title": "MONITOR VICTIM",       "detail": "Keep victim warm, calm, and lying down. Talk to them until EMS arrives."},
    ],
    InjuryType.BURNS: [
        {"step": 1, "title": "REMOVE FROM SOURCE",   "detail": "Stop the burning. Remove clothing unless stuck to skin. Remove jewelry."},
        {"step": 2, "title": "COOL THE BURN",        "detail": "Run cool water over burn for 20 minutes. Do not use ice."},
        {"step": 3, "title": "COVER THE BURN",       "detail": "Cover loosely with cling film or clean cloth. Do not burst blisters."},
        {"step": 4, "title": "MANAGE SHOCK",         "detail": "Keep victim warm. Do not give food or water. Elevate burned area if possible."},
    ],
    InjuryType.UNCONSCIOUS: [
        {"step": 1, "title": "CHECK RESPONSE",       "detail": "Tap shoulder and shout. No response? Check breathing."},
        {"step": 2, "title": "RECOVERY POSITION",    "detail": "If breathing: turn onto side. Support head, bend top knee forward."},
        {"step": 3, "title": "MONITOR AIRWAY",       "detail": "Keep airway open. Check breathing every 30 seconds until help arrives."},
    ],
    InjuryType.CHOKING: [
        {"step": 1, "title": "ENCOURAGE COUGHING",   "detail": "Ask them to cough forcefully. Do not pat their back yet."},
        {"step": 2, "title": "BACK BLOWS",           "detail": "5 firm blows between shoulder blades with heel of your hand."},
        {"step": 3, "title": "ABDOMINAL THRUSTS",    "detail": "Stand behind them. Hands above navel below chest. Pull inward and upward 5 times."},
        {"step": 4, "title": "ALTERNATE",            "detail": "Alternate 5 back blows and 5 abdominal thrusts until object clears or they lose consciousness."},
    ],
}


class MockInjuryDetector:

    def __init__(self):
        self.frame_history = deque(maxlen=10)
        self.confidence_smoother = deque(maxlen=5)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.add(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        red_ratio = np.sum(red_mask > 0) / (h * w)

        self.frame_history.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        motion_score = 0.0
        if len(self.frame_history) >= 2:
            diff = cv2.absdiff(self.frame_history[-1], self.frame_history[-2])
            motion_score = np.mean(diff) / 255.0

        brightness = np.mean(hsv[:, :, 2]) / 255.0

        cx, cy = w // 2, h // 2
        roi = frame[cy-80:cy+80, cx-60:cx+60]
        if roi.size > 0:
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(roi_hsv, np.array([0, 20, 70]), np.array([25, 180, 255]))
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        else:
            skin_ratio = 0.0

        if red_ratio > 0.04:
            conf = min(0.95, 0.5 + red_ratio * 8)
            return DetectionResult(
                injury_type=InjuryType.SEVERE_BLEEDING,
                confidence=conf,
                bbox=(cx-60, cy-80, cx+60, cy+80),
                triage=TriageLevel.RED,
                severity=min(10, int(red_ratio * 120))
            )
        elif skin_ratio > 0.15 and motion_score < 0.01 and brightness < 0.6:
            return DetectionResult(
                injury_type=InjuryType.CARDIAC_ARREST,
                confidence=0.82,
                bbox=(cx-80, cy-100, cx+80, cy+60),
                triage=TriageLevel.RED,
                severity=9
            )
        elif skin_ratio > 0.1 and motion_score < 0.02:
            return DetectionResult(
                injury_type=InjuryType.UNCONSCIOUS,
                confidence=0.70,
                bbox=(cx-60, cy-80, cx+60, cy+80),
                triage=TriageLevel.YELLOW,
                severity=6
            )

        return DetectionResult()


class TremorAnalyzer:

    def __init__(self):
        self.prev_gray = None
        self.tremor_history = deque(maxlen=30)

    def analyze(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        tremor = float(np.percentile(magnitude, 90)) / 20.0
        self.tremor_history.append(tremor)
        self.prev_gray = gray
        return min(1.0, np.mean(self.tremor_history))


class ARRenderer:

    FONTS = {
        "large":  (cv2.FONT_HERSHEY_DUPLEX, 1.1, 2),
        "medium": (cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2),
        "small":  (cv2.FONT_HERSHEY_PLAIN, 1.0, 1),
    }

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.anim_t = 0.0

    def tick(self):
        self.anim_t += 0.05

    @staticmethod
    def _put_text_bg(frame, text, pos, font, scale, thickness, fg, bg, padding=6):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + padding), bg, -1)
        cv2.putText(frame, text, pos, font, scale, fg, thickness, cv2.LINE_AA)

    def _pulse_alpha(self, base=0.6, amp=0.3):
        return base + amp * math.sin(self.anim_t * 3)

    def draw_scan_lines(self, frame):
        for i in range(0, self.h, 4):
            alpha = 0.05 + 0.03 * math.sin(self.anim_t * 2 + i * 0.05)
            overlay = frame.copy()
            cv2.line(overlay, (0, i), (self.w, i), (0, 255, 120), 1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_hud_frame(self, frame):
        c = (0, 220, 255)
        t = 3
        sz = 40
        corners = [(sz, sz), (self.w - sz, sz), (sz, self.h - sz), (self.w - sz, self.h - sz)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            cv2.line(frame, (cx, cy), (cx + dx * 30, cy), c, t)
            cv2.line(frame, (cx, cy), (cx, cy + dy * 30), c, t)

    def draw_triage_banner(self, frame, triage: TriageLevel):
        bh = 46
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, bh), triage.color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        font, scale, thick = self.FONTS["large"]
        pulse = int(220 + 35 * self._pulse_alpha(0, 1))
        cv2.putText(frame, f"  {triage.label}", (14, 32),
                    font, scale, (255, 255, pulse), thick, cv2.LINE_AA)

    def draw_injury_box(self, frame, result: DetectionResult):
        if not result.bbox:
            return
        x1, y1, x2, y2 = result.bbox
        col = result.triage.color
        glow = (min(255, col[0] + 60), min(255, col[1] + 60), min(255, col[2] + 60))
        cv2.rectangle(frame, (x1, y1), (x2, y2), glow, 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), col, cv2.MARKER_CROSS, 24, 2)
        self._put_text_bg(frame, f"CONF {result.confidence * 100:.0f}%",
                          (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_PLAIN, 1.2, 1,
                          (255, 255, 255), col)

    def draw_step_card(self, frame, step: dict, step_num: int, total: int, tremor: float):
        x, y = 14, self.h - 170
        cw, ch = self.w - 28, 155
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + cw, y + ch), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 200, 255), 2)

        dot_x = x + 10
        for i in range(total):
            col = (0, 200, 255) if i < step_num else (60, 60, 60)
            cv2.circle(frame, (dot_x + i * 20, y + 14), 5, col, -1)

        cv2.putText(frame, step["title"],
                    (x + 10, y + 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)

        detail = step["detail"]
        if tremor > 0.5:
            detail = detail.split(".")[0] + "."

        words = detail.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) + 1 > 55:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)

        for i, ln in enumerate(lines[:3]):
            cv2.putText(frame, ln, (x + 10, y + 68 + i * 24),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (210, 255, 210), 1, cv2.LINE_AA)

        cv2.putText(frame, "[ SPACE ] Next Step   [ R ] Repeat Audio   [ Q ] Quit",
                    (x + 10, y + ch - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (130, 130, 160), 1)

    def draw_cpr_metronome(self, frame, beat_phase: float):
        cx, cy = self.w - 80, 110
        r = int(35 + 15 * abs(math.sin(beat_phase)))
        alpha = 0.5 + 0.4 * abs(math.sin(beat_phase))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "PUSH!", (cx - 24, cy + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    def draw_golden_hour(self, frame, elapsed_sec: float, injury: InjuryType):
        golden_secs = {
            InjuryType.CARDIAC_ARREST:  180,
            InjuryType.SEVERE_BLEEDING: 600,
        }.get(injury, 3600)

        ratio = min(1.0, elapsed_sec / golden_secs)
        survival = max(0, int(100 - 60 * ratio))
        bar_w = int((self.w - 140) * (1 - ratio))
        bar_col = (0, int(255 * (1 - ratio)), int(255 * ratio))

        cv2.putText(frame, f"SURVIVAL: {survival}%",
                    (14, self.h - 195), cv2.FONT_HERSHEY_DUPLEX, 0.7, bar_col, 2)
        cv2.rectangle(frame, (14, self.h - 185), (14 + bar_w, self.h - 175), bar_col, -1)
        cv2.rectangle(frame, (14, self.h - 185), (self.w - 126, self.h - 175), (80, 80, 80), 2)

    def draw_tremor_indicator(self, frame, tremor: float):
        label = "CALM" if tremor < 0.3 else ("STRESSED" if tremor < 0.6 else "PANICKED")
        col = (0, 200, 0) if tremor < 0.3 else ((0, 165, 255) if tremor < 0.6 else (0, 0, 255))
        self._put_text_bg(frame, f"Responder: {label}",
                          (14, 66), cv2.FONT_HERSHEY_PLAIN, 1.2, 1, (255, 255, 255), col)

    def draw_no_victim(self, frame):
        self.draw_scan_lines(frame)
        self.draw_hud_frame(frame)
        cx, cy = self.w // 2, self.h // 2
        cv2.putText(frame, "POINT CAMERA AT VICTIM",
                    (cx - 190, cy), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "RoadSoS AR First Aid  |  IITM 2026",
                    (14, self.h - 10), cv2.FONT_HERSHEY_PLAIN, 1.1, (80, 80, 120), 1)


class VoiceEngine:

    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 155)
        self._engine.setProperty("volume", 1.0)
        self._busy = False
        self._lock = threading.Lock()

    def speak(self, text: str, slow: bool = False):
        def _run():
            with self._lock:
                self._busy = True
                self._engine.setProperty("rate", 120 if slow else 155)
                self._engine.say(text)
                self._engine.runAndWait()
                self._busy = False
        if not self._busy:
            threading.Thread(target=_run, daemon=True).start()


class IncidentReport:

    def generate(self, state: ResponderState, result: DetectionResult) -> dict:
        return {
            "timestamp":        datetime.now().isoformat(),
            "incident_id":      f"ROADSOS-{int(time.time())}",
            "injury_detected":  result.injury_type.value,
            "triage_level":     result.triage.label,
            "severity_score":   result.severity,
            "response_time_s":  round(time.time() - state.session_start, 1),
            "steps_completed":  state.current_step,
            "compressions":     state.compressions_done,
            "responder_stress": round(state.tremor_score, 2),
            "aha_protocol":     "AHA 2025",
            "system":           "RoadSoS v1.0",
        }

    def save(self, report: dict, path="incident_report.json"):
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[RoadSoS] Incident report saved to {path}")
        print(json.dumps(report, indent=2))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[RoadSoS] No camera found. Exiting.")
        return

    ret, test_frame = cap.read()
    H, W = test_frame.shape[:2] if ret else (480, 640)

    detector  = MockInjuryDetector()
    tremor_a  = TremorAnalyzer()
    renderer  = ARRenderer(W, H)
    voice     = VoiceEngine()
    reporter  = IncidentReport()
    state     = ResponderState()

    current_result   = DetectionResult()
    cpr_beat_phase   = 0.0
    detection_frames = 0
    CONFIRM_FRAMES   = 20

    voice.speak("RoadSoS activated. Point camera at the victim.")
    print("\n[RoadSoS] Ready. SPACE=next step | R=repeat audio | Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        renderer.tick()

        raw    = detector.detect(frame)
        tremor = tremor_a.analyze(frame)
        state.tremor_score = tremor

        if raw.injury_type != InjuryType.UNKNOWN and raw.confidence > 0.60:
            detection_frames += 1
        else:
            detection_frames = max(0, detection_frames - 2)

        if detection_frames >= CONFIRM_FRAMES:
            if current_result.injury_type != raw.injury_type:
                current_result = raw
                state.current_step = 0
                state.session_start = time.time()
                state.voice_spoken_steps = set()
                inj_name = raw.injury_type.value.replace("_", " ").title()
                voice.speak(
                    f"Detected: {inj_name}. "
                    f"{PROTOCOLS[raw.injury_type][0]['title']}. "
                    f"{PROTOCOLS[raw.injury_type][0]['detail']}",
                    slow=tremor > 0.5
                )
                print(f"[RoadSoS] Locked: {raw.injury_type.value} conf={raw.confidence:.2f}")
        else:
            current_result = DetectionResult()

        protocol = PROTOCOLS.get(current_result.injury_type)

        if current_result.injury_type == InjuryType.UNKNOWN or not protocol:
            renderer.draw_no_victim(frame)
        else:
            renderer.draw_triage_banner(frame, current_result.triage)
            renderer.draw_injury_box(frame, current_result)
            renderer.draw_tremor_indicator(frame, tremor)

            elapsed = time.time() - state.session_start
            renderer.draw_golden_hour(frame, elapsed, current_result.injury_type)

            step_idx = min(state.current_step, len(protocol) - 1)
            step     = protocol[step_idx]
            renderer.draw_step_card(frame, step, step_idx, len(protocol), tremor)

            if current_result.injury_type == InjuryType.CARDIAC_ARREST and step_idx >= 3:
                cpr_beat_phase += 0.1
                renderer.draw_cpr_metronome(frame, cpr_beat_phase)
                if abs(math.sin(cpr_beat_phase)) < 0.05:
                    state.compressions_done += 1

            if step_idx not in state.voice_spoken_steps:
                voice.speak(f"{step['title']}. {step['detail']}", slow=tremor > 0.5)
                state.voice_spoken_steps.add(step_idx)

        renderer.draw_hud_frame(frame)
        cv2.imshow("RoadSoS — AR First Aid  |  IITM 2026", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' ') and protocol:
            if state.current_step < len(protocol) - 1:
                state.current_step += 1
                new_step = protocol[state.current_step]
                voice.speak(
                    f"Step {state.current_step + 1}. {new_step['title']}. {new_step['detail']}",
                    slow=tremor > 0.5
                )
                print(f"[RoadSoS] Step {state.current_step + 1}: {new_step['title']}")
            else:
                voice.speak("All steps completed. Maintain care until EMS arrives.")
        elif key == ord('r') and protocol:
            step = protocol[min(state.current_step, len(protocol) - 1)]
            voice.speak(f"{step['title']}. {step['detail']}", slow=tremor > 0.5)

    cap.release()
    cv2.destroyAllWindows()

    if current_result.injury_type != InjuryType.UNKNOWN:
        report = reporter.generate(state, current_result)
        reporter.save(report)

    print("[RoadSoS] Session ended.")


if __name__ == "__main__":
    main()
