# RoadSoS — AR First Aid Guidance System
### IITM Road Safety Hackathon 2026

## What This Does
Turns any bystander into a first responder during road accidents.
Point your phone camera at the victim — the app detects the injury
and shows step by step first aid instructions on screen in real time.

## Key Features
- Live injury detection using computer vision
- AR overlays with triage level and survival probability
- Bystander stress detection via hand tremor analysis
- CPR metronome at 100-120 BPM
- Auto emergency alert to 112, nearest hospital and contacts
- Works fully offline — no internet needed
- Auto generates medical report for ambulance crew

## How to Run
Step 1 — Install requirements
pip install -r requirements.txt

Step 2 — Run the app
python app.py

Step 3 — Controls
SPACE = next first aid step
R = repeat audio instruction
Q = quit

## Files
- app.py — main AR system
- emergency_alert.py — sends alerts to 112 and hospitals
- train_models.py — trains YOLOv5 detection models
- requirements.txt — all libraries needed

## Tech Stack
- OpenCV for AR overlays
- YOLOv5 for injury detection
- pyttsx3 for offline voice instructions
- Python 3.10+

## Impact
- 1.5 lakh road deaths in India every year
- Average ambulance response is 8 minutes
- CPR within 3 minutes increases survival by 75 percent
- RoadSoS fills that critical gap

Built for RoadSoS Hackathon — IITM 2026
