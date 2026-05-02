# RoadSoS — Be Someone's Reason to Survive

Every 4 minutes, someone dies in a road accident in India.
Not always because help was too far. But because nobody around knew what to do.

RoadSoS changes that.

---

## The Problem

Imagine witnessing a road accident. The victim is bleeding. People gather around.
Everyone is watching. Nobody is helping. Not because they don't care.
But because they are scared. They don't know what to do.
The ambulance is 8 minutes away. Those 8 minutes can cost a life.

---

## What RoadSoS Does

Point your phone camera at the victim.
That is all you need to do.

RoadSoS looks at the victim through your camera, figures out the injury,
and shows you exactly what to do — step by step — right on your screen.
Like a doctor guiding you through your phone.

At the same time, it automatically sends the GPS location of the accident
to the nearest hospital, ambulance and your emergency contacts.
So while you are giving first aid, help is already on the way.

---

## Features

- Detects injury type from live camera using computer vision
- Shows AR step by step first aid instructions on screen
- Reads instructions out loud so you don't need to look away
- Detects if you are panicking and simplifies instructions automatically
- Pulses a CPR beat guide at exactly 100 to 120 per minute
- Shows survival probability countdown in real time
- Captures GPS location of accident automatically
- Sends injury type and location to nearest trauma center instantly
- Fires emergency SMS to your saved contacts within seconds
- Alerts India 112 national emergency dispatch automatically
- Works completely offline — no internet needed at accident scene
- Generates a full medical report for the ambulance crew on arrival

---

## How the GPS Alert Works

The moment an injury is detected on camera:

1. The app captures the exact GPS coordinates of where you are
2. It finds the nearest trauma center and ambulance
3. It sends the injury type, triage level and your live location to the hospital
4. It sends an emergency SMS to your saved personal contacts
5. All of this happens in under 3 seconds — automatically

You do not press any button. You do not make any call.
RoadSoS does it all while you focus on helping the victim.

---

## How to Run

Install the requirements:
pip install -r requirements.txt

Run the app:
python app.py

Controls:
SPACE — move to next first aid step
R — repeat the audio instruction
Q — quit the app

---

## Tech Used

- OpenCV — for real time AR overlays on camera feed
- YOLOv5 — for detecting injury type from camera
- pyttsx3 — for reading instructions out loud, works offline
- Firebase — for sending real time GPS alert to nearby hospitals
- Twilio — for emergency SMS to personal contacts
- India 112 API — for national emergency dispatch
- Python 3.10

---

## Why This Matters

1.5 lakh people die in road accidents in India every year.
Most of them die not in hospitals. They die on the road.
Waiting for help that arrives too late.

The average ambulance takes 8 minutes to arrive.
A person in cardiac arrest has 3 minutes before brain damage begins.
That gap of 5 minutes is where RoadSoS lives.

We are not replacing doctors or ambulances.
We are just making sure the victim is not alone until they arrive.

---

Built with the belief that technology should save lives.
RoadSoS — IITM Road Safety Hackathon 2026
