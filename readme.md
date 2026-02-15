# MindAssist

A **non-invasive, low-cost, mind-controlled robotic arm** using the NeuroSky MindWave Mobile 2 EEG headset to help people with limited mobility perform assistive tasks such as feeding and simple object grasping — purely through thought.

Inspired by Neuralink's CONVOY trial demo, recreated with affordable consumer hardware and open-source software.

## Problem Statement

> "One in three U.S. stroke survivors faces food insecurity — nearly twice the rate of people without stroke — because they often can't reliably feed themselves without assistance (American Heart Association, 2022). We're building a mind-controlled assistive arm so survivors can trigger an entire feeding routine with simple mental states, restoring autonomy even when fine motor control is gone."

## Hardware

- **EEG**: NeuroSky MindWave Mobile 2 (provides Attention, Meditation, and Blink values)
- **Robotic arm**: Hiwonder / LewanSoul miniArm Standard Kit — 5 DOF, high-precision digital servos, built-in Bluetooth, 6-channel knob controller for manual testing
- **Microcontroller**: miniArm Atmega328 controller board (kit default) / Arduino Uno R3 clone (backup)
- **Extras included with kit**: ESP32-Cam, glowing ultrasonic sensor, touch sensor, acceleration sensor

## Control Concept

| Mental State | Threshold | Action |
|---|---|---|
| **Focus** (high Attention) | > 65–75 | Move arm forward / lift / extend toward target |
| **Relax** (high Meditation) | > 65–75 | Retract arm / open gripper / return to rest |

## Architecture

```
MindWave Mobile 2 ──Bluetooth──▸ Laptop (Python)
                                    │
                              EEG parsing &
                              state machine (FSM)
                                    │
                              USB Serial commands
                                    │
                              Arduino miniArm ──▸ Servos
```

1. **EEG ingestion** — MindWave ↔ Bluetooth ↔ Python service parsing ThinkGear binary protocol directly (`pyserial`, no external EEG libs).
2. **State → motion planner** — Python FSM converts `FOCUS` / `RELAX` / `BLINK` into commands (`POS1`, `FEED`, `HOME`). Auto-completes the feeding routine if focus is sustained > 2 s.
3. **Robot layer** — Arduino serial parser + `Servo` library with preset joint positions (`reach()`, `lift()`, `feed()`, `home()`).

## Real-time EEG -> Commander bridge

`EEG/eeg_visualizer.py` now publishes live EEG packets on localhost UDP (`127.0.0.1:8765`), and `PythonInput/Commander.py` subscribes to that stream in real time.

- EEG stream payload fields: `attention`, `meditation`, `signal`, `blink`, `state`, `ts`
- Transport: local UDP JSON (same laptop, no cloud)
- Commander behavior: finite-state machine for robot-arm step commands (`0..3`)

FSM used by `PythonInput/Python.py`:

- `HOME` -> `REACH` when focus is high (`FOCUS`, attention >= 60) -> send `1`
- `REACH` -> `GRAB` on blink -> send `2`
- `GRAB` -> `RETURN` when relax is high (`RELAX`, meditation >= 60) -> send `3`
- `RETURN` -> `HOME` on relax/idle -> send `0`

If EEG stream is stale for 5 seconds, commander sends `0` (safe/home).

### Run both processes together

1. Terminal A: `python EEG/eeg_visualizer.py`
2. Terminal B: `python PythonInput/Commander.py`

Make sure `PORT` in `PythonInput/Python.py` matches your Arduino serial device.

## Documentation & Links

- [NeuroSky MindWave Mobile & Arduino tutorial](https://developer.neurosky.com/docs/doku.php?id=mindwave_mobile_and_arduino)
- [Hiwonder miniArm wiki](https://wiki.hiwonder.com/projects/miniArm/en/latest/)
- [miniArm Arduino IDE setup](https://docs.hiwonder.com/projects/miniArm/en/latest/docs/2.Set_Arduino_Environment.html)
- [MindWave Mobile 2 (RobotShop)](https://ca.robotshop.com/products/neurosky-mindwave-mobile-2-eeg-sensor-starter-kit)
- [miniArm Standard Kit (Amazon)](https://www.amazon.com/dp/B0DCB8KLGT?_encoding=UTF8&th=1)

Built with Codex.
