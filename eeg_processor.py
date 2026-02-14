"""MindAssist EEG Processor

Connects to MindWave Mobile 2 via Bluetooth serial, parses the ThinkGear
binary protocol directly, smooths Attention/Meditation signals, and emits
control states (FOCUS, RELAX, BLINK, IDLE) for downstream robotic-arm control.

No external EEG libraries required — only pyserial.
"""

import signal
import sys
import time
import struct
import threading
from collections import deque

import serial

# ── Configuration ───────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.MindWaveMobile"
BAUD_RATE = 115200

# Smoothing: rolling average window size (number of samples)
SMOOTH_WINDOW = 5

# Thresholds (0-100 scale from MindWave)
ATTENTION_THRESHOLD = 65
MEDITATION_THRESHOLD = 65
BLINK_THRESHOLD = 50          # blink strength threshold

# Sustained-focus duration: how many consecutive readings above threshold
# before we commit to a state change (prevents flickering)
SUSTAIN_COUNT = 3

# ── ThinkGear protocol byte codes ──────────────────────────────────────────
SYNC       = 0xAA
EXCODE     = 0x55
CODE_POOR_SIGNAL    = 0x02
CODE_ATTENTION      = 0x04
CODE_MEDITATION     = 0x05
CODE_BLINK          = 0x16
CODE_RAW_VALUE      = 0x80
CODE_ASIC_EEG_POWER = 0x83

WAVE_NAMES = ['delta', 'theta', 'low-alpha', 'high-alpha',
              'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']


# ── Signal smoother ─────────────────────────────────────────────────────────
class SignalSmoother:
    """Rolling-average smoother for a single EEG metric."""
    def __init__(self, window=SMOOTH_WINDOW):
        self._buf = deque(maxlen=window)

    def update(self, value):
        self._buf.append(value)
        return self.value

    @property
    def value(self):
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)


# ── State detector ──────────────────────────────────────────────────────────
class MindState:
    IDLE   = "IDLE"
    FOCUS  = "FOCUS"
    RELAX  = "RELAX"
    BLINK  = "BLINK"


class StateDetector:
    """
    Converts smoothed Attention / Meditation / Blink values into a discrete
    control state.  Requires `sustain_count` consecutive above-threshold
    readings before switching to FOCUS or RELAX (debounce).
    """
    def __init__(self,
                 attn_thresh=ATTENTION_THRESHOLD,
                 med_thresh=MEDITATION_THRESHOLD,
                 blink_thresh=BLINK_THRESHOLD,
                 sustain=SUSTAIN_COUNT):
        self.attn_thresh = attn_thresh
        self.med_thresh = med_thresh
        self.blink_thresh = blink_thresh
        self.sustain = sustain

        self._focus_streak = 0
        self._relax_streak = 0
        self._blink_flag = False
        self.state = MindState.IDLE

    def feed(self, attention, meditation, blink_strength=0):
        """Return the current MindState after ingesting new values."""

        # Blink is instantaneous — highest priority
        if blink_strength >= self.blink_thresh:
            self._blink_flag = True

        if self._blink_flag:
            self._blink_flag = False
            self.state = MindState.BLINK
            return self.state

        # Sustained focus
        if attention >= self.attn_thresh:
            self._focus_streak += 1
        else:
            self._focus_streak = 0

        # Sustained relax
        if meditation >= self.med_thresh:
            self._relax_streak += 1
        else:
            self._relax_streak = 0

        # Decide state (focus takes priority over relax when both are high)
        if self._focus_streak >= self.sustain:
            self.state = MindState.FOCUS
        elif self._relax_streak >= self.sustain:
            self.state = MindState.RELAX
        else:
            self.state = MindState.IDLE

        return self.state


# ── Callback interface ──────────────────────────────────────────────────────
class EEGProcessor:
    """
    Main processor: reads ThinkGear packets, smooths, detects state,
    and invokes a callback with every new state + raw values.
    """
    def __init__(self, on_state_change=None, on_data=None):
        """
        on_state_change(state: str)          — called when state transitions
        on_data(attn, med, signal_q, state)  — called every update cycle
        """
        self.on_state_change = on_state_change
        self.on_data = on_data

        self.attn_smoother = SignalSmoother()
        self.med_smoother = SignalSmoother()
        self.detector = StateDetector()

        self._port = None
        self._running = False
        self._prev_state = MindState.IDLE

        # Latest values (accessible from outside)
        self.attention = 0.0
        self.meditation = 0.0
        self.signal_quality = 200   # 200 = no contact
        self.blink = 0
        self.state = MindState.IDLE

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self):
        self._running = True
        signal.signal(signal.SIGINT,  self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)

        # Input listener for graceful quit
        t = threading.Thread(target=self._input_listener, daemon=True)
        t.start()

        self._connect_and_read()

    def stop(self):
        self._running = False

    def _cleanup(self, *_args):
        self._running = False
        if self._port and self._port.is_open:
            self._port.close()
            print("\nSerial port closed.")
        sys.exit(0)

    def _input_listener(self):
        while self._running:
            try:
                if input().strip().lower() == "q":
                    print("\nQuitting...")
                    self._cleanup()
            except EOFError:
                break

    # ── main loop ───────────────────────────────────────────────────────
    def _connect_and_read(self):
        while self._running:
            try:
                print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
                self._port = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=5)
                print("Connected! Reading EEG data (press 'q' + Enter to quit)...\n")

                tg = SerialThinkGear(self._port)
                blink_val = 0
                waiting_printed = False

                while self._running:
                    data_points = tg.read()
                    got_esense = False  # did this batch have attention/meditation?

                    for pt in data_points:
                        if isinstance(pt, thinkgear.PoorSignalDataPoint):
                            self.signal_quality = pt.value
                            if pt.value > 0 and not waiting_printed:
                                print(f"  Waiting for good signal... "
                                      f"(quality={pt.value}, 0=good, 200=off head)")
                                print("  Tip: adjust ear clip and forehead sensor.")
                                waiting_printed = True
                            elif pt.value == 0 and waiting_printed:
                                print("  Signal acquired! Reading EEG data...")
                                waiting_printed = False

                        elif isinstance(pt, thinkgear.AttentionDataPoint):
                            self.attention = self.attn_smoother.update(pt.value)
                            got_esense = True

                        elif isinstance(pt, thinkgear.MeditationDataPoint):
                            self.meditation = self.med_smoother.update(pt.value)
                            got_esense = True

                        elif isinstance(pt, thinkgear.BlinkDataPoint):
                            blink_val = pt.value
                            self.blink = blink_val

                        elif isinstance(pt, thinkgear.EegDataPoints):
                            pass  # EEG band powers — available if needed later

                        elif isinstance(pt, thinkgear.RawDataPoint):
                            continue  # skip raw wave (too noisy to print)

                    # Only process state + callbacks when we got real eSense data
                    if not got_esense:
                        continue

                    # Run state detection after processing the batch
                    self.state = self.detector.feed(
                        self.attention, self.meditation, blink_val
                    )
                    blink_val = 0  # reset blink after consuming

                    # Callbacks
                    if self.on_data:
                        self.on_data(self.attention, self.meditation,
                                     self.signal_quality, self.state)

                    if self.state != self._prev_state:
                        if self.on_state_change:
                            self.on_state_change(self.state)
                        self._prev_state = self.state

            except serial.SerialException as e:
                print(f"\nSerial error: {e}")
                print("Retrying in 3 s...")
                time.sleep(3)
            except IOError as e:
                print(f"\nIO error: {e}")
                print("Retrying in 3 s...")
                time.sleep(3)


# ── Standalone demo ─────────────────────────────────────────────────────────
def _print_state_change(state):
    label = {
        MindState.FOCUS: "🧠 FOCUS  → move arm forward",
        MindState.RELAX: "😌 RELAX  → retract arm",
        MindState.BLINK: "👁  BLINK  → toggle gripper",
        MindState.IDLE:  "💤 IDLE",
    }
    print(f"\n>>> STATE: {label.get(state, state)}")


def _print_data(attn, med, sig, state):
    bar_a = "█" * int(attn / 5)
    bar_m = "█" * int(med / 5)
    print(
        f"  Attn {attn:5.1f} [{bar_a:<20}] | "
        f"Med {med:5.1f} [{bar_m:<20}] | "
        f"Sig {sig:3d} | {state}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  MindAssist EEG Processor")
    print("  Thresholds — Attention ≥ %d, Meditation ≥ %d, Blink ≥ %d" %
          (ATTENTION_THRESHOLD, MEDITATION_THRESHOLD, BLINK_THRESHOLD))
    print("  Smoothing window: %d samples, sustain: %d readings" %
          (SMOOTH_WINDOW, SUSTAIN_COUNT))
    print("=" * 70)

    proc = EEGProcessor(
        on_state_change=_print_state_change,
        on_data=_print_data,
    )
    proc.start()
