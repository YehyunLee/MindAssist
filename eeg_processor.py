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


# ── ThinkGear packet parser ────────────────────────────────────────────────
def parse_payload(payload):
    """Parse a ThinkGear payload into a dict of decoded values.

    Returns a dict that may contain keys:
        poor_signal, attention, meditation, blink, raw_value, waves
    """
    result = {}
    i = 0
    while i < len(payload):
        code = payload[i]
        i += 1

        # Skip extended code bytes
        while code == EXCODE and i < len(payload):
            code = payload[i]
            i += 1

        if code < 0x80:
            # Single-byte value
            if i >= len(payload):
                break
            value = payload[i]
            i += 1

            if code == CODE_POOR_SIGNAL:
                result['poor_signal'] = value
            elif code == CODE_ATTENTION:
                result['attention'] = value
            elif code == CODE_MEDITATION:
                result['meditation'] = value
            elif code == CODE_BLINK:
                result['blink'] = value
        else:
            # Multi-byte value
            if i >= len(payload):
                break
            vlength = payload[i]
            i += 1
            if i + vlength > len(payload):
                break
            value = payload[i:i + vlength]
            i += vlength

            if code == CODE_RAW_VALUE and len(value) >= 2:
                raw = value[0] * 256 + value[1]
                if raw >= 32768:
                    raw -= 65536
                result['raw_value'] = raw
            elif code == CODE_ASIC_EEG_POWER and len(value) >= 24:
                waves = {}
                for j, name in enumerate(WAVE_NAMES):
                    offset = j * 3
                    waves[name] = (value[offset] * 255 * 255 +
                                   value[offset + 1] * 255 +
                                   value[offset + 2])
                result['waves'] = waves

    return result


def read_packet(ser):
    """Block until a valid ThinkGear packet is read from serial.

    Returns parsed dict or None on timeout/error.
    """
    # Wait for double-sync
    while True:
        b = ser.read(1)
        if len(b) == 0:
            return None
        if b[0] != SYNC:
            continue
        b = ser.read(1)
        if len(b) == 0:
            return None
        if b[0] == SYNC:
            break

    # Read plength (skip if 170 = another sync)
    while True:
        b = ser.read(1)
        if len(b) == 0:
            return None
        plength = b[0]
        if plength != 170:
            break
    if plength > 170:
        return None

    # Read payload + checksum
    data = ser.read(plength + 1)
    if len(data) != plength + 1:
        return None

    payload = data[:plength]
    # (checksum validation skipped — matches working mindwave.py behavior)

    return parse_payload(payload)


# ── Main processor ─────────────────────────────────────────────────────────
class EEGProcessor:
    """
    Main processor: reads ThinkGear packets, smooths, detects state,
    and invokes callbacks with every new state + raw values.
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
        self.signal_quality = 255
        self.blink = 0
        self.raw_value = 0
        self.waves = {}
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

                waiting_printed = False

                while self._running:
                    packet = read_packet(self._port)
                    if packet is None:
                        continue

                    # Update signal quality
                    if 'poor_signal' in packet:
                        self.signal_quality = packet['poor_signal']
                        if self.signal_quality > 0 and not waiting_printed:
                            print(f"  Waiting for good signal... "
                                  f"(quality={self.signal_quality}, 0=good, 200=off head)")
                            print("  Tip: adjust ear clip and forehead sensor.")
                            waiting_printed = True
                        elif self.signal_quality == 0 and waiting_printed:
                            print("  Signal acquired! Reading EEG data...")
                            waiting_printed = False

                    # Update raw value
                    if 'raw_value' in packet:
                        self.raw_value = packet['raw_value']

                    # Update wave bands
                    if 'waves' in packet:
                        self.waves = packet['waves']

                    # Update blink
                    blink_val = 0
                    if 'blink' in packet:
                        blink_val = packet['blink']
                        self.blink = blink_val

                    # Update attention / meditation (eSense values)
                    got_esense = False
                    if 'attention' in packet:
                        self.attention = self.attn_smoother.update(packet['attention'])
                        got_esense = True
                    if 'meditation' in packet:
                        self.meditation = self.med_smoother.update(packet['meditation'])
                        got_esense = True

                    # Only run state detection + callbacks on eSense packets
                    if not got_esense:
                        continue

                    self.state = self.detector.feed(
                        self.attention, self.meditation, blink_val
                    )

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
