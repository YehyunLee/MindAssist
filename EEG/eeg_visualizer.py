"""MindAssist EEG Real-Time Visualizer

Live line-graph dashboard showing:
  - Top panel:  Attention & Meditation (0–100) + state indicator
  - Bottom panel: 8 EEG wave bands (delta, theta, alpha, beta, gamma)

Uses matplotlib animation + EEGProcessor from eeg_processor.py.
Run:  python EEG/eeg_visualizer.py
"""

import os
import threading
import time
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from eeg_processor import (
    EEGProcessor, MindState, WAVE_NAMES,
    ATTENTION_THRESHOLD, MEDITATION_THRESHOLD, BLINK_THRESHOLD,
    SERIAL_PORT,
)
import bt_reset

# ── Configuration ───────────────────────────────────────────────────────────
HISTORY = 60          # how many data points to show (≈ seconds of data)
UPDATE_MS = 200       # chart refresh interval in milliseconds
STALE_TIMEOUT = 10    # seconds without eSense data before triggering reconnect

# ── Data buffers (thread-safe via deque) ────────────────────────────────────
timestamps = deque(maxlen=HISTORY)
attn_hist = deque(maxlen=HISTORY)
med_hist = deque(maxlen=HISTORY)
sig_hist = deque(maxlen=HISTORY)
state_hist = deque(maxlen=HISTORY)
blink_hist = deque(maxlen=HISTORY)

wave_hists = {name: deque(maxlen=HISTORY) for name in WAVE_NAMES}

_start_time = time.time()

# ── Shared state for auto-reconnect ────────────────────────────────────────
proc = None                # current EEGProcessor instance
_last_data_time = 0.0      # wall-clock time of last eSense callback
_reconnecting = False
_reconnect_msg = ""


# ── EEGProcessor callbacks ─────────────────────────────────────────────────
def on_data(attn, med, sig, blink, state):
    """Called ~1/sec by EEGProcessor when eSense data arrives."""
    global _last_data_time
    t = time.time() - _start_time
    timestamps.append(t)
    attn_hist.append(attn)
    med_hist.append(med)
    sig_hist.append(sig)
    blink_hist.append(blink)
    state_hist.append(state)
    _last_data_time = time.time()
    on_waves()  # keep wave_hists in sync with timestamps


def on_waves():
    """Pull latest wave values from the current processor."""
    p = proc
    if p and p.waves:
        for name in WAVE_NAMES:
            wave_hists[name].append(p.waves.get(name, 0))
    else:
        for name in WAVE_NAMES:
            if len(wave_hists[name]) > 0:
                wave_hists[name].append(wave_hists[name][-1])
            else:
                wave_hists[name].append(0)


# ── Colors ──────────────────────────────────────────────────────────────────
ESENSE_COLORS = {
    "attention": "#FF6B6B",
    "meditation": "#4ECDC4",
}

WAVE_COLORS = {
    "delta":      "#264653",
    "theta":      "#2A9D8F",
    "low-alpha":  "#E9C46A",
    "high-alpha": "#F4A261",
    "low-beta":   "#E76F51",
    "high-beta":  "#D62828",
    "low-gamma":  "#7209B7",
    "mid-gamma":  "#3A0CA3",
}

STATE_COLORS = {
    MindState.IDLE:  "#888888",
    MindState.FOCUS: "#FF6B6B",
    MindState.RELAX: "#4ECDC4",
    MindState.BLINK: "#FFD93D",
}


# ── Processor lifecycle helpers ────────────────────────────────────────────
def _create_processor():
    """Create a new EEGProcessor wired to the dashboard callbacks."""
    global proc, _last_data_time
    proc = EEGProcessor(
        on_state_change=lambda s: print(f">>> STATE: {s}"),
        on_data=on_data,
    )
    _last_data_time = time.time()  # grace period for new connection
    return proc


def _start_processor():
    """Start the current processor in a daemon thread."""
    t = threading.Thread(target=proc.start, daemon=True)
    t.start()


def _stop_processor():
    """Gracefully stop the current processor."""
    if proc:
        proc.stop()


# ── Auto-reconnect supervisor ─────────────────────────────────────────────
def _supervisor():
    """Background thread: watches for stale data and triggers BT reset + reconnect."""
    global _reconnecting, _reconnect_msg, _last_data_time

    # Give the initial connection time to establish
    time.sleep(STALE_TIMEOUT + 5)

    while True:
        time.sleep(2)

        if _reconnecting:
            continue

        # Still waiting for the very first data
        if _last_data_time == 0.0:
            continue

        elapsed = time.time() - _last_data_time
        if elapsed < STALE_TIMEOUT:
            continue

        # ── Stale data detected ──
        _reconnecting = True
        print(f"\n\u26a0  No EEG data for {elapsed:.0f}s \u2014 triggering Bluetooth reset...")

        try:
            _reconnect_msg = "Stopping processor..."
            _stop_processor()
            time.sleep(1)

            _reconnect_msg = "Resetting Bluetooth (unpair / re-pair)..."
            bt_reset.main()

            _reconnect_msg = "Waiting for serial port..."
            for _attempt in range(30):  # up to 30 seconds
                if os.path.exists(SERIAL_PORT):
                    time.sleep(2)  # extra settle time
                    break
                time.sleep(1)

            _reconnect_msg = "Reconnecting to MindWave..."
            _create_processor()
            _start_processor()

            _reconnect_msg = "Waiting for signal..."
            time.sleep(STALE_TIMEOUT)

        except Exception as e:
            print(f"  Reconnect error: {e}")
            _reconnect_msg = f"Error: {e} \u2014 retrying..."
            time.sleep(10)

        _reconnecting = False
        _reconnect_msg = ""


# ── Build the figure ────────────────────────────────────────────────────────
def build_dashboard():
    fig, (ax_esense, ax_waves) = plt.subplots(
        2, 1, figsize=(12, 7), facecolor="#1a1a2e",
        gridspec_kw={"height_ratios": [1, 1.2]},
    )
    fig.suptitle("MindAssist EEG Dashboard", color="white",
                 fontsize=16, fontweight="bold")
    fig.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08,
                        left=0.08, right=0.95)

    for ax in (ax_esense, ax_waves):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # ── Top panel: Attention / Meditation ────────────────────────────────
    ax_esense.set_title("Attention & Meditation")
    ax_esense.set_ylabel("Level (0–100)")
    ax_esense.set_ylim(-5, 105)
    ax_esense.axhline(y=ATTENTION_THRESHOLD, color=ESENSE_COLORS["attention"],
                       linestyle="--", alpha=0.4, label=f"Attn thresh ({ATTENTION_THRESHOLD})")
    ax_esense.axhline(y=MEDITATION_THRESHOLD, color=ESENSE_COLORS["meditation"],
                       linestyle="--", alpha=0.4, label=f"Med thresh ({MEDITATION_THRESHOLD})")

    line_attn, = ax_esense.plot([], [], color=ESENSE_COLORS["attention"],
                                 linewidth=2, label="Attention")
    line_med, = ax_esense.plot([], [], color=ESENSE_COLORS["meditation"],
                                linewidth=2, label="Meditation")
    ax_esense.legend(loc="upper left", fontsize=8,
                     facecolor="#16213e", edgecolor="#333",
                     labelcolor="white")

    blink_scatter = ax_esense.scatter([], [], color="#FFD93D", s=120,
                                       marker="|", zorder=5,
                                       label="Blink")

    state_text = ax_esense.text(
        0.98, 0.95, "IDLE", transform=ax_esense.transAxes,
        fontsize=14, fontweight="bold", color="#888",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                  edgecolor="#333", alpha=0.8),
    )
    sig_text = ax_esense.text(
        0.98, 0.05, "Sig: --", transform=ax_esense.transAxes,
        fontsize=9, color="#aaa", ha="right", va="bottom",
    )
    reconnect_text = ax_esense.text(
        0.5, 0.5, "", transform=ax_esense.transAxes,
        fontsize=13, fontweight="bold", color="#FFD93D",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                  edgecolor="#FFD93D", alpha=0.9),
        visible=False,
    )

    # ── Bottom panel: EEG wave bands ─────────────────────────────────────
    ax_waves.set_title("EEG Wave Bands")
    ax_waves.set_ylabel("Power")
    ax_waves.set_xlabel("Time (s)")

    wave_lines = {}
    for name in WAVE_NAMES:
        ln, = ax_waves.plot([], [], color=WAVE_COLORS[name],
                            linewidth=1.5, label=name)
        wave_lines[name] = ln
    ax_waves.legend(loc="upper left", fontsize=7, ncol=4,
                    facecolor="#16213e", edgecolor="#333",
                    labelcolor="white")

    # ── Animation update ─────────────────────────────────────────────────
    def update(_frame):
        ts = list(timestamps)
        if not ts:
            return []

        # eSense lines
        line_attn.set_data(ts, list(attn_hist))
        line_med.set_data(ts, list(med_hist))
        ax_esense.set_xlim(max(0, ts[-1] - HISTORY), ts[-1] + 1)

        # Blink markers — show as vertical yellow lines at y=95
        blinks = list(blink_hist)
        blink_ts = [ts[i] for i in range(len(blinks)) if blinks[i] >= BLINK_THRESHOLD]
        blink_ys = [95] * len(blink_ts)
        if blink_ts:
            blink_scatter.set_offsets(list(zip(blink_ts, blink_ys)))
        else:
            blink_scatter.set_offsets(np.empty((0, 2)))

        # State indicator
        if state_hist:
            current_state = state_hist[-1]
            state_text.set_text(current_state)
            state_text.set_color(STATE_COLORS.get(current_state, "#888"))

        # Signal quality
        if sig_hist:
            sq = sig_hist[-1]
            if sq == 0:
                sig_text.set_text("Signal: GOOD")
                sig_text.set_color("#4ECDC4")
            elif sq < 50:
                sig_text.set_text(f"Signal: FAIR ({sq})")
                sig_text.set_color("#E9C46A")
            elif sq < 200:
                sig_text.set_text(f"Signal: POOR ({sq})")
                sig_text.set_color("#E76F51")
            else:
                sig_text.set_text("Signal: OFF HEAD")
                sig_text.set_color("#FF6B6B")

        # Wave lines
        wave_ts = ts[:len(wave_hists[WAVE_NAMES[0]])]
        y_max = 1
        for name in WAVE_NAMES:
            data = list(wave_hists[name])[:len(wave_ts)]
            if data:
                wave_lines[name].set_data(wave_ts[:len(data)], data)
                local_max = max(data)
                if local_max > y_max:
                    y_max = local_max
        ax_waves.set_xlim(max(0, ts[-1] - HISTORY), ts[-1] + 1)
        ax_waves.set_ylim(0, y_max * 1.1 if y_max > 0 else 100)

        # Show reconnect overlay when auto-reconnecting
        if _reconnecting and _reconnect_msg:
            reconnect_text.set_text(_reconnect_msg)
            reconnect_text.set_visible(True)
        else:
            reconnect_text.set_visible(False)

        return [line_attn, line_med, blink_scatter, state_text, sig_text,
                reconnect_text] + list(wave_lines.values())

    ani = animation.FuncAnimation(
        fig, update, interval=UPDATE_MS, blit=False, cache_frame_data=False,
    )
    return fig, ani


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Always start fresh: unpair + re-pair to avoid stale BT connections
    print("Resetting Bluetooth connection...")
    try:
        bt_reset.main()
    except Exception as e:
        print(f"  BT reset warning: {e} — will retry via auto-reconnect")

    # Wait for serial port to appear after pairing
    for _ in range(15):
        if os.path.exists(SERIAL_PORT):
            time.sleep(2)
            break
        time.sleep(1)

    _create_processor()
    _start_processor()

    # Start the auto-reconnect supervisor
    sup = threading.Thread(target=_supervisor, daemon=True)
    sup.start()

    print("Starting EEG Dashboard... (close window or Ctrl+C to quit)")
    print(f"  Auto-reconnect enabled (timeout: {STALE_TIMEOUT}s)")
    fig, ani = build_dashboard()
    plt.show()

    # Cleanup after window closed
    _stop_processor()
