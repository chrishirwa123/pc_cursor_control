#!/usr/bin/env python3
"""
Gesture Mouse - control your cursor, scroll and click with hand gestures.

Gestures (one hand, palm facing the camera):
    MOVE    : Index finger up, middle finger down      -> cursor follows index tip
    SCROLL  : Index + middle up (ring/pinky down)      -> move hand up/down to scroll
    CLICK   : Touch thumb tip to index fingertip       -> left click (pinch)
    LOCK    : All five fingers extended, held ~1.8 s   -> locks the computer screen

Keys (in the preview window):
    q  - quit
    p  - pause / resume control (preview keeps running)

Dependencies:
    pip install opencv-python mediapipe pyautogui numpy

Uses the MediaPipe Tasks API (HandLandmarker), which works on current MediaPipe
releases where the old `mp.solutions` module no longer exists. The hand model
file (~8 MB) is downloaded automatically on first run.
"""

import math
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# =============================================================================
# CONFIGURATION  (tweak these to taste)
# =============================================================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


@dataclass
class Config:
    # --- Camera ---
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    camera_fps: int = 60

    # --- MediaPipe ---
    detection_confidence: float = 0.7
    presence_confidence: float = 0.6
    tracking_confidence: float = 0.6

    # --- Cursor mapping ---
    # Fraction of the frame ignored at each edge. The hand only needs to travel
    # inside this "active box" to reach every corner of the screen.
    margin_x: float = 0.15
    margin_top: float = 0.12
    margin_bottom: float = 0.30

    # --- Smoothing (adaptive exponential moving average, in screen pixels) ---
    alpha_min: float = 0.07              # heavy smoothing when moving slowly (kills jitter)
    alpha_max: float = 0.65              # light smoothing when moving fast (kills lag)
    speed_ref_px: float = 140.0          # per-frame movement at which alpha reaches alpha_max
    deadzone_px: float = 1.5             # ignore sub-pixel tremor

    # --- Finger-extension detection ---
    extend_margin: float = 1.08          # tip must be this much farther from wrist than PIP joint
    mode_confirm_frames: int = 2         # frames a gesture must persist before switching modes

    # --- Click (pinch) ---
    # Ratios are relative to hand size (wrist -> middle-finger knuckle), so they
    # work at any distance from the camera.
    pinch_on_ratio: float = 0.22         # thumb-index closer than this  => pinch begins
    pinch_off_ratio: float = 0.36        # thumb-index farther than this => pinch released
    pinch_freeze_ratio: float = 0.50     # freeze cursor when approaching a pinch (no drift on click)
    click_cooldown_s: float = 0.40       # debounce between clicks
    click_flash_s: float = 0.35          # how long "CLICKED" stays on screen

    # --- Scrolling ---
    scroll_gain: float = 150.0           # base sensitivity
    scroll_accel: float = 25.0           # extra speed for fast hand movement
    scroll_deadzone: float = 0.0025      # normalized per-frame movement ignored
    scroll_smooth: float = 0.5           # EMA factor for vertical delta (0..1)
    invert_scroll: bool = False          # flip direction if you prefer "natural" scrolling

    # --- Screen lock gesture (all five fingers open) ---
    lock_hold_s: float = 1.8             # gesture must be held this long (seconds)
    lock_grace_s: float = 0.15           # tolerated tracking dropout while holding
    lock_cooldown_s: float = 8.0         # no re-trigger for this long after a lock
    lock_extend_margin: float = 1.15     # stricter than normal: fingers must be FULLY open
    lock_thumb_ratio: float = 0.60       # thumb-tip to index-knuckle distance / hand size


# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================
WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
THUMB_IP = 3
INDEX_MCP = 5

# Skeleton connections (same as MediaPipe's HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# BGR colours for the overlay
COLORS = {
    "MOVING": (80, 220, 80),
    "SCROLLING": (255, 170, 60),
    "CLICKED": (60, 60, 255),
    "PINCH": (60, 200, 255),
    "IDLE": (170, 170, 170),
    "PAUSED": (0, 140, 255),
    "NO HAND": (120, 120, 120),
    "LOCKING SCREEN...": (60, 60, 255),
    "SCREEN LOCKED": (60, 60, 255),
}


# =============================================================================
# MODEL DOWNLOAD
# =============================================================================
def ensure_model() -> str:
    """Download the HandLandmarker model next to this script if it's missing."""
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH

    print("Downloading hand tracking model (one-time, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the model automatically ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        ) from exc
    print("Model downloaded.")
    return MODEL_PATH


# =============================================================================
# THREADED CAMERA READER  (keeps latency low by always serving the newest frame)
# =============================================================================
class VideoStream:
    def __init__(self, cfg: Config):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.camera_index, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {cfg.camera_index}. "
                "Check that no other app is using it, or change Config.camera_index."
            )

        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "VideoStream":
        self._thread.start()
        return self

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._frame_id, (None if self._frame is None else self._frame.copy())

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# =============================================================================
# SMOOTHING
# =============================================================================
class AdaptiveEMA:
    """
    Exponential moving average whose smoothing strength adapts to hand speed.

    Slow movement  -> small alpha -> very stable, jitter-free cursor.
    Fast movement  -> large alpha -> responsive, minimal lag.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.x: Optional[float] = None
        self.y: Optional[float] = None

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self.x, self.y = x, y

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self.x is None or self.y is None:
            self.x, self.y = x, y
            return x, y

        dist = math.hypot(x - self.x, y - self.y)
        if dist < self.cfg.deadzone_px:
            return self.x, self.y

        t = min(dist / self.cfg.speed_ref_px, 1.0)
        alpha = self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * t
        self.x += alpha * (x - self.x)
        self.y += alpha * (y - self.y)
        return self.x, self.y


# =============================================================================
# GESTURE CONTROLLER
# =============================================================================
class GestureController:
    def __init__(self, cfg: Config, screen_w: int, screen_h: int):
        self.cfg = cfg
        self.sw, self.sh = screen_w, screen_h
        self.smoother = AdaptiveEMA(cfg)

        # Mode stabilisation
        self.mode = "IDLE"
        self._candidate = "IDLE"
        self._candidate_count = 0

        # Click state
        self.pinching = False
        self.last_click_time = 0.0
        self.flash_until = 0.0
        self.pinch_ratio = 1.0

        # Scroll state
        self._prev_scroll_y: Optional[float] = None
        self._scroll_dy_smooth = 0.0
        self._scroll_accum = 0.0
        # pyautogui on Windows interprets 120 units as one wheel notch;
        # on macOS/Linux units are lines.
        self._scroll_unit = 40 if platform.system() == "Windows" else 1

    # ---------- helpers ----------
    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _extended(self, pts: np.ndarray, tip: int, pip: int) -> bool:
        """Finger is extended if its tip is farther from the wrist than its PIP joint.
        Rotation-invariant, so it works even if the hand is tilted."""
        return self._dist(pts[tip], pts[WRIST]) > self._dist(pts[pip], pts[WRIST]) * self.cfg.extend_margin

    def _stabilise(self, raw: str) -> str:
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = raw, 1
        if self._candidate_count >= self.cfg.mode_confirm_frames and raw != self.mode:
            self._on_mode_change(self.mode, raw)
            self.mode = raw
        return self.mode

    def _on_mode_change(self, old: str, new: str) -> None:
        self._prev_scroll_y = None
        self._scroll_dy_smooth = 0.0
        self._scroll_accum = 0.0
        if new == "MOVING":
            # Start the glide from the current cursor position instead of jumping.
            cx, cy = pyautogui.position()
            self.smoother.reset(float(cx), float(cy))

    def reset(self) -> None:
        """Call when the hand is lost."""
        self.mode = self._candidate = "IDLE"
        self._candidate_count = 0
        self.pinching = False
        self._prev_scroll_y = None
        self._scroll_dy_smooth = 0.0
        self._scroll_accum = 0.0

    # ---------- actions ----------
    def _move_cursor(self, norm_x: float, norm_y: float) -> None:
        c = self.cfg
        x = np.interp(norm_x, (c.margin_x, 1.0 - c.margin_x), (0, self.sw - 1))
        y = np.interp(norm_y, (c.margin_top, 1.0 - c.margin_bottom), (0, self.sh - 1))
        sx, sy = self.smoother.update(float(x), float(y))
        sx = int(np.clip(sx, 0, self.sw - 1))
        sy = int(np.clip(sy, 0, self.sh - 1))
        pyautogui.moveTo(sx, sy, _pause=False)

    def _scroll(self, norm_y: float) -> None:
        c = self.cfg
        if self._prev_scroll_y is None:
            self._prev_scroll_y = norm_y
            return

        dy = self._prev_scroll_y - norm_y            # positive = hand moved UP
        self._prev_scroll_y = norm_y
        if c.invert_scroll:
            dy = -dy

        self._scroll_dy_smooth += c.scroll_smooth * (dy - self._scroll_dy_smooth)
        d = self._scroll_dy_smooth
        if abs(d) < c.scroll_deadzone:
            return

        # Dynamic speed: faster hand movement => disproportionately faster scroll.
        amount = d * c.scroll_gain * (1.0 + abs(d) * c.scroll_accel)
        self._scroll_accum += amount

        whole = int(self._scroll_accum)               # accumulate fractions, send whole lines
        if whole != 0:
            pyautogui.scroll(whole * self._scroll_unit, _pause=False)
            self._scroll_accum -= whole

    # ---------- main per-frame update ----------
    def update(self, pts: np.ndarray, norm: np.ndarray, now: float) -> str:
        """
        pts  : (21,2) landmark positions in pixels (for geometry)
        norm : (21,2) landmark positions normalised 0..1 (for mapping)
        Returns the label to display.
        """
        c = self.cfg

        # ----- Pinch detection with hysteresis (scale-invariant) -----
        hand_size = max(self._dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)
        self.pinch_ratio = self._dist(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size

        was_pinching = self.pinching
        if self.pinch_ratio < c.pinch_on_ratio:
            self.pinching = True
        elif self.pinch_ratio > c.pinch_off_ratio:
            self.pinching = False

        # Click fires once, on the pinch's rising edge, subject to cooldown.
        if self.pinching and not was_pinching and (now - self.last_click_time) >= c.click_cooldown_s:
            pyautogui.click(_pause=False)
            self.last_click_time = now
            self.flash_until = now + c.click_flash_s

        clicked_recently = now < self.flash_until

        # While pinching (or about to), freeze the cursor so the click lands where
        # the user was pointing instead of drifting as the fingers close.
        if self.pinching or self.pinch_ratio < c.pinch_freeze_ratio:
            self._prev_scroll_y = None
            return "CLICKED" if clicked_recently else "PINCH"

        # ----- Finger states -----
        idx = self._extended(pts, INDEX_TIP, INDEX_PIP)
        mid = self._extended(pts, MIDDLE_TIP, MIDDLE_PIP)
        ring = self._extended(pts, RING_TIP, RING_PIP)
        pinky = self._extended(pts, PINKY_TIP, PINKY_PIP)

        if idx and mid and not ring and not pinky:
            raw = "SCROLLING"
        elif idx and not mid:
            raw = "MOVING"
        else:
            raw = "IDLE"

        mode = self._stabilise(raw)

        if mode == "MOVING":
            self._move_cursor(norm[INDEX_TIP][0], norm[INDEX_TIP][1])
        elif mode == "SCROLLING":
            self._scroll((norm[INDEX_TIP][1] + norm[MIDDLE_TIP][1]) / 2.0)

        return "CLICKED" if clicked_recently else mode


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def draw_hand(frame: np.ndarray, pts: np.ndarray) -> None:
    """Draw the hand skeleton and joints (replacement for mp.solutions.drawing_utils)."""
    p = pts.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, (p[a][0], p[a][1]), (p[b][0], p[b][1]), (255, 255, 255), 2, cv2.LINE_AA)
    for x, y in p:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), cv2.FILLED)


def draw_overlay(frame: np.ndarray, label: str, fps: float, cfg: Config, ratio: Optional[float]) -> None:
    h, w = frame.shape[:2]
    color = COLORS.get(label, (255, 255, 255))

    # Active region box (the area of the frame that maps to the full screen)
    x1, x2 = int(cfg.margin_x * w), int((1 - cfg.margin_x) * w)
    y1, y2 = int(cfg.margin_top * h), int((1 - cfg.margin_bottom) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

    # Mode banner
    cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.putText(frame, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.0f} FPS", (w - 110, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    if ratio is not None:
        cv2.putText(frame, f"pinch {ratio:.2f}", (w - 230, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1, cv2.LINE_AA)

    # Help footer
    cv2.putText(frame, "Index=Move | Index+Middle=Scroll | Thumb+Index pinch=Click | p=Pause | q=Quit",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


# =============================================================================
# SCREEN LOCK GESTURE  (all five fingers extended and held)
# =============================================================================
def lock_screen() -> bool:
    """Lock the workstation using the best method for the current OS.
    Returns True if a lock command was issued."""
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            # Direct API call is more reliable than injecting Win+L.
            if ctypes.windll.user32.LockWorkStation():
                return True
            pyautogui.hotkey("win", "l")
            return True

        if system == "Darwin":
            # Ctrl+Cmd+Q = Lock Screen (macOS 10.13+). Needs Accessibility permission.
            pyautogui.hotkey("ctrl", "command", "q")
            return True

        # Linux / BSD: try the common session lockers in order.
        commands = [
            ["loginctl", "lock-session"],
            ["xdg-screensaver", "lock"],
            ["gnome-screensaver-command", "-l"],
            ["cinnamon-screensaver-command", "-l"],
            ["xscreensaver-command", "-lock"],
            ["dm-tool", "lock"],
            ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"],
        ]
        for cmd in commands:
            if shutil.which(cmd[0]):
                if subprocess.run(cmd, timeout=5, check=False).returncode == 0:
                    return True
    except Exception as exc:
        print(f"[lock] failed: {exc}")
        return False

    print("[lock] no supported lock command found on this system")
    return False


class ScreenLockDetector:
    """
    Detects an open palm (all 5 fingers fully extended) held continuously for
    `lock_hold_s` seconds, then locks the screen. After a lock the gesture must
    be released and a cooldown must pass before it can trigger again.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.progress = 0.0                  # 0..1 hold progress (used by the overlay)
        self.hold_s = cfg.lock_hold_s
        self._hold_start: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._armed = True                   # must release the gesture before re-arming
        self._cooldown_until = 0.0
        self._flash_until = 0.0

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def five_fingers_open(self, pts: np.ndarray) -> bool:
        c = self.cfg
        hand_size = max(self._dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)

        # Index, middle, ring, pinky: tip clearly farther from wrist than its PIP joint.
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
            if self._dist(pts[tip], pts[WRIST]) <= self._dist(pts[pip], pts[WRIST]) * c.lock_extend_margin:
                return False

        # Thumb: splayed away from the palm (far from the index knuckle) and straight.
        thumb_out = self._dist(pts[THUMB_TIP], pts[INDEX_MCP]) / hand_size
        thumb_straight = self._dist(pts[THUMB_TIP], pts[INDEX_MCP]) > self._dist(pts[THUMB_IP], pts[INDEX_MCP]) * 1.1
        return thumb_out >= c.lock_thumb_ratio and thumb_straight

    def reset(self) -> None:
        """Call when the hand is lost or control is paused."""
        self._hold_start = None
        self._lost_since = None
        self.progress = 0.0
        self._armed = True

    def update(self, pts: np.ndarray, now: float) -> str:
        """Returns "IDLE", "HOLDING" (countdown running) or "LOCKED" (just locked)."""
        c = self.cfg

        # Post-lock cooldown
        if now < self._cooldown_until:
            self._hold_start = None
            self.progress = 0.0
            return "LOCKED" if now < self._flash_until else "IDLE"

        detected = self.five_fingers_open(pts)

        # After a lock, the open palm must be released before it can arm again.
        if not self._armed:
            if not detected:
                self._armed = True
            return "IDLE"

        if detected:
            self._lost_since = None
            if self._hold_start is None:
                self._hold_start = now
            elapsed = now - self._hold_start
            self.progress = min(elapsed / c.lock_hold_s, 1.0)

            if elapsed >= c.lock_hold_s:
                ok = lock_screen()
                self._hold_start = None
                self.progress = 0.0
                self._armed = False
                self._cooldown_until = now + c.lock_cooldown_s
                if ok:
                    self._flash_until = now + 3.0
                    return "LOCKED"
                return "IDLE"
            return "HOLDING"

        # Gesture not detected: tolerate a brief tracking dropout, then cancel.
        if self._hold_start is not None:
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since <= c.lock_grace_s:
                return "HOLDING"
            self._hold_start = None
            self._lost_since = None
            self.progress = 0.0
        return "IDLE"


def draw_lock_overlay(frame: np.ndarray, detector: ScreenLockDetector) -> None:
    """Big centred warning + progress bar while the lock countdown is running."""
    if detector.progress <= 0:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "LOCKING SCREEN..."
    (tw, _), _ = cv2.getTextSize(text, font, 1.0, 3)
    x, y = (w - tw) // 2, h // 2

    cv2.putText(frame, text, (x, y), font, 1.0, (0, 0, 0), 6, cv2.LINE_AA)       # outline
    cv2.putText(frame, text, (x, y), font, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

    bar_w = int(w * 0.6)
    bx, by = (w - bar_w) // 2, y + 18
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * detector.progress), by + 16), (0, 0, 255), -1)

    remaining = max(detector.hold_s * (1.0 - detector.progress), 0.0)
    cv2.putText(frame, f"Hold... {remaining:.1f}s  (lower your hand to cancel)", (bx, by + 40),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    cfg = Config()

    # PyAutoGUI settings: no artificial delays. FAILSAFE is disabled because
    # a hand-driven cursor can legitimately reach screen corners; use 'q' to quit.
    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.FAILSAFE = False

    screen_w, screen_h = pyautogui.size()
    controller = GestureController(cfg, screen_w, screen_h)
    lock_detector = ScreenLockDetector(cfg)

    model_path = ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=cfg.detection_confidence,
        min_hand_presence_confidence=cfg.presence_confidence,
        min_tracking_confidence=cfg.tracking_confidence,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    stream = VideoStream(cfg).start()

    window = "Gesture Mouse"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    paused = False
    last_id = -1
    last_ts_ms = -1
    fps = 0.0
    prev_t = time.perf_counter()

    print("Gesture Mouse running. Press 'q' in the preview window to quit, 'p' to pause.")

    try:
        while True:
            frame_id, frame = stream.read()
            if frame is None or frame_id == last_id:
                # No new frame yet; avoid busy-spinning but stay responsive to keys.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.001)
                continue
            last_id = frame_id

            frame = cv2.flip(frame, 1)                       # mirror view
            h, w = frame.shape[:2]

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps must be strictly increasing in VIDEO mode.
            ts_ms = int(time.perf_counter() * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            results = landmarker.detect_for_video(mp_image, ts_ms)

            label = "NO HAND"
            ratio: Optional[float] = None

            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                norm = np.array([(lm.x, lm.y) for lm in hand], dtype=np.float32)
                pts = norm * np.array([w, h], dtype=np.float32)

                draw_hand(frame, pts)

                if paused:
                    label = "PAUSED"
                else:
                    label = controller.update(pts, norm, time.perf_counter())
                ratio = controller.pinch_ratio

                # --- Screen lock gesture (five fingers open, held) ---
                if paused:
                    lock_detector.reset()
                else:
                    lock_state = lock_detector.update(pts, time.perf_counter())
                    if lock_state == "HOLDING":
                        label = "LOCKING SCREEN..."
                    elif lock_state == "LOCKED":
                        label = "SCREEN LOCKED"

                # Highlight the pointer (index tip) and thumb tip
                ix, iy = int(pts[INDEX_TIP][0]), int(pts[INDEX_TIP][1])
                tx, ty = int(pts[THUMB_TIP][0]), int(pts[THUMB_TIP][1])
                cv2.circle(frame, (ix, iy), 9, COLORS.get(label, (255, 255, 255)), cv2.FILLED)
                cv2.circle(frame, (tx, ty), 7, (255, 255, 0), cv2.FILLED)
            else:
                controller.reset()
                lock_detector.reset()
                if paused:
                    label = "PAUSED"

            # FPS (smoothed)
            now = time.perf_counter()
            inst = 1.0 / max(now - prev_t, 1e-6)
            fps = inst if fps == 0 else fps * 0.9 + inst * 0.1
            prev_t = now

            draw_overlay(frame, label, fps, cfg, ratio)
            draw_lock_overlay(frame, lock_detector)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                controller.reset()
                lock_detector.reset()

    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        stream.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
