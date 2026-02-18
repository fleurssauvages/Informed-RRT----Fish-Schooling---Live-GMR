import time
import threading
import numpy as np

import pyspacemouse  # :contentReference[oaicite:3]{index=3}

class SpaceMouse3D:
    """
    Reads 3Dconnexion SpaceMouse and provides:
      - v_trans (3,) translation
      - v_rot   (3,) rotation (optional)
      - buttons (int bitmask or list depending on device)
    """

    def __init__(
        self,
        trans_scale=1.0,     # scale translation output
        rot_scale=1.0,       # scale rotation output
        deadzone=0.05,       # deadzone in normalized units (0..1-ish)
        rate_hz=200,
        lowpass=0.2,         # EMA smoothing (0=no smoothing, 0.2-0.4 typical)
        axis_map=(0, 1, 2),  # re-order translation axes
        axis_sign=(1, 1, 1), # flip axes if needed
    ):
        self.trans_scale = float(trans_scale)
        self.rot_scale = float(rot_scale)
        self.deadzone = float(deadzone)
        self.rate_hz = float(rate_hz)
        self.lowpass = float(lowpass)
        self.axis_map = tuple(axis_map)
        self.axis_sign = np.array(axis_sign, dtype=float)

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._v_trans = np.zeros(3, dtype=float)
        self._v_rot = np.zeros(3, dtype=float)
        self._buttons = 0

    def start(self):
        ok = pyspacemouse.open()  # :contentReference[oaicite:4]{index=4}
        if not ok:
            raise RuntimeError("Failed to open SpaceMouse. Check permissions / device connection.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            pyspacemouse.close()
        except Exception:
            pass

    def read(self):
        """Return (v_trans, v_rot, buttons)"""
        with self._lock:
            return self._v_trans.copy(), self._v_rot.copy(), self._buttons

    def _apply_deadzone(self, v):
        # v is in roughly [-1,1] after normalization
        mag = np.linalg.norm(v)
        if mag < self.deadzone:
            return np.zeros_like(v)
        # rescale so it ramps smoothly after deadzone
        return v * ((mag - self.deadzone) / (mag + 1e-12))

    def _loop(self):
        dt = 1.0 / max(self.rate_hz, 1.0)
        alpha = self.lowpass

        while self._running:
            st = pyspacemouse.read()
            if st is None:
                time.sleep(dt)
                continue

            # pyspacemouse state typically exposes x,y,z and roll,pitch,yaw plus buttons
            # Translate axes:
            raw_trans = np.array([st.x, st.y, st.z], dtype=float)
            raw_rot = np.array([st.roll, st.pitch, st.yaw], dtype=float)

            # Normalize: pyspacemouse values are usually around [-350,350] depending on device :contentReference[oaicite:5]{index=5}
            # Use 350 as a reasonable default scale:
            trans = raw_trans / 350.0
            rot = raw_rot / 350.0

            # Reorder / sign (often needed depending on your world frame)
            trans = trans[list(self.axis_map)] * self.axis_sign

            # Deadzone
            trans = self._apply_deadzone(trans)
            rot = self._apply_deadzone(rot)

            # Scale to your desired units
            trans *= self.trans_scale
            rot *= self.rot_scale

            # Smooth (EMA)
            with self._lock:
                if alpha > 0.0:
                    self._v_trans = (1 - alpha) * trans + alpha * self._v_trans
                    self._v_rot = (1 - alpha) * rot + alpha * self._v_rot
                else:
                    self._v_trans = trans
                    self._v_rot = rot

                # buttons varies by device; pyspacemouse commonly gives .buttons as int bitmask
                self._buttons = getattr(st, "buttons", 0)

            time.sleep(dt)
