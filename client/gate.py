"""11-18: the instruction gate of the real-robot client (user's hard requirement: without an instruction the robot does not
move; it moves only when an instruction is given).

Three states. The student's output reaches the robot in RUNNING only.
  IDLE       nothing from the student is forwarded; the client keeps sending the HOLD command (stand still, hands open).
  RUNNING    entered ONLY by submit(<text>) with a text that, after lower-casing and collapsing spaces, equals one of the
             allowed sentences (the sentences the student was trained on; anything else is refused, nothing is guessed).
             Left when the episode time is up, on stop(), or when the sensors are stale (camera or robot state too old).
  RETURNING  the client blends from the last command back to HOLD over `blend_s`, then IDLE.
One instruction = one episode: the instruction is cleared when the episode ends; the next motion needs a new submit().
An empty text is never an instruction. submit() while not IDLE is refused (stop first).
Pure Python, no robot or model imports; `python gate.py` runs its self-test."""
from __future__ import annotations

IDLE, RUNNING, RETURNING = "IDLE", "RUNNING", "RETURNING"
ALLOWED = ("put the orange on the plate", "put the orange back on the table")


def normalise(text: str) -> str:
    return " ".join(str(text).strip().lower().rstrip(".!").split())


class InstructionGate:
    def __init__(self, allowed=ALLOWED, episode_s: float = 14.0, blend_s: float = 1.5):
        self.allowed = {normalise(a): a for a in allowed}
        self.episode_s, self.blend_s = float(episode_s), float(blend_s)
        self.state, self.task, self.t0, self.reason = IDLE, None, 0.0, "start-up"
        self.log = []  # (time, event) for the operator

    def _go(self, state: str, now: float, why: str) -> None:
        self.state, self.t0, self.reason = state, now, why
        if state == IDLE:
            self.task = None
        self.log.append((now, f"{state}: {why}"))

    def submit(self, text: str, now: float):
        """Operator typed an instruction. -> (accepted, message)."""
        key = normalise(text)
        if not key:
            return False, "empty instruction: not moving"
        if self.state != IDLE:
            return False, f"busy ({self.state}): stop first, then give the instruction again"
        if key not in self.allowed:
            return False, f"unknown instruction '{text}': not moving. Allowed: " + " | ".join(self.allowed.values())
        self.task = self.allowed[key]; self._go(RUNNING, now, f"instruction '{self.task}'")
        return True, f"running '{self.task}' for at most {self.episode_s:.0f} s"

    def stop(self, now: float, why: str = "operator stop") -> None:
        if self.state == RUNNING:
            self._go(RETURNING, now, why)

    def tick(self, now: float, sensors_ok: bool = True) -> str:
        """Call once per control tick; returns the state for this tick."""
        if self.state == RUNNING:
            if not sensors_ok:
                self._go(RETURNING, now, "camera or robot state is stale")
            elif now - self.t0 >= self.episode_s:
                self._go(RETURNING, now, "episode time is up")
        elif self.state == RETURNING and now - self.t0 >= self.blend_s:
            self._go(IDLE, now, "back at hold; waiting for an instruction")
        return self.state

    @property
    def forward(self) -> bool:
        """True only while the student's actions may reach the robot."""
        return self.state == RUNNING

    def blend_weight(self, now: float) -> float:
        """In RETURNING: 0 at its start (last command) -> 1 at its end (HOLD)."""
        return 1.0 if self.state != RETURNING or self.blend_s <= 0 else min(1.0, max(0.0, (now - self.t0) / self.blend_s))


if __name__ == "__main__":
    g = InstructionGate(episode_s=2.0, blend_s=0.5)
    assert g.tick(0.0) == IDLE and not g.forward
    for bad in ("", "   ", "go", "put the orange", "put the apple on the plate", "PUT THE ORANGE ON THE PLATE and then dance"):
        ok, msg = g.submit(bad, 0.1); assert not ok and g.state == IDLE and not g.forward, bad
    ok, msg = g.submit("  Put the  orange on the plate. ", 1.0); assert ok and g.forward and g.task == ALLOWED[0]
    ok, msg = g.submit(ALLOWED[1], 1.1); assert not ok and g.task == ALLOWED[0]  # busy
    assert g.tick(2.9) == RUNNING
    assert g.tick(3.0) == RETURNING and not g.forward and abs(g.blend_weight(3.25) - 0.5) < 1e-9
    assert g.tick(3.5) == IDLE and g.task is None  # the instruction does not carry over
    assert g.tick(100.0) == IDLE and not g.forward  # nothing happens without a new instruction
    ok, _ = g.submit(ALLOWED[1], 101.0); assert ok
    assert g.tick(101.5, sensors_ok=False) == RETURNING  # stale sensors end the episode
    assert g.tick(102.1) == IDLE
    ok, _ = g.submit(ALLOWED[0], 103.0); g.stop(103.2); assert g.state == RETURNING and g.tick(103.8) == IDLE
    print("gate self-test passed:", len(g.log), "transitions")
