import time

from src.interpreter import BrainfuckInterpreter

def _format_error(interp: BrainfuckInterpreter, exc: Exception):
    pc = getattr(exc, "pc", None)
    near = getattr(exc, "near", "") or ""
    kind = getattr(exc, "kind", None)
    blocked_reason = getattr(exc, "blocked_reason", None)
    if pc is None:
        pc = getattr(interp.state, "pc", None)
    if not near:
        code = getattr(interp.state, "code", "") or ""
        if isinstance(pc, int) and 0 <= pc <= len(code):
            start = max(0, pc - 12)
            end = min(len(code), pc + 12)
            near = code[start:end]
    return {
        "type": "error",
        "error": str(exc),
        "error_type": exc.__class__.__name__,
        "kind": kind,
        "pc": pc,
        "blocked_reason": blocked_reason if blocked_reason is not None else getattr(interp.state, "blocked_reason", None),
        "near": near,
    }


def run_bf_process(code: str, out_queue, ctrl_conn):
    interp = BrainfuckInterpreter()
    tape_window = 31
    last_tape_emit = 0.0

    def _emit_tape(force=False):
        nonlocal last_tape_emit
        now = time.time()
        if (not force) and (now - last_tape_emit) < 0.05:
            return
        dp = int(getattr(interp.state, "dp", 0))
        mem = getattr(interp.state, "memory", []) or []
        half = tape_window // 2
        start = max(0, dp - half)
        end = min(len(mem), start + tape_window)
        start = max(0, end - tape_window)
        cells = list(mem[start:end])
        out_queue.put(
            {
                "type": "tape",
                "dp": dp,
                "offset": start,
                "cells": cells,
            }
        )
        last_tape_emit = now

    try:
        interp.execute(code or "", max_steps=5000)
        interp.validate_code()
        out_queue.put(
            {
                "type": "status",
                "pc": interp.state.pc,
                "blocked_reason": interp.state.blocked_reason,
                "hit_step_limit": interp.state.hit_step_limit,
            }
        )
        _emit_tape(force=True)
        out_queue.put({"type": "output", "output": interp.state.output})
    except Exception as e:
        out_queue.put(_format_error(interp, e))
        out_queue.put({"type": "finished"})
        return

    while True:
        if ctrl_conn is not None:
            try:
                if ctrl_conn.poll():
                    msg = ctrl_conn.recv()
                    if msg == "STOP":
                        out_queue.put({"type": "finished", "stopped": True})
                        return
            except (OSError, EOFError):
                ctrl_conn = None

        if interp.state.pc >= len(interp.state.code):
            out_queue.put({"type": "finished"})
            return

        try:
            interp.execute("", max_steps=5000)
        except Exception as e:
            out_queue.put(_format_error(interp, e))
            out_queue.put({"type": "finished"})
            return

        out_queue.put(
            {
                "type": "status",
                "pc": interp.state.pc,
                "blocked_reason": interp.state.blocked_reason,
                "hit_step_limit": interp.state.hit_step_limit,
            }
        )
        _emit_tape()
        out_queue.put({"type": "output", "output": interp.state.output})

        if interp.state.blocked_reason in ("awaiting_input", "awaiting_closing_bracket"):
            out_queue.put({"type": "finished"})
            return

        time.sleep(0.01)
