"""Metadata of a simulation run, saved as the real flights save theirs.

A real flight writes <log>_meta.json next to its trajectory log (the flight
code, the start time, the scene, the drones and every setting in use). A
simulation writes the same file next to its own log (sim.csv, <log>_sim.csv
or the .pkl of the run):

    <sim log>_meta.json
        flight_code, kind ("simulation"), mode ("nominal", "replay", "dry_run"),
        replayed_log, start, command, git (commit, dirty), seed,
        settings (the Param of the simulation, or the flight-code constants),
        drones (initial position of every drone), extra

The seed is the one of seed_rng(), called by every simulation entry point
before it simulates, so a run can be repeated with --seed <seed>.

Usage (in a sim entry point):

    import sim_meta
    sim_meta.seed_rng(args.seed)
    ...
    sim_meta.write_sim_meta(sim_csv, "8traj", param=param, drones=sim_meta.drones_from_sim_csv(sim_df))
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())

SEED = None          # seed of this run (seed_rng)
START = time.strftime("%Y-%m-%d %H:%M:%S")

MAX_ARRAY = 64       # arrays larger than this are stored as their shape only
MAX_DEPTH = 4


def seed_rng(seed=None):
    """Seed numpy and random with `seed` (None: a new random seed) and remember it for the metadata."""
    global SEED
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    SEED = int(seed)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Random seed {SEED}")
    return SEED


def git_state():
    """Commit of the code that ran, and whether the working tree had uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}
    except OSError:
        return {"commit": None, "dirty": None}


def to_jsonable(obj, depth=0):
    """Settings object -> JSON (public attributes; callables skipped; large arrays as their shape)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= MAX_ARRAY else {"array_shape": list(obj.shape)}
    if depth >= MAX_DEPTH:
        return repr(obj)[:200]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, depth + 1) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_ARRAY:
            return {"list_length": len(obj)}
        return [to_jsonable(v, depth + 1) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v, depth + 1) for k, v in vars(obj).items()
                if not k.startswith("_") and not callable(v)}
    return repr(obj)[:200]


def module_settings(module):
    """The upper-case constants of a flight-code settings module (e.g. Structs.py of a flight code)."""
    return {k: to_jsonable(v) for k, v in vars(module).items()
            if k.isupper() and not k.startswith("_") and not callable(v)}


def drones_from_sim_csv(sim_df, pos_cols=("opti_x", "opti_y", "opti_z")):
    """Initial position of every drone of a simulated log (first row of each id)."""
    drones = {}
    for did, g in sim_df.groupby("id", sort=False):
        drones[str(did)] = {"initial_position": [float(v) for v in g.iloc[0][list(pos_cols)]]}
    return drones


def write_sim_meta(log_path, flight_code, *, param=None, settings=None, mode="nominal", replayed_log=None,
                   drones=None, extra=None):
    """Write <log_path stem>_meta.json next to the simulation log and return its path."""
    log_path = Path(log_path)
    meta = {
        "flight_code": flight_code,
        "kind": "simulation",
        "mode": mode,
        "replayed_log": None if replayed_log is None else str(replayed_log),
        "start": START,
        "end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "git": git_state(),
        "seed": SEED,
        "settings": to_jsonable(param) if param is not None else to_jsonable(settings),
        "drones": drones or {},
        "extra": to_jsonable(extra or {}),
    }
    path = log_path.with_name(log_path.stem + "_meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=lambda o: to_jsonable(o)))
    print(f"Simulation metadata saved to {path}")
    return path
