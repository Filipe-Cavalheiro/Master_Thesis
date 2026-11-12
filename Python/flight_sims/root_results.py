"""Copy simulation results into the repo-root Results/ folder as well.

The sims keep saving to their own local Results/ (and .pkl) as before. At the
end of the run, every file in the watched paths that was created or changed
during the run is also copied to

    <repo>/Results/<sim name>_<YYYYmmdd_HHMMSS>/      (default), or
    <dest>/                                          (if dest is given)

The run stamp is fixed when this module is imported, so all files of one run
end up in the same folder. Files that did not change are not copied.

Usage (in a sim entry point):

    import root_results
    root_results.mirror_on_exit("V16_generic_mesh", LOCAL_RESULTS, PKL_FILE)
"""
import atexit
import shutil
from datetime import datetime
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
ROOT_RESULTS = REPO / "Results"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file()]
    return []


def _snapshot(paths):
    return {f: f.stat().st_mtime_ns for p in paths for f in _files(p)}


def run_folder(name):
    """Root Results/ folder for a sim run without a flight log."""
    return ROOT_RESULTS / f"{name}_{RUN_STAMP}"


def mirror_on_exit(name, *paths, dest=None):
    """At exit, copy files under `paths` that are new or changed since now.

    paths: local result files or folders (folders are scanned recursively;
           they need not exist yet). The path relative to a watched folder is
           kept in the copy.
    dest:  root folder to copy into (default: run_folder(name)).
    """
    paths = [Path(p).resolve() for p in paths]
    before = _snapshot(paths)

    def _mirror():
        out = Path(dest) if dest else run_folder(name)
        copied = 0
        for p in paths:
            for f in _files(p):
                if before.get(f) == f.stat().st_mtime_ns:
                    continue
                rel = f.relative_to(p) if p.is_dir() else Path(f.name)
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied += 1
        if copied:
            print(f"[results] copied {copied} file(s) to {out}")

    atexit.register(_mirror)
