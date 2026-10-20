"""Download the Qwen model collection into /root/autodl-tmp/models.

Usage:
    source /etc/network_turbo
    python scripts/download_models.py --dry-run
    python scripts/download_models.py

The script resumes partial downloads, skips already-complete directories, and
checks free disk space before starting.  It never deletes an existing model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

MODEL_ROOT = Path("/root/autodl-tmp/models")

# Ordered smallest to largest so the useful small models arrive first.
MODELS: list[tuple[str, str]] = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
    ("Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Thinking-2507"),
    ("Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    ("Qwen2.5-14B", "Qwen/Qwen2.5-14B"),
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct"),
    ("Qwen2.5-32B", "Qwen/Qwen2.5-32B"),
]


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def model_size(api, repo_id: str) -> int:
    """Estimate repo size using the file tree, falling back to model metadata."""
    total = 0
    try:
        for item in api.list_repo_tree(repo_id, recursive=True):
            total += int(getattr(item, "size", 0) or 0)
    except Exception:
        total = 0
    if total:
        return total
    info = api.model_info(repo_id, files_metadata=True)
    for sibling in info.siblings or []:
        total += int(getattr(sibling, "size", 0) or 0)
    return total


def is_complete(target: Path) -> bool:
    return (target / "config.json").is_file() and (target / ".download_complete").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(MODEL_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="local model names")
    parser.add_argument("--force", action="store_true", help="ignore free-space estimate")
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()

    # Must be set before huggingface_hub is imported in worker code.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None
    api = HfApi()

    selected = [(local, repo) for local, repo in MODELS if wanted is None or local in wanted]
    if not selected:
        raise SystemExit("No matching models")
    print(f"Model root: {root}")
    print(f"Disk free:  {human(shutil.disk_usage(root).free)}")
    print(f"Models:     {len(selected)}")

    estimates: list[tuple[str, str, Path, int, bool]] = []
    for local, repo in selected:
        target = root / local
        try:
            size = model_size(api, repo)
        except Exception as exc:
            print(f"[WARN] cannot query {repo}: {exc}")
            size = 0
        complete = is_complete(target)
        estimates.append((local, repo, target, size, complete))
        print(
            f"  {'SKIP' if complete else 'GET ':<4} {local:<28} "
            f"{human(size):>9}  {repo}"
        )

    needed = sum(size for _, _, _, size, complete in estimates if not complete)
    free = shutil.disk_usage(root).free
    print(f"Estimated additional download: {human(needed)}")
    if not args.force and needed and needed > free - 10 * 1024**3:
        raise SystemExit(
            f"Refusing: need {human(needed)}, only {human(free)} free "
            "(leave 10GB buffer; use --force to override)"
        )
    if args.dry_run:
        print("Dry run only; nothing downloaded.")
        return

    failed: list[str] = []
    for local, repo, target, _, complete in estimates:
        if complete:
            print(f"[SKIP] {local}")
            continue
        print(f"\n===== {local} <- {repo} =====", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=str(target),
                    resume_download=True,
                    max_workers=4,
                )
                (target / ".download_complete").write_text(
                    f"{repo}\n", encoding="utf-8"
                )
                print(f"[DONE] {local}", flush=True)
                break
            except Exception as exc:
                wait = min(30 * attempt, 300)
                print(
                    f"[RETRY] {local} attempt {attempt}/{args.retries} "
                    f"failed: {type(exc).__name__}: {exc}; sleeping {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        else:
            failed.append(local)
            print(f"[FAILED] {local}", flush=True)

    if failed:
        print(f"\nIncomplete models: {', '.join(failed)}", flush=True)
        raise SystemExit(1)
    print("\nAll selected models are complete.")


if __name__ == "__main__":
    main()
