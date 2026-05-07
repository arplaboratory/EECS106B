#!/usr/bin/env python3
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""HuggingFace push/pull utility for the EECS106B racing-drone repo.

Two subcommands, both default to the dataset
``ckwolfe/eecs106b-racing-drone-assets`` (override with ``--repo``):

  push   Upload a local file or directory to HF Hub.
         Each item is uploaded with its basename (or the optional `--as` /
         `--prefix` translation) into the destination repo.
  pull   Download a path or whole subtree from HF Hub into a local target dir.

The token is read from the ``HF_TOKEN`` environment variable. If it is not
set, the script falls back to whatever ``huggingface-cli login`` cached
under ``~/.cache/huggingface/token``. Refuses to write the token to the
git credential helper.

Examples
--------
    # one-shot upload of a single file under videos/
    HF_TOKEN=hf_xxx python scripts/hf_io.py push demo.mp4 --prefix videos

    # upload a directory tree, preserving structure under videos/clips/
    HF_TOKEN=hf_xxx python scripts/hf_io.py push out_dir/ --prefix videos/clips

    # rename a single file on the way up
    HF_TOKEN=hf_xxx python scripts/hf_io.py push out.mp4 --as videos/foo.mp4

    # pull just the videos/ folder back down
    python scripts/hf_io.py pull videos --to ./_videos

    # pull a single file
    python scripts/hf_io.py pull videos/demo_fig8_4k.mp4 --to .
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_REPO = "ckwolfe/eecs106b-racing-drone-assets"
DEFAULT_TYPE = "dataset"


def _login(token: str | None) -> None:
    from huggingface_hub import login

    if token:
        login(token=token, add_to_git_credential=False)


def _iter_local_files(root: Path) -> Iterable[Path]:
    """Yield every regular file beneath ``root`` (or just ``root`` if file)."""
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def _resolve_dst(local: Path, root: Path, prefix: str | None,
                 as_path: str | None) -> str:
    """Compute the path-in-repo for ``local`` relative to ``root``."""
    if as_path is not None:
        # explicit override only makes sense for a single-file push
        return as_path.lstrip("/")
    rel = local.name if local == root else str(local.relative_to(root))
    if prefix:
        prefix = prefix.strip("/")
        return f"{prefix}/{rel}"
    return rel


def cmd_push(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"[hf push] source path does not exist: {src}", file=sys.stderr)
        return 2

    if args.as_path and src.is_dir():
        print("[hf push] --as can only be used with a single file source",
              file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or args.token
    _login(token)
    api = HfApi(token=token)

    files = list(_iter_local_files(src))
    if not files:
        print(f"[hf push] nothing to upload under {src}", file=sys.stderr)
        return 1

    print(f"[hf push] {len(files)} file(s) -> {args.repo} ({args.type})",
          flush=True)
    for f in files:
        dst = _resolve_dst(f, src, args.prefix, args.as_path if src.is_file() else None)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[hf push] {f}  ({size_mb:.2f} MB)  ->  {dst}", flush=True)
        if args.dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=dst,
            repo_id=args.repo,
            repo_type=args.type,
            commit_message=args.message or f"hf_io.push: add {dst}",
        )
    if args.dry_run:
        print("[hf push] DRY RUN — nothing actually uploaded", flush=True)
    else:
        print(f"[hf push] done. Tree: "
              f"https://huggingface.co/{args.type}s/{args.repo}/tree/main",
              flush=True)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or args.token
    _login(token)
    api = HfApi(token=token)

    target = Path(args.to).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    # Decide whether `path` is a file or a directory in the repo.
    repo_files = api.list_repo_files(
        repo_id=args.repo, repo_type=args.type, token=token,
    )
    norm = args.path.strip("/")
    if norm in repo_files:
        # exact file match
        local = hf_hub_download(
            repo_id=args.repo, repo_type=args.type, filename=norm,
            local_dir=str(target), token=token,
        )
        print(f"[hf pull] downloaded {norm} -> {local}", flush=True)
        return 0

    matches = [f for f in repo_files if f == norm or f.startswith(norm + "/")]
    if not matches:
        print(f"[hf pull] no files match '{args.path}' in {args.repo}",
              file=sys.stderr)
        return 1

    print(f"[hf pull] {len(matches)} file(s) -> {target}/", flush=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type=args.type,
        allow_patterns=[norm + "/*", norm + "/**"],
        local_dir=str(target),
        token=token,
    )
    for m in matches:
        print(f"[hf pull]   {m}", flush=True)
    print(f"[hf pull] done. Local: {target}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push/pull files between local disk and HF Hub.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF repo id (default {DEFAULT_REPO}).")
    parser.add_argument("--type", default=DEFAULT_TYPE,
                        choices=("dataset", "model", "space"),
                        help=f"repo type (default {DEFAULT_TYPE}).")
    parser.add_argument("--token", default=None,
                        help="HF token. Falls back to $HF_TOKEN.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="upload a file or directory")
    p_push.add_argument("src", help="local file or directory to upload")
    p_push.add_argument("--prefix", default=None,
                        help="path-in-repo prefix to prepend to every "
                             "uploaded file (e.g. 'videos').")
    p_push.add_argument("--as", dest="as_path", default=None,
                        help="rename a single-file upload to this path "
                             "(mutually exclusive with --prefix for files).")
    p_push.add_argument("-m", "--message", default=None,
                        help="commit message used for each upload.")
    p_push.add_argument("--dry-run", action="store_true",
                        help="print what would be uploaded but do nothing.")
    p_push.set_defaults(func=cmd_push)

    p_pull = sub.add_parser("pull", help="download a path from HF Hub")
    p_pull.add_argument("path",
                        help="path in the repo: a single file (e.g. "
                             "'videos/foo.mp4') or a folder (e.g. 'videos').")
    p_pull.add_argument("--to", default=".",
                        help="local target directory (default cwd).")
    p_pull.set_defaults(func=cmd_pull)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
