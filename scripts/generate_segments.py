#!/usr/bin/env python3
"""Pre-generate trimmed audio segments for all clips in the database.

This script iterates over every clip stored in the annotations database and
creates an audio snippet that includes a configurable amount of context
around each clip. Existing snippets are reused unless --force is provided.

The generated files live inside <audio_folder>/<segment_subdir>/<audio_name>/
so they can be served by the annotation UI. For any timestamp measured in
the segment file, you can recover the original time in the full audio with:

    original_t = segment_t + segment_start_timestamp
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import textwrap
from typing import Iterable, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_backend import ClipRecord, DatabaseBackend
from src.fast_audio_annotate.config import AppConfig
from src.fast_audio_annotate.segments import (
    SegmentGenerationError,
    SegmentGenerationResult,
    generate_segment,
    locate_ffmpeg,
    remove_previous_segment,
)


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_PADDING_SECONDS = 2.0
DEFAULT_SEGMENT_SUBDIR = "segments"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-clip audio segments for all clips in the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            The script iterates over every clip stored in the annotations database and
            creates an audio snippet that includes a configurable amount of context
            around each clip. Existing snippets are reused unless --force is
            provided. All generated files live inside the <audio_folder>/<segment_subdir>
            directory so they can be served by the annotation UI.
            """
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file (default: %(default)s).",
    )
    parser.add_argument(
        "--audio-folder",
        help="Override the audio_folder from the configuration file.",
    )
    parser.add_argument(
        "--database-url",
        help="Override the database URL from the configuration file.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_SECONDS,
        help="Seconds of context to include before and after each clip (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-subdir",
        default=DEFAULT_SEGMENT_SUBDIR,
        help="Directory (relative to the audio folder) where segments are stored.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        help="Path to the ffmpeg executable (defaults to auto-detection).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate segments even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the actions without creating or updating any files.",
    )
    parser.add_argument(
        "--audio-path-filter",
        help=(
            "Optional audio_path filter. If provided, only clips whose audio_path "
            "matches this value will be processed (e.g. 'routine_60.webm')."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Número de hilos en paralelo para generar segmentos (default: núm. de CPUs).",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> AppConfig:
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    return AppConfig(**data)


@dataclass
class SegmentJob:
    """Work item representing a clip and its source audio file."""

    clip: ClipRecord
    source_path: Path


def collect_jobs(
    db: DatabaseBackend,
    audio_root: Path,
    audio_path_filter: Optional[str] = None,
) -> List[SegmentJob]:
    """Collect all jobs to process, optionally filtering by audio_path.

    If `audio_path_filter` is provided, only clips with that audio_path are fetched.
    Otherwise, all clips in the database are considered.

    Además, solo se procesan clips cuyo archivo de audio termine en .webm.
    """

    if audio_path_filter:
        clips = db.fetch_clips(audio_path_filter)
    else:
        clips = db.fetch_all_clips()

    jobs: List[SegmentJob] = []

    for clip in clips:
        source_path = audio_root / clip.audio_path

        # 🔥 Solo procesar archivos .webm
        if source_path.suffix.lower() != ".webm":
            continue

        if not source_path.exists():
            print(f"⚠️  Skipping clip {clip.id}: audio source missing ({source_path}).")
            continue

        jobs.append(SegmentJob(clip=clip, source_path=source_path))

    if audio_path_filter:
        print(f"🎯 Collected {len(jobs)} .webm clips for audio_path='{audio_path_filter}'")
    else:
        print(f"🎯 Collected {len(jobs)} .webm clips across all audio files")

    return jobs


def process_clip_segment(
    clip: ClipRecord,
    source_path: Path,
    audio_root: Path,
    segment_subdir: str,
    padding: float,
    ffmpeg_binary: str,
) -> SegmentGenerationResult:
    """
    Función que corre en un hilo: solo se encarga de llamar a generate_segment.
    No toca la base de datos.
    """
    return generate_segment(
        clip_id=clip.id,
        audio_root=audio_root,
        source_path=source_path,
        segment_dir_name=segment_subdir,
        clip_start=clip.start_timestamp,
        clip_end=clip.end_timestamp,
        padding=padding,
        ffmpeg_binary=ffmpeg_binary,
    )


def run(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # Load configuration file
    config_path = Path(args.config)
    app_config = load_config(config_path)

    # Allow overriding config through CLI flags
    if args.audio_folder:
        app_config.audio_folder = args.audio_folder
    if args.database_url:
        app_config.database_url = args.database_url

    if app_config.audio_folder.startswith(("http://", "https://")):
        print("Error: audio_folder points to a remote location. Download the audio locally first.")
        return 1

    audio_root = Path(app_config.audio_folder)
    if not audio_root.is_absolute():
        # Always resolve relative paths from the project root, not from current working directory
        audio_root = (ROOT_DIR / audio_root).resolve()

    sqlite_path = audio_root / "annotations.db"

    # Resolve database URL from config, environment variables, or default to SQLite
    database_url = (
        app_config.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("NEON_DATABASE_URL")
    )

    db_backend = DatabaseBackend(sqlite_path, database_url)

    ffmpeg_binary = args.ffmpeg_binary or locate_ffmpeg()
    if ffmpeg_binary is None:
        print("Error: ffmpeg binary not found. Provide --ffmpeg-binary or install ffmpeg.")
        return 1

    jobs = collect_jobs(db_backend, audio_root, args.audio_path_filter)
    if not jobs:
        print("No clips found to process. Nothing to do.")
        return 0

    generated = 0
    skipped = 0
    failures = 0

    # 1) Filtramos qué trabajos realmente necesitan generarse
    pending_jobs: List[SegmentJob] = []

    for job in jobs:
        clip = job.clip
        existing_path = clip.segment_path

        # Reutilizar segmento existente si es válido y no hay --force
        if (
            not args.force
            and existing_path
            and (audio_root / existing_path).exists()
            and clip.segment_start_timestamp is not None
            and clip.segment_end_timestamp is not None
        ):
            skipped += 1
            continue

        # Sólo mostramos lo que haríamos en modo dry-run
        if args.dry_run:
            print(
                f"[DRY RUN] Would generate segment for clip {clip.id} "
                f"({clip.audio_path}: {clip.start_timestamp:.2f}s–{clip.end_timestamp:.2f}s)."
            )
            skipped += 1
            continue

        # Si forzamos, borramos el segmento previo (si lo hay)
        if args.force and existing_path:
            remove_previous_segment(audio_root, existing_path)

        pending_jobs.append(job)

    if args.dry_run:
        # Ya hemos impreso todo en el bucle anterior
        print(
            f"[DRY RUN] Total clips: {len(jobs)}, "
            f"{skipped} reutilizados/omitidos, {len(pending_jobs)} que se generarían."
        )
        return 0

    if not pending_jobs:
        print(
            f"Processed {len(jobs)} clips: 0 generated, {skipped} reused/skipped, 0 failed."
        )
        return 0

    # 2) Paralelizamos solo la generación de segmentos
    max_workers = max(1, args.workers)
    print(f"🚀 Generating {len(pending_jobs)} segments with {max_workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {
            executor.submit(
                process_clip_segment,
                job.clip,
                job.source_path,
                audio_root,
                args.segment_subdir,
                args.padding,
                ffmpeg_binary,
            ): job.clip
            for job in pending_jobs
        }

        for future in as_completed(future_to_clip):
            clip = future_to_clip[future]
            try:
                result: SegmentGenerationResult = future.result()
            except SegmentGenerationError as exc:
                print(f"❌ Failed to generate segment for clip {clip.id}: {exc}")
                failures += 1
                continue
            except Exception as exc:
                # Por si acaso algo más raro explota
                print(f"💥 Unexpected error for clip {clip.id}: {exc}")
                failures += 1
                continue

            # 3) Guardamos el resultado en la base de datos (solo en el hilo principal)
            db_backend.upsert_clip_segment(
                clip.id,
                result.relative_path,
                result.start_timestamp,
                result.end_timestamp,
            )
            generated += 1

    total = len(jobs)
    print(
        f"Processed {total} clips: {generated} generated, {skipped} reused/skipped, {failures} failed."
    )

    return 0 if failures == 0 else 2


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
