"""FastHTML Audio Annotation Tool - Crowdsourced clip review interface."""
from fasthtml.common import *
from starlette.responses import FileResponse, Response
from pathlib import Path
from typing import Optional
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files, load_audio_metadata_from_file
from fast_audio_annotate.segments import compute_segment_window

from db_backend import ClipRecord, DatabaseBackend

config: AppConfig = parse_app_config()
APP_BRAND = "¡shiaaa!"

BRAND_ORANGE = "#f7931e"
BRAND_ORANGE_DARK = "#c86400"
BRAND_ORANGE_LIGHT = "#fff4e8"
SITE_DOMAIN = os.environ.get("SITE_DOMAIN", "shiaaa.cl")
SITE_URL = os.environ.get("SITE_URL") or f"https://{SITE_DOMAIN}"
SOCIAL_DESCRIPTION = (
    "Ayuda a mejorar los modelos de voz en español chileno corrigiendo audios reales."
)
SOCIAL_HEADERS = Socials(
    title=f"{APP_BRAND} · Corrige audios reales de humor chileno",
    site_name=SITE_DOMAIN,
    description=SOCIAL_DESCRIPTION,
    image="/shiaaa3.png",
    url=SITE_URL,
    card="summary_large_image",
)

# Database setup
database_url = (
    config.database_url
    or os.environ.get("DATABASE_URL")
    or os.environ.get("NEON_DATABASE_URL")
)
db_backend = DatabaseBackend(config.audio_path / "annotations.db", database_url)

load_audio_metadata_from_file(config.audio_path, db_backend, config.metadata_filename)

# Runtime helpers for audio segments
SEGMENT_PADDING_SECONDS = 2.0
SEGMENT_SUBDIR_NAME = "segments"
AUDIO_FOLDER_IS_REMOTE = config.audio_folder.startswith(("http://", "https://"))


# Initialize FastHTML app with custom styles and scripts
app, rt = fast_app(
    hdrs=(
        Link(rel='stylesheet', href='/styles.css'),
        # WaveSurfer.js and plugins
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.min.js'),
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.min.js'),
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.min.js'),
        *SOCIAL_HEADERS,
    ),
    pico=False,
    debug=True
)

fasthtml_serve = serve


# Helper functions
def get_username(contributor_name: str = "") -> str:
    """Return the username for audit purposes, preferring contributor name."""
    if contributor_name and contributor_name.strip():
        return contributor_name.strip()
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'


def get_contributor_stats() -> dict:
    """Get statistics about contributors."""
    try:
        stats = db_backend.get_contributor_stats()
        return stats
    except Exception as e:
        print(f"Error getting contributor stats: {e}")
        return {"total_contributors": 0, "total_contributions": 0, "contributors": []}


def get_audio_metadata(audio_path: Optional[str]) -> Optional[dict]:
    """Fetch metadata for an audio file from the database."""

    if not audio_path:
        return None

    record = db_backend.fetch_audio_metadata(str(audio_path))
    return record.metadata if record else None


def select_random_clip() -> Optional[ClipRecord]:
    """Pick a random clip that still needs human review and has segment file available."""
    import random

    all_clips = db_backend.fetch_all_clips()

    def has_valid_segment(clip: ClipRecord) -> bool:
        """Check if clip has a valid segment file available."""
        if not clip.segment_path:
            return False
        
        if AUDIO_FOLDER_IS_REMOTE:
            # For remote storage, trust that the path exists if it's set
            return True
        
        # For local storage, check if file actually exists
        audio_root = config.audio_path
        segment_path = audio_root / clip.segment_path
        return segment_path.exists()

    unreviewed_clips_with_segments = [
        clip for clip in all_clips
        if not clip.human_reviewed
        and not clip.marked
        and has_valid_segment(clip)
    ]

    if not unreviewed_clips_with_segments:
        print("🎯 No more clips with segments need review")
        return None

    clip = random.choice(unreviewed_clips_with_segments)
    print(f"🎯 Selected clip {clip.id} from {clip.audio_path} ({len(unreviewed_clips_with_segments)} clips with segments remaining)")
    return ensure_clip_segment(clip)


def get_clip(clip_id: Optional[str]) -> Optional[ClipRecord]:
    """Return a clip by id, or ``None`` if unavailable."""

    if not clip_id:
        return None
    try:
        clip = db_backend.get_clip(int(clip_id))
        return ensure_clip_segment(clip)
    except (TypeError, ValueError):
        return None


def compute_display_window(
    start: float,
    end: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
) -> tuple[float, float]:
    """Return the window that should be visible in the waveform.

    In this app we always want to show the *full segment* and nothing else,
    so the display window is defined by the segment range:
    - lower_bound: segment_start_timestamp
    - upper_bound: segment_end_timestamp
    """

    display_start = lower_bound
    display_end = upper_bound if upper_bound is not None else end
    if display_end <= display_start:
        display_end = display_start + max(end - start, 0.5)
    return display_start, display_end


def parse_relative_offsets(
    start_value: str,
    end_value: str,
) -> tuple[Optional[float], Optional[float]]:
    """Parse user-provided relative offsets, returning ``None`` on failure."""

    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        return None, None

    if start < 0 or end <= start:
        return None, None

    return start, end


def ensure_clip_segment(clip: Optional[ClipRecord]) -> Optional[ClipRecord]:
    """Attach stored segment metadata to ``clip`` if available.

    If segment metadata is missing or stale, synthesize a fallback segment window
    around the clip using the original audio.

    Note: the UI will *only* play the segment file when available. If the
    segment file is missing, we still keep the timestamps (for alignment),
    but we do not fall back to playing the full routine.
    """

    if clip is None:
        return None

    # Ensure we have a segment window
    if clip.segment_start_timestamp is None or clip.segment_end_timestamp is None:
        fallback_start, fallback_end = compute_segment_window(
            clip.start_timestamp,
            clip.end_timestamp,
            padding=SEGMENT_PADDING_SECONDS,
            lower_bound=0.0,
        )
        clip.segment_start_timestamp = fallback_start
        clip.segment_end_timestamp = fallback_end

    if not clip.segment_path:
        return clip

    if AUDIO_FOLDER_IS_REMOTE:
        # We cannot reliably check existence for remote storage; trust the path.
        return clip

    audio_root = config.audio_path
    segment_path = audio_root / clip.segment_path
    if segment_path.exists():
        return clip

    # If the stored segment path is stale, drop it.
    clip.segment_path = None
    return clip


def render_clip_editor(clip: ClipRecord) -> Div:
    """Render the editor for a single clip."""

    clip = ensure_clip_segment(clip)
    metadata = get_audio_metadata(clip.audio_path)
    clip_id_value = str(clip.id)
    # If there is no segment file, we cannot show the waveform for this clip.
    if not clip.segment_path:
        return Div(
            H2("No encontramos el segmento", style="color: #dc3545; margin-bottom: 12px;"),
            P(
                f"Este clip (ID {clip.id}) no tiene el archivo recortado listo.",
                style="margin-bottom: 8px;"
            ),
            P(
                "Espera un rato y abre la página nuevamente más tarde.",
                style="color: #6c757d;"
            ),
            id="main-content",
            style="max-width: 640px; margin: 40px auto; background: white; padding: 24px; border-radius: 10px; border: 1px solid #f1f3f5;"
        )

    # Segment offsets in the original full audio
    segment_offset = clip.segment_start_timestamp or 0.0
    segment_end = clip.segment_end_timestamp

    # Display window = full segment range (segment_start -> segment_end)
    padded_start, padded_end = compute_display_window(
        clip.start_timestamp,
        clip.end_timestamp,
        lower_bound=segment_offset,
        upper_bound=segment_end,
    )

    # Relative clip boundaries (in seconds inside the segment)
    if clip.relative_start_offset is not None:
        relative_clip_start = max(0.0, clip.relative_start_offset)
    else:
        relative_clip_start = max(0.0, clip.start_timestamp - segment_offset)

    if clip.relative_end_offset is not None:
        relative_clip_end = max(relative_clip_start, clip.relative_end_offset)
    else:
        relative_clip_end = max(relative_clip_start, clip.end_timestamp - segment_offset)

    # Relative display boundaries (full segment)
    relative_display_start = max(0.0, padded_start - segment_offset)
    relative_display_end = max(relative_display_start, padded_end - segment_offset)

    segment_duration = None
    if segment_end is not None:
        segment_duration = max(0.0, segment_end - segment_offset)

    # IMPORTANT: always play the pre-generated segment, never the full routine.
    audio_path_for_playback = clip.segment_path
    duration = clip.end_timestamp - clip.start_timestamp

    intro = Div(
        H2(
            "Ayúdanos a transcribir este audio en español chileno",
            style=f"margin-bottom: 8px; color: {BRAND_ORANGE};",
        ),
        P(
            "Estamos construyendo una base de datos de transcripciones de audio en español chileno "
            "para poder entrenar un modelo de IA que pueda entender cómo hablamos los chilenos. ",
            Strong("¿Qué tienes que hacer? "),
            "Primero, ",
            Strong("escucha el audio completo"),
            ". Después ",
            Strong("corrige el texto"),
            " si tiene errores y ",
            Strong("ajusta los tiempos de inicio y fin del clip"),
            " si ves que el recorte quedó corrido. No agregues cosas que no se escuchan. ",
            Strong("Si el audio no se escucha bien o no tiene texto a transcribir, usa el botón \"Reportar audio\" para avisarnos."),
            " ¿Tienes dudas? ",
            A(
                "Lee las preguntas frecuentes",
                href="#",
                hx_get=f"/tab/faq?clip_id={clip_id_value}",
                hx_target="#tab-shell",
                hx_swap="outerHTML",
                hx_indicator="#tab-loading",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;",
            ),
            ".",
            style="color: #495057; margin-bottom: 0; font-size: 0.95rem;",
        ),
        style=(
            f"margin-bottom: 18px; background: {BRAND_ORANGE_LIGHT}; padding: 16px; "
            "border-radius: 10px;"
        ),
    )

    metadata_line = None
    if metadata:
        artist = metadata.get("artist")
        event = metadata.get("event")
        year = metadata.get("year")
        if artist and event and year:
            metadata_line = P(
                "Audio extraído del show de ",
                Strong(str(artist)),
                ", en el ",
                Strong(str(event)),
                " de ",
                Strong(str(year)),
                ".",
                style="margin: 4px 0 0; color: #6c757d;",
            )

    clip_info_children = [
        Div(
            Span(
                f"Clip #{clip.id}",
                style=f"font-weight: 600; color: {BRAND_ORANGE};",
            ),
            Span(f"· {duration:.1f}s"),
            style="display: flex; gap: 8px; align-items: center; color: #495057;",
        )
    ]
    if metadata_line:
        clip_info_children.append(metadata_line)

    clip_info = Div(
        *clip_info_children,
        style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px;",
    )

    form_inputs = Div(
        Input(type="hidden", name="clip_id", value=str(clip.id)),
        Div(
            Div(
                Label(
                    "Inicio (segundos)",
                    style="display: block; margin-bottom: 4px; font-weight: 600;",
                ),
                Input(
                    type="hidden",
                    name="start_time",
                    value=f"{clip.start_timestamp:.6f}",
                    id="start-time-hidden",
                ),
                Input(
                    type="hidden",
                    name="start_time_relative",
                    value=f"{relative_clip_start:.6f}",
                    id="start-time-relative-hidden",
                ),
                Input(
                    type="number",
                    value=f"{relative_clip_start:.2f}",
                    step="0.01",
                    min="0",
                    id="start-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            Div(
                Label(
                    "Fin (segundos)",
                    style="display: block; margin-bottom: 4px; font-weight: 600;",
                ),
                Input(
                    type="hidden",
                    name="end_time",
                    value=f"{clip.end_timestamp:.6f}",
                    id="end-time-hidden",
                ),
                Input(
                    type="hidden",
                    name="end_time_relative",
                    value=f"{relative_clip_end:.6f}",
                    id="end-time-relative-hidden",
                ),
                Input(
                    type="number",
                    value=f"{relative_clip_end:.2f}",
                    step="0.01",
                    min="0",
                    id="end-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;"
        ),
        Div(
            Label("Texto (Escribe acá exactamente lo que se escucha)", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px;"),
            Textarea(
                clip.text or "",
                name="transcription",
                id="transcription-input",
                rows="6",
                placeholder="Escribe acá exactamente lo que se escucha, sin adornos ni cosas inventadas.",
                style="width: 100%; padding: 12px; border: 1px solid #ced4da; border-radius: 6px; font-size: 15px; resize: vertical;",
            ),
            style="margin-bottom: 16px;"
        ),
        Div(
            Label("Tu nombre (opcional)", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px; color: #495057;"),
            Input(
                value=clip.username if hasattr(clip, 'username') and clip.username and clip.username != 'Alonso' else "",
                name="contributor_name",
                id="contributor-name-input",
                placeholder="Escribe cómo quieres aparecer en el ranking...",
                style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
            ),
            Div(
                "💡 Así te podemos agradecer y sumar tus aportes.",
                style="font-size: 12px; color: #6c757d; margin-top: 4px; font-style: italic;"
            ),
            style="margin-bottom: 20px;"
        ),
        id="clip-form"
    )

    actions = Div(
        Button(
            "💾 Guardar anotación",
            cls="complete-btn",
            hx_post="/complete_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-complete",
            style=(
                f"padding: 12px 18px; border-radius: 6px; background: {BRAND_ORANGE}; "
                "color: white; border: none; font-size: 15px; cursor: pointer;"
            ),
        ),
        Button(
            "➡️ Siguiente clip",
            cls="next-btn",
            hx_post="/next_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-next",
            style=(
                "padding: 12px 18px; border-radius: 6px; background: #ffe1be; color: #6a3b00; "
                "border: 1px solid #f7a74d; font-size: 15px; cursor: pointer;"
            ),
        ),
        Button(
            "🚩 Reportar audio",
            cls="flag-btn",
            hx_post="/flag_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_confirm="¿Marcamos este clip porque tiene algo raro?",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-flag",
            style="padding: 12px 18px; border-radius: 6px; background: #dc3545; color: white; border: none; font-size: 15px; cursor: pointer;"
        ),
        Div(
            "🔄 Cargando otro clip...",
            id="loading-next",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; font-size: 14px;"
        ),
        Div(
            "✅ Guardando...",
            id="loading-complete",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; color: #0c5460; font-size: 14px;"
        ),
        Div(
            "🚩 Marcando...",
            id="loading-flag",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 14px;"
        ),
        style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;"
    )

    waveform_controls = Div(
        Div(
            Button("▶ Reproducir", id="play-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            Button("⏸ Pausa", id="pause-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            Button("⏹ Volver", id="stop-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            style="display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px;"
        ),
        Div(
            Label("Velocidad:", style="font-weight: 600;"),
            Select(
                Option("0.75x", value="0.75"),
                Option("1x", value="1", selected=True),
                Option("1.25x", value="1.25"),
                Option("1.5x", value="1.5"),
                Option("2x", value="2"),
                id="speed-select",
                style="padding: 8px; border-radius: 6px; border: 1px solid #ced4da; min-width: 90px;"
            ),
            style="display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap;"
        ),
        style="display: flex; flex-direction: column; align-items: stretch;"
    )

    waveform = Div(
        Div(
            Div(
                "Tiempo actual: ",
                Span(
                    "0.00",
                    id="current-time",
                    style=f"font-weight: bold; color: {BRAND_ORANGE};",
                ),
                " s",
                style="font-size: 16px; margin-bottom: 12px;"
            ),
            Div(
                "Atajos: [",
                Span("Q", style="font-weight: 600; color: #198754;"),
                "] inicio • [",
                Span("W", style="font-weight: 600; color: #dc3545;"),
                "] fin • [",
                Span(
                    "Espacio",
                    style=f"font-weight: 600; color: {BRAND_ORANGE};",
                ),
                "] reproducir/pausa",
                style="color: #6c757d; font-size: 14px;"
            ),
            style="margin-bottom: 16px;"
        ),
        Div(
            id="waveform",
            style=(
                "width: 100%; height: 140px; background: #fff3e6; border-radius: 8px; "
                "margin-bottom: 12px;"
            ),
        ),
        Div(id="timeline", style="width: 100%; margin-bottom: 16px;"),
        waveform_controls,
        style="margin-bottom: 24px;"
    )

    return Div(
        intro,
        clip_info,
        waveform,
        form_inputs,
        actions,
        id="main-content",
        data_clip_id=str(clip.id),
        data_audio_path=str(audio_path_for_playback),
        data_original_audio_path=str(clip.audio_path),
        data_clip_start=f"{clip.start_timestamp:.2f}",
        data_clip_end=f"{clip.end_timestamp:.2f}",
        data_display_start=f"{padded_start:.2f}",
        data_display_end=f"{padded_end:.2f}",
        data_segment_offset=f"{segment_offset:.2f}",
        data_segment_duration=(
            f"{segment_duration:.2f}" if segment_duration is not None else ""
        ),
        # Mark that we are ALWAYS using a segment audio file here
        data_is_segment_audio="1",
    )


def render_empty_state() -> Div:
    """Render a friendly message when no clips are available."""

    return Div(
        H2("No hay más clips por ahora", style="text-align: center; color: #198754;"),
        P(
            "Apenas haya nuevo material para anotar lo vas a ver acá. Gracias por darte una vuelta.",
            style="text-align: center; font-size: 16px; color: #6c757d;"
        ),
        id="main-content",
        style="max-width: 640px; margin: 60px auto; background: white; padding: 32px; border-radius: 12px;"
    )


def render_main_content(clip: Optional[ClipRecord]) -> Div:
    """Render the main content area."""
    if clip:
        return render_clip_editor(clip)
    return render_empty_state()


def render_about_panel() -> Div:
    """Contenido de la pestaña About/Sobre el proyecto."""
    return Div(
        H2(
            "Sobre este proyecto",
            style=f"margin-bottom: 12px; color: {BRAND_ORANGE};",
        ),
        P(
            "Esta herramienta existe para fines científicos: queremos construir un conjunto de datos de "
            "transcripciones de audio en español chileno, para entrenar modelos de ",
            A(
                "reconocimiento automático de voz (ASR)",
                href="https://huggingface.co/tasks/automatic-speech-recognition",
                target="_blank",
                rel="noopener",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
            ),
            " de código abierto.",
            style="color: #495057; margin-bottom: 12px;"
        ),
        P(
            "Todas las ",
            Strong("transcripciones"),
            " que se generen aquí y los ",
            Strong("modelos"),
            " que entrenemos con estos datos serán liberados como ",
            Strong("recursos de código abierto"),
            ", para que cualquiera los pueda usar, revisar y mejorar.",
            style="color: #495057; margin-bottom: 12px;"
        ),
        P(
            "Cada vez que corriges un texto o ajustas un clip, estás ayudando a que en el futuro existan "
            "modelos de voz que entiendan mejor cómo hablamos en Chile.",
            style="color: #495057; margin-bottom: 16px;"
        ),
        H3("¿Quieres saber más o colaborar?", style="margin-bottom: 8px;"),
        P(
            "Si quieres conocer más de este proyecto o colaborar, escribe a ",
            A(
                "alonsoastroza@udd.cl",
                href="mailto:alonsoastroza@udd.cl",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
            ),
            ".",
            style="color: #495057; margin-bottom: 12px;"
        ),
        P(
            "El código de esta herramienta está disponible en ",
            A(
                "GitHub",
                href="https://github.com/idsudd/shiaaa",
                target="_blank",
                rel="noopener",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
            ),
            ".",
            style="color: #495057; margin-bottom: 12px;"
        ),
        P(
            "Agradecimientos especiales a ",
            A(
                "Fabián Choppelo",
                href="https://www.linkedin.com/in/fchoppelo/",
                target="_blank",
                rel="noopener",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
            ),
            " por el logo y la identidad visual, y a ",
            A(
                "Thomas Capelle",
                href="https://github.com/tcapelle",
                target="_blank",
                rel="noopener",
                style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
            ),
            " por el código original de la herramienta de anotación de datos visuales.",
            style="color: #495057; margin-bottom: 0;"
        ),
        style="margin-bottom: 20px; padding: 20px; border-radius: 12px; background: #f8f9fa;"
    )


def render_tab_shell(
    active_tab: str,
    clip: Optional[ClipRecord],
    status_message: Optional[str] = None,
) -> Div:
    """Render the tab navigation plus whichever panel is active."""

    clip_id_value = str(clip.id) if clip else ""

    def tab_button_style(is_active: bool) -> str:
        base = (
            "padding: 10px 18px; border-radius: 999px; border: none; font-weight: 600; "
            "cursor: pointer; transition: background 0.2s ease, color 0.2s ease;"
        )
        colors = (
            f" background: {BRAND_ORANGE}; color: #ffffff; box-shadow: 0 8px 16px rgba(247, 147, 30, 0.25);"
            if is_active
            else " background: #fff1e1; color: #8b4d13; border: 1px solid #ffd2a4;"
        )
        return base + colors

    annotate_button = Button(
        "Anotar",
        type="button",
        cls=f"tab-button{' active' if active_tab == 'anotar' else ''}",
        hx_get=f"/tab/anotar?clip_id={clip_id_value}",
        hx_target="#tab-shell",
        hx_swap="outerHTML",
        hx_indicator="#tab-loading",
        style=tab_button_style(active_tab == "anotar"),
    )

    ranking_button = Button(
        "Ranking",
        type="button",
        cls=f"tab-button{' active' if active_tab == 'ranking' else ''}",
        hx_get=f"/tab/ranking?clip_id={clip_id_value}",
        hx_target="#tab-shell",
        hx_swap="outerHTML",
        hx_indicator="#tab-loading",
        style=tab_button_style(active_tab == "ranking"),
    )

    faq_button = Button(
        "FAQ",
        type="button",
        cls=f"tab-button{' active' if active_tab == 'faq' else ''}",
        hx_get=f"/tab/faq?clip_id={clip_id_value}",
        hx_target="#tab-shell",
        hx_swap="outerHTML",
        hx_indicator="#tab-loading",
        style=tab_button_style(active_tab == "faq"),
    )

    about_button = Button(
        "Acerca de",
        type="button",
        cls=f"tab-button{' active' if active_tab == 'about' else ''}",
        hx_get=f"/tab/about?clip_id={clip_id_value}",
        hx_target="#tab-shell",
        hx_swap="outerHTML",
        hx_indicator="#tab-loading",
        style=tab_button_style(active_tab == "about"),
    )

    indicator = Div(
        "🔄 Cambiando de pestaña...",
        id="tab-loading",
        cls="htmx-indicator",
        style="padding: 8px 12px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #664d03; font-size: 14px;",
    )

    status_box = None
    if status_message:
        status_box = Div(
            status_message,
            role="status",
            style=(
                "margin-bottom: 16px; padding: 12px; border-radius: 8px; background: #fff3cd; "
                "border: 1px solid #ffeaa7; color: #664d03;"
            ),
        )

    if active_tab == "ranking":
        tab_content = render_contributor_stats()
    elif active_tab == "faq":
        tab_content = render_faq_panel()
    elif active_tab == "about":
        tab_content = render_about_panel()
    else:
        tab_content = render_main_content(clip)

    body_children = [child for child in (status_box, tab_content) if child is not None]

    return Div(
        Div(
            annotate_button,
            ranking_button,
            faq_button,
            about_button,
            indicator,
            cls="tab-nav",
            style="display: flex; gap: 8px; margin-bottom: 20px; align-items: center; justify-content: flex-end; flex-wrap: wrap;",
        ),
        *body_children,
        id="tab-shell",
    )


def render_contributor_stats() -> Div:
    """Render a panel showing contributor statistics."""
    try:
        stats = get_contributor_stats()

        if stats["total_contributors"] == 0:
            return Div(
                H4("Ranking de quienes están dando una mano", style=f"margin-bottom: 10px; color: {BRAND_ORANGE};"),
                P(
                    "Todavía no hay nombres en la lista. Deja el tuyo cuando mandes una anotación y aparecerás acá.",
                    style="color: #6c757d; font-style: italic;"
                ),
                cls="contributor-stats-panel",
                style=(
                    "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                    "border-radius: 8px;"
                ),
            )

        contributor_list = []
        for i, contributor in enumerate(stats["contributors"]):
            if i == 0:
                rank_emoji = "🥇"
            elif i == 1:
                rank_emoji = "🥈"
            elif i == 2:
                rank_emoji = "🥉"
            else:
                rank_emoji = "⭐"
            contributor_list.append(
                Div(
                    Span(f"{rank_emoji} {contributor['name']}", style="font-weight: 600;"),
                    Span(
                        f" - {contributor['contributions']} aportes",
                        style="color: #6c757d; margin-left: 8px;"
                    ),
                    style="margin-bottom: 4px;"
                )
            )

        return Div(
            H4("Ranking de quienes están dando una mano", style=f"margin-bottom: 10px; color: {BRAND_ORANGE};"),
            Div(
                P(
                    f"Personas que ayudaron: {stats['total_contributors']} · Transcripciones enviadas: {stats['total_contributions']}",
                    style="margin-bottom: 12px; font-weight: 500; color: #495057;"
                ),
                *contributor_list,
                style="margin-bottom: 8px;"
            ),
            P(
                "Gracias por la ayuda 💚. ",
                style="color: #198754; font-style: italic; margin-bottom: 0; font-size: 14px;"
            ),
            cls="contributor-stats-panel",
            style=(
                "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                "border-radius: 8px;"
            ),
        )
    except Exception as e:
        print(f"Error rendering contributor stats: {e}")
        return Div()  # Return empty div on error


def render_faq_panel() -> Div:
    """Render the frequently asked questions section."""
    return Div(
            Div(
                P(
                    Strong("¿Cómo debo transcribir palabras como \"weno\"?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "Escribe lo que escuchas. Ambas formas sirven, pero preferimos \"weno\" porque refleja cómo hablaríamos y cómo te gustaría verlo transcrito por tu app favorita.",
                    style="color: #495057; margin-bottom: 14px;",
                ),
            ),
            Div(
                P(
                    Strong("¿Escribo solo lo que se entiende o relleno lo que falta?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "No completes con suposiciones ni inventes texto. Corrige lo que sí se entiende y deja fuera lo que no se escucha o queda incompleto. En el caso que justo al audio corte la pronunciación de una palabra, puedes ajustar el tiempo de fin para que no quede a medias.",
                    style="color: #495057; margin-bottom: 14px;",
                ),
            ),
            Div(
                P(
                    Strong("¿Qué hago si el recorte del clip quedó corrido?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "Mueve los tiempos de inicio y fin para que el fragmento se alinee con lo que escuchas. Solo ajusta los extremos, no necesitas recortar todo el audio.",
                    style="color: #495057; margin-bottom: 14px;",
                ),
            ),
            Div(
                P(
                    Strong("El audio se escucha mal o no tiene texto, ¿qué hago?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "Avísanos con el botón \"Reportar audio\" cuando el clip no se entienda o no corresponda a una transcripción.",
                    style="color: #495057; margin-bottom: 14px;",
                ),
            ),
            Div(
                P(
                    Strong("¿Cómo se usarán los datos que envío?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "Todos los datos y los modelos entrenados con ellos serán de código abierto y estarán disponibles públicamente para cualquiera.",
                    style="color: #495057; margin-bottom: 14px;",
                ),
            ),
            Div(
                P(
                    Strong("¿Habrá una API para usar el modelo en mi aplicación?"),
                    style="margin-bottom: 6px; color: #343a40;",
                ),
                P(
                    "El modelo aún no está listo, pero lo liberaremos como un modelo open source para que puedas integrarlo como prefieras.",
                    style="color: #495057; margin-bottom: 0;",
                ),
            ),
            style="background: #f8f9fa; padding: 16px; border-radius: 10px; border: 1px solid #e9ecef; display: flex; flex-direction: column; gap: 8px;",
        ),
    )


APP_SCRIPT = Script("""
    let wavesurfer = null;
    let wsRegions = null;
    let currentRegion = null;

    function initWaveSurfer() {
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
        }

        const mainContent = document.getElementById('main-content');
        if (!mainContent) {
            return;
        }

        const audioPath = mainContent.dataset.audioPath;
        const segmentOffset = parseFloat(mainContent.dataset.segmentOffset || '0');
        const segmentDuration = parseFloat(mainContent.dataset.segmentDuration || '0');
        const clipStartAbsolute = parseFloat(mainContent.dataset.clipStart || '0');
        const clipEndAbsolute = parseFloat(mainContent.dataset.clipEnd || '0');
        const displayStartAbsolute = parseFloat(mainContent.dataset.displayStart || clipStartAbsolute);
        const displayEndAbsolute = parseFloat(mainContent.dataset.displayEnd || clipEndAbsolute);
        const isSegmentAudio = true; // we always play the segment audio now

        const clipStartRelative = Math.max(0, clipStartAbsolute - segmentOffset);
        const clipEndRelative = Math.max(clipStartRelative, clipEndAbsolute - segmentOffset);
        let displayStartRelative = Math.max(0, displayStartAbsolute - segmentOffset);
        let displayEndRelative = Math.max(displayStartRelative, displayEndAbsolute - segmentOffset);

        if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
            displayStartRelative = Math.min(displayStartRelative, segmentDuration);
            displayEndRelative = Math.min(displayEndRelative, segmentDuration);
        }

        if (!audioPath) {
            return;
        }

        const startInput = document.getElementById('start-time-input');
        const endInput = document.getElementById('end-time-input');
        const startHiddenInput = document.getElementById('start-time-hidden');
        const endHiddenInput = document.getElementById('end-time-hidden');
        const startRelativeHiddenInput = document.getElementById('start-time-relative-hidden');
        const endRelativeHiddenInput = document.getElementById('end-time-relative-hidden');

        const clampRelativeTime = (value) => {
            let result = Number.isFinite(value) ? value : 0;
            result = Math.max(0, result);
            if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                result = Math.min(result, segmentDuration);
            }
            return result;
        };

        const toWaveformTime = (relativeValue) => {
            // Segment audio: waveform timeline is 0..segmentDuration
            if (!Number.isFinite(relativeValue)) return relativeValue;
            return relativeValue;
        };

        const fromWaveformTime = (waveformValue) => {
            if (!Number.isFinite(waveformValue)) return waveformValue;
            return waveformValue;
        };

        const updateInputsFromRegion = () => {
            if (!currentRegion) return;
            const startRelative = clampRelativeTime(fromWaveformTime(currentRegion.start));
            const endRelative = clampRelativeTime(fromWaveformTime(currentRegion.end));
            if (startInput) startInput.value = startRelative.toFixed(2);
            if (endInput) endInput.value = endRelative.toFixed(2);
            if (startHiddenInput) startHiddenInput.value = (segmentOffset + startRelative).toFixed(6);
            if (endHiddenInput) endHiddenInput.value = (segmentOffset + endRelative).toFixed(6);
            if (startRelativeHiddenInput) startRelativeHiddenInput.value = startRelative.toFixed(6);
            if (endRelativeHiddenInput) endRelativeHiddenInput.value = endRelative.toFixed(6);
        };

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#4F4A85',
            progressColor: '#383351',
            height: 140,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            responsive: true,
        });

        wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());
        const formatTimelineLabel = (seconds) => {
            const relativeSeconds = clampRelativeTime(fromWaveformTime(seconds));

            if (!Number.isNaN(segmentDuration) && relativeSeconds > segmentDuration) {
                return '';
            }

            if (relativeSeconds >= 100) return relativeSeconds.toFixed(0);
            if (relativeSeconds >= 10) return relativeSeconds.toFixed(1);
            return relativeSeconds.toFixed(2);
        };
        wavesurfer.registerPlugin(WaveSurfer.Timeline.create({
            container: '#timeline',
            formatTimeCallback: formatTimelineLabel,
        }));

        // Build audio URL depending on whether audio_folder is local or remote
        const audioFolder = '""" + f"{config.audio_folder}" + """';
        const audioUrl = audioFolder.startsWith('http')
            ? audioFolder + '/' + audioPath
            : '/' + audioFolder + '/' + audioPath;

        wavesurfer.load(audioUrl);

        wavesurfer.on('ready', () => {
            wsRegions.clearRegions();
            currentRegion = wsRegions.addRegion({
                start: toWaveformTime(clipStartRelative),
                end: toWaveformTime(clipEndRelative),
                color: 'rgba(13, 110, 253, 0.3)',
                drag: true,
                resize: true,
            });

            currentRegion.on('update', updateInputsFromRegion);
            currentRegion.on('update-end', () => {
                if (!currentRegion) return;
                const startRelative = clampRelativeTime(fromWaveformTime(currentRegion.start));
                const endRelative = clampRelativeTime(fromWaveformTime(currentRegion.end));
                const clampedStart = toWaveformTime(startRelative);
                const clampedEnd = toWaveformTime(Math.max(endRelative, startRelative));
                if (clampedStart !== currentRegion.start || clampedEnd !== currentRegion.end) {
                    currentRegion.setOptions({ start: clampedStart, end: clampedEnd });
                }
                updateInputsFromRegion();
            });
            updateInputsFromRegion();
        });

        const updateCurrentTime = () => {
            const timeDisplay = document.getElementById('current-time');
            if (timeDisplay && wavesurfer) {
                const relativeTime = clampRelativeTime(fromWaveformTime(wavesurfer.getCurrentTime()));
                timeDisplay.textContent = relativeTime.toFixed(2);
            }
        };

        wavesurfer.on('audioprocess', updateCurrentTime);
        wavesurfer.on('pause', updateCurrentTime);

        if (startInput) {
            startInput.addEventListener('input', (event) => {
                if (!currentRegion) return;
                const value = parseFloat(event.target.value);
                if (!Number.isNaN(value)) {
                    const desiredStart = clampRelativeTime(value);
                    const currentEnd = clampRelativeTime(fromWaveformTime(currentRegion.end));
                    const newStart = Math.min(desiredStart, currentEnd);
                    currentRegion.setOptions({ start: toWaveformTime(newStart) });
                    updateInputsFromRegion();
                }
            });
        }

        if (endInput) {
            endInput.addEventListener('input', (event) => {
                if (!currentRegion) return;
                const value = parseFloat(event.target.value);
                if (!Number.isNaN(value)) {
                    const desiredEnd = clampRelativeTime(value);
                    const currentStart = clampRelativeTime(fromWaveformTime(currentRegion.start));
                    const newEnd = Math.max(desiredEnd, currentStart);
                    currentRegion.setOptions({ end: toWaveformTime(newEnd) });
                    updateInputsFromRegion();
                }
            });
        }

        const playButton = document.getElementById('play-btn');
        const pauseButton = document.getElementById('pause-btn');
        const stopButton = document.getElementById('stop-btn');
        const speedSelect = document.getElementById('speed-select');

        if (playButton) {
            playButton.addEventListener('click', () => {
                let startRelative = currentRegion
                    ? clampRelativeTime(fromWaveformTime(currentRegion.start))
                    : clipStartRelative;
                let endRelative = currentRegion
                    ? clampRelativeTime(fromWaveformTime(currentRegion.end))
                    : clipEndRelative;
                startRelative = Math.max(0, startRelative - 0.2);
                endRelative = Math.max(startRelative, endRelative + 0.2);
                if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                    startRelative = clampRelativeTime(startRelative);
                    endRelative = clampRelativeTime(endRelative);
                }
                wavesurfer.play(
                    toWaveformTime(startRelative),
                    toWaveformTime(endRelative)
                );
            });
        }

        if (pauseButton) {
            pauseButton.addEventListener('click', () => wavesurfer && wavesurfer.pause());
        }

        if (stopButton) {
            stopButton.addEventListener('click', () => {
                if (!wavesurfer) return;
                wavesurfer.stop();
                wavesurfer.setTime(toWaveformTime(clipStartRelative));
            });
        }

        if (speedSelect) {
            speedSelect.addEventListener('change', (event) => {
                const rate = parseFloat(event.target.value);
                if (!Number.isNaN(rate) && wavesurfer) {
                    wavesurfer.setPlaybackRate(rate);
                }
            });
        }

        document.addEventListener('keydown', (event) => {
            if (!wavesurfer) return;
            if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) {
                return;
            }
            if (event.code === 'Space') {
                event.preventDefault();
                if (wavesurfer.isPlaying()) {
                    wavesurfer.pause();
                } else {
                    let startRelative = currentRegion
                        ? clampRelativeTime(fromWaveformTime(currentRegion.start))
                        : clipStartRelative;
                    let endRelative = currentRegion
                        ? clampRelativeTime(fromWaveformTime(currentRegion.end))
                        : clipEndRelative;
                    startRelative = Math.max(0, startRelative - 0.2);
                    endRelative = Math.max(startRelative, endRelative + 0.2);
                    if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                        startRelative = clampRelativeTime(startRelative);
                        endRelative = clampRelativeTime(endRelative);
                    }
                    wavesurfer.play(
                        toWaveformTime(startRelative),
                        toWaveformTime(endRelative)
                    );
                }
            }
            if (event.key.toLowerCase() === 'q' && currentRegion) {
                event.preventDefault();
                const time = clampRelativeTime(fromWaveformTime(wavesurfer.getCurrentTime()));
                const currentEnd = clampRelativeTime(fromWaveformTime(currentRegion.end));
                const newStart = Math.min(time, currentEnd);
                currentRegion.setOptions({ start: toWaveformTime(newStart) });
                updateInputsFromRegion();
            }
            if (event.key.toLowerCase() === 'w' && currentRegion) {
                event.preventDefault();
                const time = clampRelativeTime(fromWaveformTime(wavesurfer.getCurrentTime()));
                const currentStart = clampRelativeTime(fromWaveformTime(currentRegion.start));
                const newEnd = Math.max(time, currentStart);
                currentRegion.setOptions({ end: toWaveformTime(newEnd) });
                updateInputsFromRegion();
            }
        });
    }

    function syncClipRoute() {
        const mainContent = document.getElementById('main-content');
        if (!mainContent) {
            return;
        }
        const clipId = mainContent.dataset.clipId;

        if (clipId) {
            const newPath = `/clip/${clipId}`;
            if (window.location.pathname !== newPath) {
                window.history.replaceState({}, '', newPath);
            }
        } else if (window.location.pathname !== '/') {
            window.history.replaceState({}, '', '/');
        }
    }

    function handleClipLoaded() {
        initWaveSurfer();
        syncClipRoute();
    }

    document.addEventListener('DOMContentLoaded', handleClipLoaded);
    document.body.addEventListener('htmx:afterSwap', (event) => {
        if (event.target.id === 'main-content' || (event.target.querySelector && event.target.querySelector('#main-content'))) {
            handleClipLoaded();
        }
    });
""")


def render_app_page(clip: Optional[ClipRecord], status_message: Optional[str] = None) -> Titled:
    """Render the full application shell for the given clip."""

    tab_shell = render_tab_shell("anotar", clip, status_message)

    footer = Div(
        "Una iniciativa originada en el ",
        A(
            "Instituto de Data Science UDD",
            href="https://github.com/idsudd",
            target="_blank",
            rel="noopener",
            style=f"color: {BRAND_ORANGE_DARK}; font-weight: 600;"
        ),
        ".",
        style="margin-top: 32px; text-align: center; color: #6c757d;"
    )

    hero_header = Div(
        Img(
            src="/shiaaa.png",
            alt="¡shiaaa! logo",
            style="width: clamp(120px, 18vw, 180px); height: auto;",
            loading="lazy",
        ),
        cls="brand-hero",
        style=(
            "display: flex; justify-content: flex-start; align-items: center; gap: 12px;"
            "text-align: left; margin-bottom: 24px; padding: 8px 0;"
        ),
    )

    body_children = [
        hero_header,
        tab_shell,
        footer,
    ]

    return Titled(
        "",
        Div(
            Div(
                *body_children,
                cls="container",
                hx_boost="true",
            ),
            cls="page-shell",
        ),
        APP_SCRIPT,
    )


# Routes
@rt("/")
def index():
    """Main entry point for the crowdsourced clip review interface."""
    clip = select_random_clip()
    return render_app_page(clip)


@rt("/clip/{clip_id:int}")
def clip_detail(clip_id: int):
    """Load a specific clip directly via permalink."""
    clip = get_clip(str(clip_id))
    status_message = None
    if clip is None:
        status_message = f"El clip {clip_id} no está disponible. Te mostramos otro para que sigas."
        clip = select_random_clip()
    return render_app_page(clip, status_message=status_message)


@rt("/tab/{tab_name}")
def switch_tab(tab_name: str, clip_id: str = "", status_message: str = ""):
    """Render the requested tab via htmx."""

    normalized_tab = tab_name.lower()
    if normalized_tab not in {"anotar", "ranking", "faq", "about"}:
        normalized_tab = "anotar"

    clip = get_clip(clip_id) if clip_id else None
    if normalized_tab == "anotar" and clip is None:
        clip = select_random_clip()

    return render_tab_shell(normalized_tab, clip, status_message or None)


@rt("/next_clip", methods=["POST"])
def next_clip(
    clip_id: str = "",
    start_time: str = "0",
    end_time: str = "0",
    start_time_relative: str = "0",
    end_time_relative: str = "0",
    transcription: str = "",
    contributor_name: str = "",
):
    """Move to the next random clip without completing the current one."""
    current_clip = get_clip(clip_id)
    if current_clip:
        try:
            start = float(start_time)
            end = float(end_time)
        except ValueError:
            start = end = None
        if start is not None and start >= 0 and end > start:
            updates = {
                'start_timestamp': start,
                'end_timestamp': end,
                'text': transcription,
                'timestamp': datetime.now().isoformat(),
                'username': get_username(contributor_name),
            }
            rel_start, rel_end = parse_relative_offsets(start_time_relative, end_time_relative)
            if rel_start is not None and rel_end is not None:
                updates['relative_start_offset'] = rel_start
                updates['relative_end_offset'] = rel_end
            db_backend.update_clip(current_clip.id, updates)

    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/complete_clip", methods=["POST"])
def complete_clip(
    clip_id: str = "",
    start_time: str = "0",
    end_time: str = "0",
    start_time_relative: str = "0",
    end_time_relative: str = "0",
    transcription: str = "",
    contributor_name: str = "",
):
    """Finalize a clip as human reviewed and move to another task."""
    clip = get_clip(clip_id)
    if clip:
        try:
            start = float(start_time)
            end = float(end_time)
        except ValueError:
            start = end = None
        if start is not None and start >= 0 and end > start:
            updates = {
                'start_timestamp': start,
                'end_timestamp': end,
                'text': transcription,
                'timestamp': datetime.now().isoformat(),
                'username': get_username(contributor_name),
                'human_reviewed': True,
                'marked': False,
            }
            rel_start, rel_end = parse_relative_offsets(start_time_relative, end_time_relative)
            if rel_start is not None and rel_end is not None:
                updates['relative_start_offset'] = rel_start
                updates['relative_end_offset'] = rel_end
            db_backend.update_clip(clip.id, updates)
    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/flag_clip", methods=["POST"])
def flag_clip(
    clip_id: str = "",
    transcription: str = "",
    start_time: str = "0",
    end_time: str = "0",
    start_time_relative: str = "0",
    end_time_relative: str = "0",
    contributor_name: str = "",
):
    """Mark a clip as problematic so it disappears from the review queue."""
    clip = get_clip(clip_id)
    if clip:
        updates = {
            'text': transcription,
            'timestamp': datetime.now().isoformat(),
            'username': get_username(contributor_name),
            'marked': True,
        }
        try:
            start = float(start_time)
            end = float(end_time)
        except ValueError:
            start = end = None
        if start is not None and start >= 0 and end > start:
            updates['start_timestamp'] = start
            updates['end_timestamp'] = end
            rel_start, rel_end = parse_relative_offsets(start_time_relative, end_time_relative)
            if rel_start is not None and rel_end is not None:
                updates['relative_start_offset'] = rel_start
                updates['relative_end_offset'] = rel_end
        db_backend.update_clip(clip.id, updates)
    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/styles.css")
def get_styles():
    """Serve the CSS file."""
    css_path = Path("styles.css")
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    return Response("/* Styles not found */", media_type="text/css")


@rt("/shiaaa.png")
def get_logo():
    """Serve the site logo for the UI and social previews."""
    logo_path = ROOT_DIR / "shiaaa.png"
    if logo_path.exists():
        return FileResponse(str(logo_path), media_type="image/png")
    return Response("", status_code=404)


# Only create local audio route if audio_folder is a local path (not a URL)
if not AUDIO_FOLDER_IS_REMOTE:
    @rt(f"/{config.audio_folder}/{{audio_name:path}}")
    def get_audio(audio_name: str):
        """Serve audio files with security checks."""
        if ".." in audio_name or audio_name.startswith("/"):
            return Response("Invalid path", status_code=400)

        valid_exts = ('.webm', '.mp3', '.wav', '.ogg', '.m4a', '.flac')
        if not audio_name.lower().endswith(valid_exts):
            return Response("Invalid file type", status_code=400)

        audio_path = Path(config.audio_folder) / audio_name

        try:
            audio_dir = Path(config.audio_folder).resolve()
            resolved_path = audio_path.resolve()
            if not str(resolved_path).startswith(str(audio_dir)):
                return Response("Access denied", status_code=403)
        except Exception:
            return Response("Invalid path", status_code=400)

        if audio_path.exists():
            return FileResponse(
                str(audio_path),
                headers={"Cache-Control": "public, max-age=3600"}
            )
        return Response("Audio not found", status_code=404)


# Print startup info
if __name__ == "__main__":
    print(f"Starting {config.title}")
    print("Configuration:")
    print(f"  - Audio folder: {config.audio_folder}")
    print(f"  - Database: {db_backend.backend_label()}")
    print(f"  - Annotating as: {get_username()}")

    audio_files = [path for path in iter_audio_files(config.audio_path)]
    print(f"  - Total audio files: {len(audio_files)}")

    total_clips = db_backend.count_clips()
    print(f"  - Total clips: {total_clips}")

    try:
        fasthtml_serve(host="localhost", port=5001)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Goodbye!")
