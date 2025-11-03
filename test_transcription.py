#!/usr/bin/env python3
"""
Test script to verify Modal transcription works correctly.
"""
from pathlib import Path
import sys

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.modal_transcription import ModalWhisperTranscriber


def test_transcription():
    """Basic transcription test with Modal."""

    print("="*80)
    print("MODAL TRANSCRIPTION TEST")
    print("="*80)

    # Configuration
    model_name = "openai/whisper-large-v3"
    language = "es"

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Language: {language}")
    print(f"  VAD: Enabled (auto)")
    print(f"  Word timestamps: Disabled")

    # Create transcriber
    print("\nInitializing transcriber with Modal...")
    transcriber = ModalWhisperTranscriber(
        model_name=model_name,
        language=language,
        return_word_timestamps=False,
        chunking_strategy="auto",  # Uses VAD automatically
        batch_size=8,
    )
    print("✓ Transcriber initialized")

    # Find audio file to test
    audio_dir = Path("./audio")
    if not audio_dir.exists():
        print(f"\n❌ Error: Directory {audio_dir} does not exist")
        print("   Create an 'audio' directory and add .wav or .mp3 files to test")
        return

    # Find first audio file
    audio_files = list(audio_dir.glob("**/*.wav")) + list(audio_dir.glob("**/*.mp3"))
    if not audio_files:
        print(f"\n❌ Error: No audio files found in {audio_dir}")
        print("   Add .wav or .mp3 files to the 'audio' directory")
        return

    test_file = audio_files[0]
    print(f"\nTest file: {test_file}")

    # Transcribe
    print("\nTranscribing (this may take a few minutes on first run)...")
    print("  - Modal will download the model if not cached")
    print("  - GPU will start automatically")
    print("  - Audio will be segmented using VAD")

    try:
        result = transcriber.transcribe_file(test_file)

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        print(f"\nDetected language: {result.language or 'N/A'}")
        print(f"Number of segments: {len(result.segments)}")

        print("\nSegments:")
        for i, segment in enumerate(result.segments, 1):
            duration = segment.end - segment.start
            print(f"\n  [{i}] {segment.start:.2f}s - {segment.end:.2f}s (duration: {duration:.2f}s)")
            print(f"      \"{segment.text}\"")

            if segment.words:
                print(f"      Words: {len(segment.words)}")

        # Full text
        full_text = " ".join(s.text for s in result.segments)
        print(f"\nFull text:\n  {full_text}")

        print("\n" + "="*80)
        print("✓ Test completed successfully")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during transcription:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nTo process multiple files, use:")
    print("  python scripts/preprocess_audio.py --audio-folder ./audio")


if __name__ == "__main__":
    test_transcription()
