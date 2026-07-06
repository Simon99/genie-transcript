from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .structurize import structurize_transcript


def main():
    parser = argparse.ArgumentParser(description="Convert meeting recording to structured notes")
    parser.add_argument("input", help="Path to video/audio file, or .srt/.json transcript")
    parser.add_argument("-o", "--output", help="Output directory (default: <input>_notes/)")
    parser.add_argument("--language", default="zh", help="Whisper language hint (default: zh)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size (default: medium)")
    parser.add_argument("--llm-model", default="qwen3.6-35b-a3b-mtp", help="LM Studio model for structuring")
    parser.add_argument("--url", default="http://localhost:1234/v1", help="LM Studio API URL")
    parser.add_argument("--context-tokens", type=int, default=8192,
                        help="LLM context size in tokens for chunking budget (default: 8192)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or str(input_path.with_suffix("")) + "_notes"

    def on_progress(stage, pct):
        stages = {
            "transcribing": "Transcribing audio",
            "structuring": "Structuring with LLM",
            "synthesizing": "Synthesizing chunk summaries",
            "saving": "Saving output",
            "done": "Done",
        }
        label = stages.get(stage, stage)
        print(f"\r[{pct:.0%}] {label}...", end="", flush=True)

    print(f"Processing: {input_path}")
    try:
        result = structurize_transcript(
            str(input_path),
            output_dir,
            language=args.language,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model,
            lm_studio_url=args.url,
            context_tokens=args.context_tokens,
            progress_callback=on_progress,
        )
    except (RuntimeError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone! {result['topics']} topics found")
    print(f"  Structured: {result['structured']}")
    print(f"  Transcript: {result['transcript']}")
    print(f"  Markdown:   {result['markdown']}")
    print(f"  HTML:       {result['html']}")


if __name__ == "__main__":
    main()
