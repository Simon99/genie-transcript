from __future__ import annotations

import json
from pathlib import Path

from genie_core.audio import transcribe_audio
from genie_core.llm import LMStudioClient


SYSTEM_PROMPT = """You are a meeting analyst. Given a timestamped transcript, produce a structured JSON summary.

Output format (strict JSON, no markdown):
{
  "title": "meeting title inferred from content",
  "topics": [
    {
      "title": "topic title",
      "summary": "brief summary of what was discussed",
      "key_points": ["point 1", "point 2"],
      "decisions": ["decision made, if any"],
      "action_items": ["action item, if any"],
      "time_range": {"start": 0.0, "end": 0.0},
      "source_segments": [
        {"start": 0.0, "end": 0.0, "text": "relevant quote"}
      ]
    }
  ],
  "participants_detected": ["speaker patterns if identifiable"],
  "overall_summary": "1-2 sentence summary of the entire meeting"
}

Rules:
- Every topic MUST include time_range and source_segments with exact timestamps from the transcript
- source_segments should contain the most relevant quotes that support the summary
- Group related discussion into the same topic even if separated in time
- Keep the original language of the transcript
- Output ONLY valid JSON, no explanations"""


def structurize_transcript(
    input_path: str,
    output_dir: str,
    language: str = "zh",
    whisper_model: str = "medium",
    llm_model: str = "qwen3.6-35b-a3b-mtp",
    lm_studio_url: str = "http://localhost:1234/v1",
    progress_callback=None,
) -> dict:
    """Convert a recording or transcript to structured meeting notes.

    input_path: video/audio file, or .srt/.json transcript file
    Returns {"structured": str, "transcript": str, "topics": int}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get transcript segments
    if progress_callback:
        progress_callback("transcribing", 0)

    segments = _load_or_transcribe(input_path, language, whisper_model)

    # Save raw transcript
    transcript_path = output_dir / "transcript.json"
    transcript_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    srt_path = output_dir / "transcript.srt"
    _write_srt(segments, str(srt_path))

    # Step 2: Send to LLM for structuring
    if progress_callback:
        progress_callback("structuring", 0.4)

    transcript_text = _format_transcript_for_llm(segments)

    llm = LMStudioClient(base_url=lm_studio_url, model=llm_model)
    raw_response = llm.complete(
        prompt=f"Analyze this meeting transcript and produce structured notes:\n\n{transcript_text}",
        system=SYSTEM_PROMPT,
        temperature=0.2,
    )

    # Parse and save structured output
    if progress_callback:
        progress_callback("saving", 0.8)

    structured = _parse_llm_response(raw_response)

    structured_path = output_dir / "structured.json"
    structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate markdown with timestamp links
    md_path = output_dir / "notes.md"
    md_content = _generate_markdown(structured, segments)
    md_path.write_text(md_content, encoding="utf-8")

    # Generate HTML
    html_path = output_dir / "notes.html"
    html_content = _generate_html(md_content, structured)
    html_path.write_text(html_content, encoding="utf-8")

    if progress_callback:
        progress_callback("done", 1.0)

    topic_count = len(structured.get("topics", []))
    return {
        "structured": str(structured_path),
        "transcript": str(transcript_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "topics": topic_count,
    }


def _load_or_transcribe(input_path: str, language: str, model: str) -> list[dict]:
    """Load existing transcript or run whisper."""
    p = Path(input_path)

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    if p.suffix == ".srt":
        return _parse_srt(str(p))

    return transcribe_audio(str(p), language=language, model=model)


def _parse_srt(srt_path: str) -> list[dict]:
    """Parse SRT file into segments."""
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("\n\n"):
        parts = block.split("\n")
        if len(parts) >= 3:
            time_str = parts[1]
            text = " ".join(parts[2:])
            start_str, end_str = time_str.split(" --> ")
            segments.append({
                "start": _srt_time_to_seconds(start_str),
                "end": _srt_time_to_seconds(end_str),
                "text": text.strip(),
            })
    return segments


def _srt_time_to_seconds(time_str: str) -> float:
    h, m, rest = time_str.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _format_transcript_for_llm(segments: list[dict]) -> str:
    """Format segments into readable transcript with timestamps."""
    lines = []
    for seg in segments:
        start = _format_time(seg["start"])
        end = _format_time(seg["end"])
        lines.append(f"[{start} - {end}] {seg['text']}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _parse_llm_response(response: str) -> dict:
    """Extract JSON from LLM response, handling potential markdown wrapping."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"title": "Parse Error", "topics": [], "overall_summary": text[:500]}


def _generate_markdown(structured: dict, segments: list[dict]) -> str:
    """Generate markdown meeting notes with timestamp references."""
    lines = []
    title = structured.get("title", "Meeting Notes")
    lines.append(f"# {title}\n")

    summary = structured.get("overall_summary", "")
    if summary:
        lines.append(f"**Summary:** {summary}\n")

    participants = structured.get("participants_detected", [])
    if participants:
        lines.append(f"**Participants:** {', '.join(participants)}\n")

    lines.append("---\n")

    for i, topic in enumerate(structured.get("topics", []), 1):
        time_range = topic.get("time_range", {})
        start = _format_time(time_range.get("start", 0))
        end = _format_time(time_range.get("end", 0))

        lines.append(f"## {i}. {topic.get('title', 'Untitled')} [{start} - {end}]\n")
        lines.append(f"{topic.get('summary', '')}\n")

        key_points = topic.get("key_points", [])
        if key_points:
            lines.append("### Key Points")
            for point in key_points:
                lines.append(f"- {point}")
            lines.append("")

        decisions = topic.get("decisions", [])
        if decisions:
            lines.append("### Decisions")
            for d in decisions:
                lines.append(f"- {d}")
            lines.append("")

        actions = topic.get("action_items", [])
        if actions:
            lines.append("### Action Items")
            for a in actions:
                lines.append(f"- [ ] {a}")
            lines.append("")

        sources = topic.get("source_segments", [])
        if sources:
            lines.append("### Source References")
            for src in sources:
                s = _format_time(src.get("start", 0))
                e = _format_time(src.get("end", 0))
                lines.append(f"> [{s}-{e}] \"{src.get('text', '')}\"")
            lines.append("")

    return "\n".join(lines)


def _generate_html(markdown_text: str, structured: dict) -> str:
    """Generate HTML with clickable timestamp references."""
    lines = []
    lines.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    lines.append("<title>{}</title>".format(structured.get("title", "Meeting Notes")))
    lines.append("<style>")
    lines.append("body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;line-height:1.6}")
    lines.append("h1{color:#1a1a2e} h2{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}")
    lines.append(".time{color:#e94560;font-family:monospace;font-weight:bold}")
    lines.append("blockquote{border-left:3px solid #e94560;padding-left:10px;color:#555;margin:10px 0}")
    lines.append(".topic{margin:25px 0;padding:15px;background:#f8f9fa;border-radius:8px}")
    lines.append(".action{color:#0f3460} .decision{color:#533483}")
    lines.append("</style></head><body>")

    title = structured.get("title", "Meeting Notes")
    lines.append(f"<h1>{title}</h1>")

    summary = structured.get("overall_summary", "")
    if summary:
        lines.append(f"<p><strong>Summary:</strong> {summary}</p>")

    for i, topic in enumerate(structured.get("topics", []), 1):
        time_range = topic.get("time_range", {})
        start = _format_time(time_range.get("start", 0))
        end = _format_time(time_range.get("end", 0))

        lines.append(f'<div class="topic">')
        lines.append(f'<h2>{i}. {topic.get("title", "")} <span class="time">[{start} - {end}]</span></h2>')
        lines.append(f'<p>{topic.get("summary", "")}</p>')

        for point in topic.get("key_points", []):
            lines.append(f"<li>{point}</li>")

        for d in topic.get("decisions", []):
            lines.append(f'<li class="decision">Decision: {d}</li>')

        for a in topic.get("action_items", []):
            lines.append(f'<li class="action">TODO: {a}</li>')

        for src in topic.get("source_segments", []):
            s = _format_time(src.get("start", 0))
            e = _format_time(src.get("end", 0))
            lines.append(f'<blockquote><span class="time">[{s}-{e}]</span> "{src.get("text", "")}"</blockquote>')

        lines.append("</div>")

    lines.append("</body></html>")
    return "\n".join(lines)
