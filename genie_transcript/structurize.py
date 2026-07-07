from __future__ import annotations

import json
import logging
from pathlib import Path

from genie_core.audio import transcribe_audio
from genie_core.audio.loader import load_transcript
from genie_core.audio.srt import write_srt
from genie_core.llm import LMStudioClient, extract_json, merge_structured
from genie_core.report import esc, html_page
from genie_core.text import format_time

logger = logging.getLogger(__name__)


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


# Self-contained merge instructions (merge_structured sends this as the user
# prompt with the JSON array of summaries appended; no system prompt).
SYNTHESIS_MERGE_PROMPT = """You are a meeting analyst. You will receive a JSON array of structured
meeting summaries. Each item covers part of the same meeting (or is a partially
merged result). Merge them into ONE final structured summary.

Output format (strict JSON, no markdown):
{
  "title": "meeting title",
  "topics": [
    {
      "title": "topic title",
      "summary": "brief summary of what was discussed",
      "key_points": ["point 1", "point 2"],
      "decisions": ["decision made, if any"],
      "action_items": ["action item, if any"],
      "time_range": {"start": 0.0, "end": 0.0}
    }
  ],
  "participants_detected": ["speaker patterns if identifiable"],
  "overall_summary": "1-2 sentence summary of the entire meeting"
}

Rules:
- Combine topics that cover the same subject; use a time_range that spans both
- Keep timestamps as numbers (seconds)
- Do NOT include source_segments; they are re-attached programmatically later
- Keep the original language of the content
- Output ONLY valid JSON, no explanations"""


# Keys kept on each topic when chunk summaries are fed to the synthesis merge
# (source_segments are stripped and backfilled from the transcript afterwards).
_TOPIC_MERGE_KEYS = ("title", "summary", "key_points", "decisions", "action_items", "time_range")

# Number of representative quotes backfilled per merged topic.
_BACKFILL_TOP_N = 5


def structurize_transcript(
    input_path: str,
    output_dir: str,
    language: str = "zh",
    whisper_model: str = "medium",
    llm_model: str = None,
    lm_studio_url: str = "http://localhost:1234/v1",
    context_tokens: int = None,
    progress_callback=None,
) -> dict:
    """Convert a recording or transcript to structured meeting notes.

    input_path: video/audio file, or .srt/.json transcript file
    context_tokens: model context size used for token-budget chunking
                    (a single chunk is capped at half of this); None
                    auto-detects from LM Studio, falling back to 8192.
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
    write_srt(segments, str(srt_path))

    # Step 2: Send to LLM for structuring (chunked for long transcripts)
    if progress_callback:
        progress_callback("structuring", 0.4)

    llm = LMStudioClient(base_url=lm_studio_url, model=llm_model)

    if context_tokens is None:
        context_tokens = llm.get_context_length(default=8192)
        print("Model context: %d tokens" % context_tokens)

    # Token-budget chunking: a single chunk is capped at half the model
    # context, and additionally at 32k so the chunk's structured summary
    # still fits comfortably in the generation budget.
    chunk_budget = min(max(1, context_tokens // 2), 32768)
    chunks = _chunk_segments(segments, chunk_budget)

    if len(chunks) <= 1:
        transcript_text = _format_transcript_for_llm(segments)
        structured = _complete_and_parse(
            llm,
            prompt="Analyze this meeting transcript and produce structured notes:\n\n%s" % transcript_text,
            system=SYSTEM_PROMPT,
            what="transcript structuring",
            required_key="topics",
        )
    else:
        # Multi-chunk: summarize each chunk, then synthesize hierarchically
        chunk_summaries = []
        for ci, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback("structuring", 0.4 + 0.3 * (ci + 1) / len(chunks))

            chunk_text = _format_transcript_for_llm(chunk)
            summary = _complete_and_parse(
                llm,
                prompt="Analyze this transcript chunk (%d/%d) and produce structured notes:\n\n%s" % (
                    ci + 1, len(chunks), chunk_text),
                system=SYSTEM_PROMPT,
                what="chunk %d/%d" % (ci + 1, len(chunks)),
                required_key="topics",
            )
            chunk_summaries.append(summary)

        # Synthesize: strip quotes, tree-merge under token budget, backfill quotes
        if progress_callback:
            progress_callback("synthesizing", 0.75)

        stripped = [_strip_source_segments(cs) for cs in chunk_summaries]
        structured = merge_structured(
            stripped,
            llm,
            merge_prompt=SYNTHESIS_MERGE_PROMPT,
            budget_tokens=chunk_budget,
            required_key="topics",
        )
        if not isinstance(structured, dict):
            raise RuntimeError(
                "Synthesis merge returned %s, expected a JSON object" % type(structured).__name__)
        _backfill_source_segments(structured, segments, top_n=_BACKFILL_TOP_N)

    # Parse and save structured output
    if progress_callback:
        progress_callback("saving", 0.8)

    structured_path = output_dir / "structured.json"
    structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate markdown with timestamp links
    md_path = output_dir / "notes.md"
    md_content = _generate_markdown(structured, segments)
    md_path.write_text(md_content, encoding="utf-8")

    # Generate HTML
    html_path = output_dir / "notes.html"
    html_content = _generate_html(structured)
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

    if p.suffix.lower() in (".json", ".srt"):
        return load_transcript(str(p))

    return transcribe_audio(str(p), language=language, model=model)


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for CJK-heavy text (~1.5 chars/token)."""
    return int(len(text) // 1.5)


def _chunk_segments(segments: list[dict], budget_tokens: int) -> list[list[dict]]:
    """Split segments into chunks whose formatted text fits the token budget."""
    chunks = []
    current = []
    current_tokens = 0
    for seg in segments:
        cost = _estimate_tokens(_format_segment_line(seg))
        if current and current_tokens + cost > budget_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += cost
    if current:
        chunks.append(current)
    return chunks


def _format_segment_line(seg: dict) -> str:
    start = format_time(seg.get("start", 0))
    end = format_time(seg.get("end", 0))
    return "[%s - %s] %s" % (start, end, seg.get("text", ""))


def _format_transcript_for_llm(segments: list[dict]) -> str:
    """Format segments into readable transcript with timestamps."""
    return "\n".join(_format_segment_line(seg) for seg in segments)


def _complete_and_parse(llm, prompt: str, system: str, what: str,
                        required_key: str = None) -> dict:
    """LLM call + JSON extraction; one retry at temperature=0, then raise.

    required_key: schema anchor (e.g. "topics"). Models sometimes return
    perfectly valid JSON in a different shape (observed live: single-chunk
    path got title/summary/key_points with no topics array); the retry
    appends an explicit reminder, and a still-wrong shape raises.
    """
    def _ok(result):
        return (isinstance(result, dict)
                and (required_key is None or required_key in result))

    raw = llm.complete(prompt=prompt, system=system, temperature=0.2, max_tokens=4096)
    try:
        result = extract_json(raw)
        if _ok(result):
            return result
    except ValueError:
        pass

    reminder = prompt if required_key is None else (
        prompt + "\n\nREMINDER: the output MUST be a JSON object with a "
                 "top-level \"%s\" array, exactly as the schema specifies."
        % required_key)
    raw = llm.complete(prompt=reminder, system=system, temperature=0, max_tokens=4096)
    try:
        result = extract_json(raw)
    except ValueError as e:
        raise RuntimeError(
            "LLM returned unparseable JSON for %s (after retry at temperature=0): %s"
            % (what, e))
    if not _ok(result):
        if isinstance(result, list) and result:
            logger.warning("LLM output for %s is a bare array; coercing into %r",
                           what, required_key)
            return {"title": "", required_key: result}
        if isinstance(result, dict) and result:
            logger.warning("LLM output for %s missing %r; coercing into wrapper",
                           what, required_key)
            return {"title": result.get("title", ""), required_key: [result]}
        raise RuntimeError(
            "LLM output for %s is missing required key %r (after retry)"
            % (what, required_key))
    return result


def _strip_source_segments(summary: dict) -> dict:
    """Drop source_segments (and unknown keys) from each topic before merging."""
    out = {k: v for k, v in summary.items() if k != "topics"}
    topics = []
    for topic in summary.get("topics", []) or []:
        if isinstance(topic, dict):
            topics.append({k: topic[k] for k in _TOPIC_MERGE_KEYS if k in topic})
    out["topics"] = topics
    return out


def _coerce_seconds(value, default=0.0) -> float:
    """Best-effort conversion of an LLM-provided timestamp to seconds."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return default
        if ":" in text:
            try:
                nums = [float(p) for p in text.split(":")]
            except ValueError:
                return default
            secs = 0.0
            for n in nums:
                secs = secs * 60 + n
            return secs
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _backfill_source_segments(structured: dict, segments: list[dict], top_n: int = _BACKFILL_TOP_N):
    """Re-attach representative quotes to each merged topic from the original
    transcript, using the topic's time_range (programmatic — no LLM, so
    timestamps cannot be hallucinated)."""
    for topic in structured.get("topics", []) or []:
        if not isinstance(topic, dict):
            continue
        time_range = topic.get("time_range") or {}
        if not isinstance(time_range, dict):
            time_range = {}
        start = _coerce_seconds(time_range.get("start"), 0.0)
        end = _coerce_seconds(time_range.get("end"), start)
        if end < start:
            start, end = end, start

        in_range = [
            s for s in segments
            if _coerce_seconds(s.get("end"), 0.0) >= start
            and _coerce_seconds(s.get("start"), 0.0) <= end
            and str(s.get("text", "")).strip()
        ]
        # Longest segments as representatives, re-sorted chronologically
        picked = sorted(in_range, key=lambda s: len(str(s.get("text", ""))), reverse=True)[:top_n]
        picked.sort(key=lambda s: _coerce_seconds(s.get("start"), 0.0))

        topic["source_segments"] = [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in picked
        ]
    return structured


def _safe_format_time(value) -> str:
    """format_time that never raises on LLM-provided garbage (None, etc.)."""
    try:
        return format_time(value)
    except ValueError:
        return format_time(_coerce_seconds(value, 0.0))


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
        start = _safe_format_time(time_range.get("start", 0))
        end = _safe_format_time(time_range.get("end", 0))

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
                s = _safe_format_time(src.get("start", 0))
                e = _safe_format_time(src.get("end", 0))
                lines.append(f"> [{s}-{e}] \"{src.get('text', '')}\"")
            lines.append("")

    return "\n".join(lines)


_HTML_CSS = """
body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;line-height:1.6}
h1{color:#1a1a2e} h2{color:#16213e;border-bottom:1px solid #ddd;padding-bottom:5px}
.time{color:#e94560;font-family:monospace;font-weight:bold}
blockquote{border-left:3px solid #e94560;padding-left:10px;color:#555;margin:10px 0}
.topic{margin:25px 0;padding:15px;background:#f8f9fa;border-radius:8px}
.action{color:#0f3460} .decision{color:#533483}
""".strip()


def _generate_html(structured: dict) -> str:
    """Generate an HTML report (all LLM-derived text escaped)."""
    lines = []
    title = structured.get("title", "Meeting Notes")
    lines.append("<h1>%s</h1>" % esc(title))

    summary = structured.get("overall_summary", "")
    if summary:
        lines.append("<p><strong>Summary:</strong> %s</p>" % esc(summary))

    for i, topic in enumerate(structured.get("topics", []), 1):
        time_range = topic.get("time_range", {})
        start = _safe_format_time(time_range.get("start", 0))
        end = _safe_format_time(time_range.get("end", 0))

        lines.append('<div class="topic">')
        lines.append('<h2>%d. %s <span class="time">[%s - %s]</span></h2>' % (
            i, esc(topic.get("title", "")), esc(start), esc(end)))
        lines.append("<p>%s</p>" % esc(topic.get("summary", "")))

        items = []
        for point in topic.get("key_points", []):
            items.append("<li>%s</li>" % esc(point))
        for d in topic.get("decisions", []):
            items.append('<li class="decision">Decision: %s</li>' % esc(d))
        for a in topic.get("action_items", []):
            items.append('<li class="action">TODO: %s</li>' % esc(a))
        if items:
            lines.append("<ul>")
            lines.extend(items)
            lines.append("</ul>")

        for src in topic.get("source_segments", []):
            s = _safe_format_time(src.get("start", 0))
            e = _safe_format_time(src.get("end", 0))
            lines.append('<blockquote><span class="time">[%s-%s]</span> "%s"</blockquote>' % (
                esc(s), esc(e), esc(src.get("text", ""))))

        lines.append("</div>")

    return html_page(title, "\n".join(lines), css=_HTML_CSS)
