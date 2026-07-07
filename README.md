# genie-transcript

錄音/逐字稿 → 結構化會議筆記:whisper 轉寫 + LLM 主題分段,每個主題帶時間範圍與原文引句(可回溯)。

## 需求

- genie-core(`[mlx]` extra)+ `ffmpeg`
- LM Studio 跑一顆文字模型(建議非 thinking 模型,如 qwen3.6-35b-a3b-turboquant)

## 用法

```bash
genie-transcript meeting.mp4                     # 輸出 meeting_notes/
genie-transcript transcript.json -o notes/       # 已有 transcript,跳過 whisper
genie-transcript rec.srt --llm-model qwen3.6-35b-a3b-turboquant-mlx
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `input` | — | 影片/音檔,或 `.srt` / `.json` transcript |
| `-o, --output` | `<input>_notes/` | 輸出目錄 |
| `--language` | zh | whisper 語言碼 |
| `--whisper-model` | medium | whisper 模型大小 |
| `--llm-model` | 自動挑選 | LM Studio 文字模型 |
| `--url` | `http://localhost:1234/v1` | LM Studio API |
| `--context-tokens` | 自動偵測 | 手動指定 context 預算(除錯用) |

## 輸出

```
notes/
  transcript.json / transcript.srt   # 原始逐字稿(可餵回 input 重跑 LLM 部分)
  structured.json                    # {title, topics:[{title, summary, key_points,
                                     #   decisions, action_items, time_range, source_segments}]}
  notes.md / notes.html              # 人類可讀版
```

## 長錄音處理(分層合併)

- context 預算自動偵測(讀 LM Studio 的 `max_context_length`,262k 模型下 41 分鐘會議單 chunk 直進)
- 超長錄音自動切 chunk → 各自結構化 → 剝引句樹狀合併 → 以 time_range 程式回填原文引句(timestamp 零幻覺)
- LLM 回傳格式不符(缺 `topics`)會帶 schema 提醒重試,再錯**非零退出**,不會靜默產出空筆記

## 已知坑

- thinking 模型(qwen3.5 系、glm-4.7-flash)推理會吃光輸出預算導致失敗,client 會報明確錯誤——換模型比調參數有效
- 中斷後重跑:把輸出目錄裡的 `transcript.json` 當 input,跳過重新轉寫
