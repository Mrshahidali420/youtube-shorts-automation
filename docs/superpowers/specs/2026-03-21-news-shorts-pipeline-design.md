# News Shorts Pipeline — Design Spec
**Date:** 2026-03-21
**Status:** Approved
**Author:** Shahid Ali

---

## Overview

A fully automated pipeline that fetches top news stories, generates scripts with Gemini AI, collects relevant media assets, renders a YouTube Short using Remotion, and uploads via the existing uploader — with zero changes to existing codebase modules.

**Channel niche is configurable** — the same pipeline serves gaming, tech, finance, or general news channels by changing `NEWS_CATEGORY` in `config/config.txt`.

---

## Architecture

### Approach
Python orchestrates the full pipeline. Remotion (Node.js) is used only as a subprocess render engine — it reads a `render_data.json` file and outputs an `.mp4`. All orchestration, AI, and API logic stays in Python.

### Directory Structure

```
youtube_shorts/
├── news_shorts/                  # NEW — self-contained subpackage
│   ├── __init__.py
│   ├── news_pipeline.py          # master orchestrator
│   ├── news_fetcher.py           # fetch + score stories from NewsAPI/RSS/Reddit
│   ├── script_generator.py       # Gemini → script + metadata per story
│   ├── asset_collector.py        # orchestrates all asset sources
│   ├── news_scraper.py           # scrape images/video from article URLs
│   ├── image_search.py           # Google Images + Bing Images scraper (Playwright)
│   └── renderer.py               # writes render_data.json → calls Remotion subprocess
│
remotion-renderer/                # NEW — standalone Node.js project
├── src/
│   ├── NewsShort.tsx             # main 1080x1920 composition
│   ├── scenes/
│   │   ├── BackgroundLayer.tsx   # video/image + Ken Burns effect
│   │   ├── OverlayLayer.tsx      # semi-transparent gradient
│   │   ├── TextLayer.tsx         # animated headlines/bullets
│   │   ├── AudioLayer.tsx        # TTS voiceover
│   │   ├── ProgressBar.tsx       # bottom time progress indicator
│   │   └── BrandingLayer.tsx     # channel logo/watermark
│   └── index.ts
├── package.json
└── render.mjs                    # CLI entry: reads JSON → outputs .mp4

output/
└── news_assets/
    └── {story_id}/               # per-story downloaded assets
        ├── scraped/
        ├── stock/
        └── ai_generated/
```

### Existing Files — Unchanged
- `downloader_keyword.py`, `downloader_channel.py`, `uploader.py`, `performance_tracker.py`
- All `utils/` modules — reused directly
- `shorts_data.xlsx` — extended with new "News" sheet (schema defined below)

---

## Data Flow

```
NewsAPI / Google RSS / Reddit RSS
        ↓
   news_fetcher.py  (fetch + deduplicate + virality score)
        ↓
 script_generator.py  (Gemini → hook + scenes + CTA + metadata)
        ↓
  asset_collector.py
   ├── news_scraper.py     (article URL → scrape images/video)
   ├── image_search.py     (Google Images + Bing Images via Playwright)
   ├── Pexels API          (stock video + images)
   ├── Pixabay API         (stock images)
   └── Pollinations.ai     (free AI image generation, timeout=15s)
        ↓
    renderer.py  (Edge TTS voiceover → render_data.json → Remotion subprocess → .mp4)
        ↓
  existing uploader.py  (upload + schedule — unchanged)
        ↓
  existing Excel tracking  (new "News" sheet added)
```

---

## Module Specifications

### 0. Config Loading Pattern

All new modules follow the existing pattern — load config once at the top of `main()` or the entry function and pass the dict:

```python
from youtube_shorts.utils.config_utils import load_config, get_config_value

config = load_config()  # reads config/config.txt
count = get_config_value(config, "NEWS_SHORTS_PER_RUN", 3)
```

---

### 1. `news_fetcher.py`

**Sources (in priority order):**
1. NewsAPI — top headlines filtered by category (free tier: 100 req/day, **last 30 days only**)
2. Google News RSS — no key required, always-available fallback, supports real-time
3. Reddit RSS — `r/worldnews`, `r/gaming`, `r/technology` etc., real-time

> **Note:** `NEWS_MODE=trending` (last 6 hours) is not supported on NewsAPI free tier — it will fall back to Google News RSS and Reddit RSS automatically. A paid NewsAPI plan is required to use NewsAPI for trending mode.

**`story_id` generation:** `hashlib.md5(article_url.encode()).hexdigest()[:12]` — deterministic, collision-safe per URL.

**Output per story:**
```json
{
  "story_id": "a1b2c3d4e5f6",
  "title": "...",
  "summary": "2-3 sentence summary",
  "url": "https://...",
  "source": "BBC News",
  "published_at": "2026-03-21T10:00:00Z",
  "keywords": ["keyword1", "keyword2"],
  "virality_score": 8.2
}
```

**Virality scoring factors:** recency (last 24h weighted higher), source authority, keyword trending score.

**Deduplication:** stories with >70% title similarity are merged, keeping highest-authority source.

---

### 2. `script_generator.py`

Reuses the Gemini client from `utils/metadata_generator.py`:

```python
from youtube_shorts.utils.metadata_generator import MetadataGenerator

generator = MetadataGenerator(config)
script = generator.generate_news_script(story)  # new method to add
```

If `generate_news_script` does not yet exist in `MetadataGenerator`, add it — it follows the same retry/backoff pattern as existing generation methods.

**Prompt input:** story title + summary + channel niche + target duration (30–60s)

**Output:**
```json
{
  "hook": "You won't believe what just happened...",
  "scenes": [
    { "text": "Headline", "duration": 3, "keywords": ["..."] },
    { "text": "Key point 1", "duration": 4, "keywords": ["..."] },
    { "text": "Key point 2", "duration": 4, "keywords": ["..."] },
    { "text": "Key point 3", "duration": 4, "keywords": ["..."] }
  ],
  "cta": "Follow for daily news",
  "tts_script": "Full narration text for voiceover",
  "title": "YouTube Short title",
  "description": "YouTube description",
  "tags": ["news", "breaking", "..."]
}
```

---

### 3. `asset_collector.py`

**Priority order per scene:**
1. Scraped media from article URL (`news_scraper.py`)
2. Scraped from Google Images / Bing Images (`image_search.py` via Playwright)
3. Pexels video clip
4. Pexels / Pixabay image
5. Pollinations AI-generated image (free, no key, `timeout=15s`)
6. Gradient background (always succeeds)

**Asset ranking criteria:** keyword relevance score, minimum resolution 720p, portrait aspect ratio preferred (9:16).

**Pollinations usage:**
```python
import httpx
url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920"
response = httpx.get(url, timeout=30, follow_redirects=True)  # 10-30s expected latency
```
Pollinations is a free third-party service with no SLA. Always set `timeout=30` to cover the full latency range, and fall through to the next source on timeout.

**Output directory:** `output/news_assets/{story_id}/`

> **Copyright Notice:** Google/Bing scraped images may be subject to copyright. See README disclaimer.

---

### 4. `news_scraper.py` + `image_search.py`

**`news_scraper.py`:**
- Uses `BeautifulSoup4` to parse article HTML
- Extracts `<img>` tags, `<video>` tags, Open Graph `og:image`
- Downloads media above 400px minimum width
- Skips logos/ads (filter by size + URL pattern blacklist)

**`image_search.py`:**
- Uses `Playwright` (headless Chromium) to render and scrape Google Images and Bing Images
- Search query = story keywords + niche
- Applies Creative Commons license filter where available
- **No `requests`-based fallback** — Google/Bing require JS rendering. If Playwright is unavailable, this module is skipped entirely and the next source in the asset chain is used.

---

### 5. `renderer.py`

**Step 1 — TTS (Edge TTS, free, no key):**
```python
import asyncio
import edge_tts

async def _generate_tts(script: str, voice: str, output_path: str):
    await edge_tts.Communicate(script, voice=voice).save(output_path)

# Called synchronously from the pipeline:
asyncio.run(_generate_tts(tts_script, voice, str(voiceover_path)))
```
Voice is configurable: `TTS_VOICE = en-US-GuyNeural` in `config/config.txt`. Swappable to any TTS provider later — this is the only place TTS is called.

**Step 2 — Write `render_data.json` with absolute paths:**
```json
{
  "story_id": "a1b2c3d4e5f6",
  "duration": 45,
  "hook": "...",
  "scenes": [...],
  "cta": "...",
  "tts_audio": "C:/Users/.../output/news_assets/a1b2c3d4e5f6/voiceover.mp3",
  "style": {
    "theme": "dark",
    "accent_color": "#FF0000",
    "font": "Inter"
  }
}
```
All paths in `render_data.json` must be **absolute** — Remotion's Node.js subprocess resolves paths relative to its own working directory, not the Python CWD.

**Step 3 — Call Remotion with error handling:**
```python
result = subprocess.run(
    ["node", "remotion-renderer/render.mjs",
     "--data", str(render_data_path),
     "--output", str(output_mp4_path),
     "--concurrency", str(get_config_value(config, "REMOTION_CONCURRENCY", 4))],
    capture_output=True,
    text=True
)
if result.returncode != 0:
    raise RuntimeError(f"Remotion render failed: {result.stderr}")
if not output_mp4_path.exists():
    raise FileNotFoundError(f"Render completed but output file missing: {output_mp4_path}")
```

---

### 6. Remotion Composition (`NewsShort.tsx`)

**Spec:** 1080×1920px, 30fps, duration from `render_data.json`

**Layer stack (bottom to top):**
1. `BackgroundLayer` — video clip or image with Ken Burns slow zoom/pan + slight blur
2. `OverlayLayer` — dark gradient overlay (bottom 60% of frame) for text readability
3. `TextLayer` — per-scene text with spring-animated slide-in, typewriter, or fade
4. `AudioLayer` — TTS voiceover mp3 synced to timeline
5. `ProgressBar` — thin colored bar at bottom indicating video progress
6. `BrandingLayer` — channel logo bottom-right corner, configurable opacity

**Scene transitions:** crossfade (0.5s) between scenes.

**Optional effects (config toggle):**
- `REMOTION_PARTICLES = true` — subtle floating particle overlay
- `REMOTION_THEME = dark|light`

---

### 7. `news_pipeline.py` — Orchestrator

Mirrors the pattern of `downloader_keyword.py`. Runs as:
```bash
python -m youtube_shorts.news_shorts.news_pipeline
python -m youtube_shorts.news_shorts.news_pipeline --mode top_stories --count 3
python -m youtube_shorts.news_shorts.news_pipeline --category gaming
```

**Run modes:**
- `top_stories` — highest virality score stories (works on NewsAPI free tier)
- `trending` — most recent stories in last 6 hours (RSS/Reddit only on free tier)
- `category` — filtered by `NEWS_CATEGORY` config

**Per-story flow:**
```
fetch → script → assets → tts → render → upload → excel log
```

On failure at any step: log error, skip story, continue to next.

**`run_scripts.bat` update:** Add "News Shorts Pipeline" as a new menu option before "Exit" (currently option 6 → becomes option 7). Exit moves to slot 7.

---

## Excel "News" Sheet — Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `story_id` | string | MD5 hash of article URL |
| `title` | string | News story headline |
| `source` | string | e.g. "BBC News" |
| `category` | string | e.g. "gaming", "tech" |
| `published_at` | datetime | Original article publish time |
| `script_generated` | bool | Whether Gemini script succeeded |
| `assets_collected` | int | Number of assets downloaded |
| `render_status` | string | `success` / `failed` / `skipped` |
| `video_path` | string | Local path to rendered .mp4 |
| `upload_status` | string | `uploaded` / `failed` / `pending` |
| `youtube_url` | string | YouTube video URL after upload |
| `uploaded_at` | datetime | Upload timestamp |
| `scheduled_for` | datetime | Scheduled publish time (if scheduled) |

---

## Configuration (`config/config.txt` additions)

```ini
# News Shorts Pipeline
NEWS_SHORTS_PER_RUN = 3          # number of shorts per run (or "all")
NEWS_MODE = top_stories           # top_stories | trending | category
NEWS_CATEGORY = general           # general | gaming | tech | finance | sports
NEWS_API_KEY =                    # NewsAPI.org key (free tier, 100 req/day)
PEXELS_API_KEY =                  # Pexels API key (free)
PIXABAY_API_KEY =                 # Pixabay API key (free)
TTS_VOICE = en-US-GuyNeural       # Edge TTS voice (swappable to any provider later)
REMOTION_CONCURRENCY = 4          # parallel Chrome instances for rendering
REMOTION_THEME = dark             # dark | light
REMOTION_PARTICLES = false        # particle overlay effect
ASSET_MIN_RESOLUTION = 720        # minimum asset height in pixels
```

---

## New Dependencies

```
# requirements.txt additions
newsapi-python>=0.2.7
beautifulsoup4>=4.12.0
playwright>=1.40.0
httpx>=0.25.0
edge-tts>=6.1.9
```

**Node.js dependencies (`remotion-renderer/package.json`):**
```json
{
  "dependencies": {
    "remotion": "^4.0.0",
    "@remotion/cli": "^4.0.0",
    "@remotion/renderer": "^4.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}
```

---

## README Disclaimer

> **Asset Scraping Notice:** The image/video scraping feature (`image_search.py`) downloads publicly accessible media from Google Images, Bing Images, and news article pages. Scraped media may be subject to copyright. The operator is solely responsible for ensuring compliance with applicable copyright laws and platform terms of service before using scraped assets in published content. Use Creative Commons filtering where possible.

---

## Future Migration Path (Approach B — Full Node.js)

The `render_data.json` interface is the clean boundary for a future Node.js rewrite. When migrating:
- Replace `news_fetcher.py` → Node.js NewsAPI client
- Replace `script_generator.py` → Google Generative AI Node.js SDK
- Replace `asset_collector.py` → Node.js Pexels/Playwright
- Keep `remotion-renderer/` unchanged
- Keep existing Python uploader or replace with YouTube Data API Node.js client

---

## Verification Plan

1. `python -m youtube_shorts.news_shorts.news_pipeline --mode top_stories --count 1` — single story end-to-end
2. Check `output/news_assets/{story_id}/` — assets downloaded from multiple sources
3. Check `render_data.json` — all paths are absolute, TTS audio file exists
4. Check `.mp4` rendered at 1080×1920, correct duration
5. Check uploaded to YouTube with correct title/description/tags
6. Check "News" sheet in `shorts_data.xlsx` — all columns populated
7. Run `--category gaming` — verify category filtering works
8. Kill Playwright mid-run — verify fallback to Pexels works cleanly
9. Set Remotion to fail (bad Node path) — verify error is caught and logged, not silent
