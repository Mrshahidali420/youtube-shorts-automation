# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive menu (Windows)
run_scripts.bat

# Run modules directly
python -m youtube_shorts.downloader_keyword [--keyword "keyword"] [--max N]
python -m youtube_shorts.downloader_channel [--channel "url"] [--max N]
python -m youtube_shorts.uploader
python -m youtube_shorts.performance_tracker
```

No test framework is configured — verify behavior by running the scripts directly.

## Architecture

**Data flow:** Download → AI metadata generation → Excel review → Upload → Performance tracking → AI self-improvement

**4 entry points** in `youtube_shorts/`:
- `downloader_keyword.py` — discovers & downloads videos by keyword
- `downloader_channel.py` — downloads from specific YouTube channels
- `uploader.py` — uploads to YouTube via Selenium (YouTube Studio) + YouTube Data API
- `performance_tracker.py` — collects metrics from YouTube Analytics API

**`youtube_shorts/utils/`** — all shared logic:
- `constants.py` — single source of truth for all file/directory paths; always check here before hardcoding paths
- `metadata_generator.py` — AI metadata via Google Generative AI (Gemini)
- `api_utils.py` / `auth_utils.py` — YouTube Data API + OAuth2
- `excel_utils.py` — reads/writes `data/shorts_data.xlsx` (main tracking file)
- `ytdlp_utils.py` — wraps yt-dlp for video downloads

**`youtube_shorts/page_objects/`** — Selenium Page Object Model for YouTube Studio:
- `base_page.py` → `upload_page.py` → `details_page.py` → `visibility_page.py` → `confirmation_page.py`

## Key Directories

| Path | Purpose |
|------|---------|
| `config/config.txt` | Runtime settings (copy from `config.example.txt`) |
| `config/channels.txt` | Channel URLs for channel downloader |
| `config/niche.txt` | Keywords for keyword downloader |
| `data/client_secret.json` | OAuth credentials (not in git) |
| `data/token.json` | OAuth token cache (not in git) |
| `data/shorts_data.xlsx` | Central tracking spreadsheet |
| `output/shorts_downloads/` | Downloaded video files |
| `output/shorts_metadata/` | Per-video metadata JSON files |
| `output/uploaded_videos/` | Videos after successful upload |
| `logs/` | Per-script log files |

## Configuration

All paths are defined in `youtube_shorts/utils/constants.py`. Config values are read via `config_utils.py` from `config/config.txt`. Use `config/config.example.txt` as the template — it documents all available settings.

Authentication requires `data/client_secret.json` (Google OAuth2 credentials for YouTube Data API + Analytics API).

## Priority Skills

- `systematic-debugging` — use before proposing any fix for bugs or unexpected behavior
- `test-driven-development` — use when implementing Phase 4 test infrastructure
- `requesting-code-review` — use after completing any audit remediation phase
