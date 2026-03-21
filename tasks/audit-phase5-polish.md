# Phase 5: Polish & Maintenance

Low-priority cleanup — safe to do after Phases 1–4 are complete.

---

## Issue 1 — Upload Polling Too Slow

**Severity:** Medium
**File:** `youtube_shorts/page_objects/upload_page.py:138-239`
**Problem:** `wait_for_upload_complete` polls in 10s cycles. Each cycle calls `find_element_with_multiple_locators` with `wait_very_short` (5s timeout). With 4 locator groups × 5s = 20s minimum per cycle. A 100MB upload that takes 2 minutes actually takes 3-4 minutes to detect as complete due to polling overhead.

**Fix:**
```python
# Reduce wait_very_short to 2s for polling checks
# Use exponential backoff: start at 2s, cap at 5s

check_interval = 2  # start at 2s
max_interval = 5

while elapsed < timeout:
    status = self._check_upload_status(wait_timeout=check_interval)
    if status in ("complete", "error"):
        break
    time.sleep(check_interval)
    check_interval = min(check_interval * 1.5, max_interval)
    elapsed += check_interval
```

---

## Issue 2 — Stalled Upload Only Warns, Never Recovers

**Severity:** Medium
**File:** `youtube_shorts/page_objects/upload_page.py:207-211`
**Problem:** When upload progress is stuck at the same percentage for 5+ minutes, the code logs a warning but continues waiting indefinitely. The run can hang forever.

**Fix:** Add a stall counter and abort after 3 consecutive stalls:
```python
stall_count = 0
last_progress = -1

if progress == last_progress:
    stall_count += 1
    if stall_count >= 3:
        raise UploadStalledError(f"Upload stalled at {progress}% for {stall_count * check_interval}s")
else:
    stall_count = 0
    last_progress = progress
```

In the caller, catch `UploadStalledError` and move to the next video.

---

## Issue 3 — Dead Test Code Mixed With Production Code

**Severity:** Low
**Files:**
- `youtube_shorts/utils/test_utils.py` — manual test script using raw `assert`
- `youtube_shorts/utils/test_new_utils.py` — same pattern

**Problem:** Test files inside the production `utils/` package are imported as part of the package. They add confusion and bloat.

**Fix:** Move to `tests/` directory as proper pytest files, or delete if the tests are already superseded by Phase 4 pytest tests.

---

## Issue 4 — Missing `__all__` Exports

**Severity:** Low
**Files:** `youtube_shorts/__init__.py`, key utils modules

**Problem:** Without `__all__`, `from youtube_shorts import *` imports everything, including internal helpers. Makes the public API unclear.

**Fix:** Add `__all__` to each public module:
```python
# youtube_shorts/__init__.py
__all__ = ["downloader_keyword", "downloader_channel", "uploader", "performance_tracker"]

# youtube_shorts/utils/auth_utils.py
__all__ = ["get_authenticated_service", "refresh_credentials"]

# youtube_shorts/utils/config_utils.py
__all__ = ["load_config", "get_config_value", "set_config_value"]
```

---

## Issue 5 — Placeholder `__author__` in `__init__.py`

**Severity:** Low
**File:** `youtube_shorts/__init__.py:8`
**Problem:** `__author__ = "Your Name"` — placeholder was never updated.

**Fix:** Update to the actual author name.

---

## Issue 6 — Loose Version Pins in `requirements.txt`

**Severity:** Medium
**File:** `requirements.txt`
**Problem:** All deps use only `>=` minimum (e.g., `selenium>=4.10.0`). A breaking change in any dependency will silently break the project on a fresh install.

**Fix:** Pin upper bounds using compatible release specifier:
```
selenium~=4.10          # allows 4.x, blocks 5.x
yt-dlp>=2023.3.4        # yt-dlp releases frequently — keep >= only
google-generativeai~=0.3
google-api-python-client~=2.0
google-auth-oauthlib~=1.0
openpyxl~=3.1
colorama~=0.4
psutil~=5.9
webdriver-manager~=4.0
filelock~=3.12          # new dep from Phase 3
```

---

## Issue 7 — Missing `google-auth-httplib2` in `requirements.txt`

**Severity:** Low
**File:** `requirements.txt`
**Problem:** `uploader.py` prints install instructions mentioning `google-auth-httplib2` but it's missing from `requirements.txt`. Fresh installs may fail for users needing it.

**Fix:** Add:
```
google-auth-httplib2>=0.1.0
```

---

## Issue 8 — Unused `import csv` in `uploader.py`

**Severity:** Low
**File:** `youtube_shorts/uploader.py:24`
**Problem:** `import csv  # Keep import for potential future use` — dead import, flagged by flake8.

**Fix:** Remove the line.

---

## Verification Checklist

- [ ] `pip install -r requirements.txt` in a fresh venv succeeds
- [ ] Flake8 passes with no unused imports
- [ ] Upload polling completes faster (measure time from upload completion to detection)
- [ ] Stalled upload is aborted and next video proceeds
- [ ] `python -c "import youtube_shorts; print(youtube_shorts.__all__)"` shows expected exports
