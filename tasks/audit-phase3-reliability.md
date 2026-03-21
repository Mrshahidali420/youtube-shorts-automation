# Phase 3: Reliability & Robustness

Fix bugs causing incorrect behavior, lost uploads, resource waste, and race conditions.

---

## Issue 1 — New Browser Instance Created Per Video

**Severity:** Critical (resource waste)
**File:** `youtube_shorts/uploader.py:1051`
**Problem:** `use_pom_uploader()` calls `setup_browser()` internally to create a new Firefox/geckodriver instance for every single video upload. Meanwhile, `main()` already created a driver at line 1346 via `configure_driver()`. Two browser processes run simultaneously — the one from `main()` is completely unused. For a batch of 10 videos, 11 browser instances are created total.

**Fix:** Pass the existing driver from `main()` into `use_pom_uploader()`:
```python
# BEFORE:
def use_pom_uploader(video_path, metadata, ...):
    driver = setup_browser()  # line 1051 — creates a new browser every time
    ...

# AFTER:
def use_pom_uploader(video_path, metadata, ..., driver=None):
    own_driver = driver is None
    if own_driver:
        driver = setup_browser()
    try:
        ...
    finally:
        if own_driver:
            driver.quit()
```

In `main()`, pass the pre-existing driver:
```python
result = use_pom_uploader(video_path, metadata, ..., driver=driver)
```

**Verify:** Check log for "WebDriver setup" messages — should appear only once per run, not once per video.

---

## Issue 2 — Config Loaded at Module Import Time

**Severity:** High
**Files:**
- `youtube_shorts/downloader_channel.py:228-239`
- `youtube_shorts/uploader.py:604-615`

**Problem:** Config is loaded at module scope (not inside `main()`). If `config.txt` is missing, `print_fatal()` calls `exit(1)` during import, preventing any graceful error handling, mocking, or unit testing of these modules.

**Fix:** Move config loading inside `main()`:
```python
# BEFORE (at module scope):
config = load_config()
if not config:
    print_fatal("Failed to load config")  # calls exit(1)

# AFTER (inside main()):
def main():
    config = load_config()
    if not config:
        raise ConfigurationError("Failed to load config.txt — copy config.example.txt")
    ...
```

Define `ConfigurationError` in a shared exceptions module or at the top of each file.

**Verify:** `from youtube_shorts.uploader import use_pom_uploader` works even when `config.txt` is missing.

---

## Issue 3 — Race Conditions on JSON Cache Files

**Severity:** High
**Files:** All cache operations in `youtube_shorts/utils/cache_utils.py`, `youtube_shorts/utils/api_utils.py:172-194`, and any JSON read-modify-write in downloader/uploader

**Problem:** All JSON caches follow read-modify-write without file locking. If the downloader and uploader run simultaneously, both may read the same file, modify independently, and one will overwrite the other's changes. Cache corruption or data loss.

**Fix:**
1. Add `filelock` to `requirements.txt`: `filelock>=3.12.0`
2. In `cache_utils.py`, wrap all read-modify-write with a lock:
```python
from filelock import FileLock

def save_cache(cache_file, data):
    lock_file = cache_file + ".lock"
    with FileLock(lock_file, timeout=10):
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def load_cache(cache_file):
    lock_file = cache_file + ".lock"
    with FileLock(lock_file, timeout=10):
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    return {}
```

**Verify:** Run downloader and uploader simultaneously — no JSON decode errors, no lost entries.

---

## Issue 4 — Quota Bug: Adds Daily Limit Instead of Setting It

**Severity:** High
**File:** `youtube_shorts/utils/api_utils.py:141`
**Problem:** When a 403/429 quota error is detected, the code calls:
```python
update_quota_usage(get_config_value("API_QUOTA_DAILY_LIMIT", 10000))
```
`update_quota_usage()` ADDS the value to the current day's count. If quota is already at 9,500, it becomes 19,500. This means quota is never correctly marked as exhausted.

**Fix:** Add a dedicated function that sets the count to the daily limit:
```python
def mark_quota_exceeded():
    """Set today's quota usage to the daily limit, preventing further API calls."""
    daily_limit = get_config_value("API_QUOTA_DAILY_LIMIT", 10000)
    quota_data = load_quota_data()
    today = datetime.now().strftime("%Y-%m-%d")
    quota_data[today] = daily_limit
    save_quota_data(quota_data)
```

Replace `update_quota_usage(get_config_value(...))` at line 141 with `mark_quota_exceeded()`.

**Verify:** Simulate a quota error — daily count is set to 10000, not doubled.

---

## Issue 5 — Browser Crash Kills Entire Upload Run

**Severity:** High
**File:** `youtube_shorts/uploader.py:1526-1533`
**Problem:** On `NoSuchWindowException` or `InvalidSessionIdException`, the exception is re-raised, which terminates the entire upload run. All remaining videos in the batch are abandoned with no way to resume.

**Fix:** Catch at the main loop level and restart the browser:
```python
for video in videos_to_upload:
    try:
        result = use_pom_uploader(video, metadata, driver=driver)
    except (NoSuchWindowException, InvalidSessionIdException) as e:
        log_warning(f"Browser session lost: {e}. Restarting browser...")
        try:
            driver.quit()
        except Exception:
            pass
        driver = configure_driver()
        log_info("Browser restarted. Skipping current video and continuing.")
        continue
    except Exception as e:
        log_error(f"Upload failed for {video}: {e}")
        continue
```

**Verify:** Kill Firefox mid-run — next video in batch still uploads after browser restart.

---

## Issue 6 — POM vs Legacy `SCHEDULE` / `SCHEDULED` Mismatch

**Severity:** High
**Files:**
- `youtube_shorts/page_objects/visibility_page.py:35` — uses `SCHEDULED`
- `youtube_shorts/uploader.py:990` — legacy uses `SCHEDULE`

**Problem:** YouTube Studio may use either value depending on UI version. Using the wrong one will silently fail to select "Schedule" and default to immediate publication.

**Fix:** Add both as fallback locators in both the POM and legacy paths:
```python
# In visibility_page.py:
SCHEDULE_RADIO_LOCATORS = [
    (By.CSS_SELECTOR, "tp-yt-paper-radio-button[name='SCHEDULED']"),
    (By.CSS_SELECTOR, "tp-yt-paper-radio-button[name='SCHEDULE']"),
    (By.XPATH, "//tp-yt-paper-radio-button[@name='SCHEDULED' or @name='SCHEDULE']"),
]
```

**Verify:** Upload with scheduling enabled — confirm "Schedule" radio button is selected (not "Publish now").

---

## Issue 7 — No Duplicate Upload Detection on Retry

**Severity:** High
**File:** `youtube_shorts/uploader.py:1500-1545`
**Problem:** The retry loop (3 attempts) calls `use_pom_uploader()` which uploads the video file. If the video was uploaded but metadata filling failed, retrying will upload a duplicate video to YouTube. The YouTube channel will have multiple copies of the same Short.

**Fix:** Before uploading, check the Excel "Uploaded" sheet for the video filename:
```python
def is_already_uploaded(video_filename, excel_path):
    """Return True if this video filename appears in the Uploaded sheet."""
    from youtube_shorts.utils.excel_utils import load_uploaded_videos
    uploaded = load_uploaded_videos(excel_path)
    return any(v.get("filename") == video_filename for v in uploaded)

# In upload loop:
if is_already_uploaded(os.path.basename(video_path), SHORTS_DATA_FILE):
    log_warning(f"Skipping {video_path} — already in Uploaded sheet")
    continue
```

Additionally, consider using YouTube Data API to check recently uploaded videos before retrying.

**Verify:** Run upload twice on same video — second run skips with "already uploaded" message.

---

## Issue 8 — `set_config_value` Reloads Entire Config From Disk on Every Call

**Severity:** Medium
**File:** `youtube_shorts/utils/config_utils.py:216`
**Problem:** `set_config_value()` calls `config = load_config(config_file_arg)` at the start, reading and parsing the file from disk each time a single value is set. If called in a loop (e.g., updating multiple keys after a performance metrics run), this is O(n) file reads.

Similarly, `get_config_value()` at line 198 also reloads from disk when no dict is passed.

**Fix:** Add module-level caching:
```python
_config_cache: dict = {}
_config_file_cache: str = ""

def load_config(config_file=None, force_reload=False):
    global _config_cache, _config_file_cache
    resolved = config_file or _default_config_path()
    if not force_reload and _config_file_cache == resolved and _config_cache:
        return _config_cache.copy()
    _config_cache = _load_from_file(resolved)
    _config_file_cache = resolved
    return _config_cache.copy()

def set_config_value(key, value, config_file=None):
    config = load_config(config_file)  # uses cache
    config[key] = value
    _write_config_file(config, config_file)
    _config_cache.update({key: value})  # keep cache in sync
```

**Verify:** Call `set_config_value` 10 times rapidly — only 1 file read occurs (confirm with logging).
