# Phase 4: Testing Infrastructure & Code Quality

Add automated testing, fix medium-priority issues across Selenium locators, scheduling, constants, and logging.

---

## 4A: Test Framework Setup

**Current state:** CI only runs flake8 and `python -c "import youtube_shorts"` (which fails due to Issue 1 from Phase 1). There are no pytest test cases. `test_utils.py` and `test_new_utils.py` in `utils/` are manual scripts with raw `assert` statements — not pytest-discoverable.

**Setup:**
1. Create `tests/` directory at project root
2. Add `pytest.ini` or `pyproject.toml`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```
3. Add `tests/__init__.py` (empty)
4. Add `tests/conftest.py` with shared fixtures:
```python
import pytest, os, tempfile, json

@pytest.fixture
def temp_config(tmp_path):
    """Return a minimal config.txt path for testing."""
    config = tmp_path / "config.txt"
    config.write_text("API_QUOTA_DAILY_LIMIT = 10000\nMAX_DOWNLOADS = 5\n")
    return str(config)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Return a temp data directory with required subdirs."""
    for d in ["data", "output", "logs", "config"]:
        (tmp_path / d).mkdir()
    return tmp_path
```

**Test files to create:**

| File | What to test |
|------|-------------|
| `tests/test_config_utils.py` | `load_config()` type coercion, defaults, env-var override, missing file |
| `tests/test_youtube_limits.py` | All 3 validator functions with boundary values |
| `tests/test_cache_utils.py` | Read/write/update cycles, missing file graceful handling |
| `tests/test_api_utils.py` | Quota tracking logic, `mark_quota_exceeded()`, retry counting |
| `tests/test_auth_utils.py` | JSON credential load/save (mock google.oauth2) |
| `tests/test_metadata_generator.py` | Metadata validation, tag truncation, title/description length limits |

**Update CI (`.github/workflows/python-tests.yml`):**
```yaml
- name: Run tests
  run: pytest tests/ -v --tb=short
```

---

## 4B: Medium Code Quality Fixes

### Issue 1 — Locale-Dependent XPaths in Page Objects

**Severity:** High
**Files:** `youtube_shorts/page_objects/upload_page.py:39-43`, `details_page.py:27-29,44-48`
**Problem:** XPaths like `contains(., '100%')` and `contains(text(), 'Video upload complete')` and `@label='Title'` and `@placeholder='Add tag'` are locale-dependent. YouTube Studio in non-English accounts will not match.

**Fix:** Add structural/aria fallbacks to every text-based locator:
```python
# upload_page.py — upload complete
UPLOAD_COMPLETE_LOCATORS = [
    (By.XPATH, "//*[contains(text(), 'Video upload complete')]"),        # English
    (By.XPATH, "//*[contains(text(), '100%')]"),                         # progress
    (By.CSS_SELECTOR, "ytcp-video-upload-progress[upload-status='done']"),  # structural
    (By.XPATH, "//*[@upload-status='done']"),                            # attribute
]

# details_page.py — title input
TITLE_LOCATORS = [
    (By.XPATH, "//*[@label='Title (required)']"),
    (By.XPATH, "//*[@label='Title']"),
    (By.CSS_SELECTOR, "#title-textarea"),
    (By.XPATH, "//ytcp-social-suggestion-input[@id='title-textarea']"),
]

# details_page.py — tags
TAGS_LOCATORS = [
    (By.XPATH, "//*[@placeholder='Add tag']"),
    (By.CSS_SELECTOR, "ytcp-chip-bar#tags-container input"),
    (By.XPATH, "//ytcp-chip-bar[@id='tags-container']//input"),
]
```

---

### Issue 2 — No Timezone Handling in Scheduling

**Severity:** Medium
**File:** `youtube_shorts/uploader.py:1462-1498`
**Problem:** `datetime.now()` returns naive local time. YouTube Studio interprets dates in the account's timezone. If the machine and account timezone differ, videos schedule at wrong times.

**Fix:**
```python
from datetime import datetime, timezone
import zoneinfo  # Python 3.9+

# Get timezone from config (add to config.txt):
# ACCOUNT_TIMEZONE = America/New_York
account_tz_name = get_config_value("ACCOUNT_TIMEZONE", "UTC")
account_tz = zoneinfo.ZoneInfo(account_tz_name)

# Use aware datetime throughout:
now = datetime.now(tz=account_tz)
schedule_dt = now + timedelta(hours=schedule_offset)
```

Also add `ACCOUNT_TIMEZONE` to `config.example.txt` with documentation.

---

### Issue 3 — Conflicting Limit Constants Across 4 Files

**Severity:** Medium
**Files:**
- `youtube_shorts/utils/constants.py` — `YOUTUBE_TOTAL_TAGS_LIMIT = 460`
- `youtube_shorts/utils/config_utils.py` DEFAULT_CONFIG — `YOUTUBE_TOTAL_TAGS_LIMIT = 450`
- `youtube_shorts/utils/youtube_limits.py` — defines its own set
- `youtube_shorts/utils/metadata_generator.py` — local `MAX_TAGS = 500`, `MAX_TAG_LENGTH = 30`

**Fix:**
1. `constants.py` is the single source of truth for all YouTube limits
2. Delete `YOUTUBE_TOTAL_TAGS_LIMIT` from `config_utils.py` DEFAULT_CONFIG
3. In `metadata_generator.py`, import from constants:
```python
from youtube_shorts.utils.constants import YOUTUBE_TOTAL_TAGS_LIMIT, MAX_TAG_LENGTH
```
4. In `youtube_limits.py`, import from constants instead of redefining

---

### Issue 4 — Hardcoded "GTA" in Keyword Generation

**Severity:** Medium
**File:** `youtube_shorts/downloader_channel.py:460-486`
**Problem:** `generate_keywords_from_niche()` has hardcoded references to "GTA" and "Grand Theft Auto" in both the AI prompt and the result filter (line 482). Unusable for any other niche.

**Fix:** Make filter configurable — add `KEYWORD_FILTER_GAME` to `config.txt`, default empty:
```python
niche_filter = get_config_value("KEYWORD_FILTER_GAME", "")
if niche_filter and niche_filter.lower() not in keyword.lower():
    continue
```
Or remove the filter entirely if niche is already passed as a parameter.

---

### Issue 5 — Hardcoded Firefox User-Agent

**Severity:** Medium
**File:** `youtube_shorts/uploader.py:688`
**Problem:** Specific Firefox 124.0 user-agent string is hardcoded. This becomes outdated and may trigger bot detection.

**Fix:** Move to `constants.py` with a more flexible approach:
```python
# constants.py
FIREFOX_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]
```
Pick one at random at browser setup time.

---

### Issue 6 — `logging.basicConfig()` Called in Library Module

**Severity:** Medium
**File:** `youtube_shorts/utils/excel_utils.py:70`
**Problem:** Calling `logging.basicConfig()` inside an imported module overrides the root logger configuration set by the application. Any application importing `excel_utils` will have its logging configuration corrupted.

**Fix:**
```python
# BEFORE (anti-pattern in library):
logging.basicConfig(level=logging.INFO, ...)

# AFTER (correct library pattern):
logger = logging.getLogger(__name__)
# Do NOT call basicConfig — let the application configure root logging
```

---

### Issue 7 — `print_fatal` Calls `exit(1)` With No Cleanup

**Severity:** Medium
**Files:** `downloader_keyword.py:29`, called from lines 1596, 1597, 1601, 1603, 1624-1626, 1683, 1746, 1814-1817, 1829, 2494-2496

**Problem:** `exit(1)` terminates the entire Python process immediately. No `finally` blocks run, no Excel saves, no WebDriver cleanup. Any in-progress work is lost.

**Fix:** Define a custom exception:
```python
class FatalError(Exception):
    """Raised instead of exit(1) to allow cleanup in finally blocks."""
    pass

def print_fatal(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise FatalError(msg)
```

In `main()`, wrap with:
```python
try:
    main_logic()
except FatalError as e:
    log_error(f"Fatal error: {e}")
    sys.exit(1)
finally:
    # cleanup: save Excel, quit WebDriver, etc.
    cleanup()
```

---

### Issue 8 — Inconsistent Browser Setup (POM vs Legacy)

**Severity:** Medium
**Files:** `youtube_shorts/uploader.py:688-691` (configure_driver), `youtube_shorts/page_objects/uploader_pom.py:169-217` (setup_browser)
**Problem:** The POM's `setup_browser()` does not set anti-detection preferences that `configure_driver()` does set (disable WebRTC, disable geolocation, automation flags). Inconsistent browser fingerprints between upload paths.

**Fix:** Extract shared browser setup into `youtube_shorts/utils/browser_utils.py`:
```python
def create_firefox_driver(profile_path=None, headless=False):
    """Create a configured Firefox WebDriver with anti-detection settings."""
    options = webdriver.FirefoxOptions()
    profile = webdriver.FirefoxProfile(profile_path) if profile_path else webdriver.FirefoxProfile()
    profile.set_preference("media.peerconnection.enabled", False)
    profile.set_preference("geo.enabled", False)
    profile.set_preference("dom.webdriver.enabled", False)
    if headless:
        options.add_argument("--headless")
    options.profile = profile
    service = Service(GeckoDriverManager().install())
    return webdriver.Firefox(service=service, options=options)
```

Both `configure_driver()` and `setup_browser()` delegate to this function.

---

### Issue 9 — README References Outdated Paths

**Severity:** Medium
**File:** `README.md`
**Problem:** README instructs users to place `config.txt`, `niche.txt`, `channels.txt` in the project root. Actual structure uses `config/` directory. Installation instructions are wrong.

**Fix:** Update README Installation section:
- `config/config.txt` (copy from `config/config.example.txt`)
- `config/niche.txt` — one niche keyword per line
- `config/channels.txt` — one YouTube channel URL per line
- `data/client_secret.json` — Google OAuth2 credentials
