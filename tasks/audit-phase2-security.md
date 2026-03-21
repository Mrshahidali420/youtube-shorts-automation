# Phase 2: Security & Auth Consolidation

Eliminate the pickle vulnerability, consolidate 5 duplicated auth functions into one, unify 17 MinimalConstants copies, and merge dual config systems.

---

## Issue 1 — Insecure `pickle.load` (Arbitrary Code Execution)

**Severity:** Critical / Security
**Files & Lines:**
- `youtube_shorts/downloader_keyword.py:249`
- `youtube_shorts/downloader_channel.py:2697`
- `youtube_shorts/uploader.py:555`
- `youtube_shorts/performance_tracker.py:233`
- `youtube_shorts/utils/auth_utils.py:143`

**Problem:** All 5 files use `pickle.load(token_file)` to deserialize OAuth credentials. Pickle can execute arbitrary Python code during deserialization. If `token.pickle` is replaced by an attacker (or accidentally corrupted), it can execute malicious code with the permissions of the running process.

**Fix — Replace with Google's built-in JSON serialization:**
```python
# BEFORE (insecure):
import pickle
with open(TOKEN_FILE, 'rb') as token:
    creds = pickle.load(token)

# AFTER (secure):
from google.oauth2.credentials import Credentials
creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
```

```python
# BEFORE (saving):
with open(TOKEN_FILE, 'wb') as token:
    pickle.dump(creds, token)

# AFTER (saving):
with open(TOKEN_FILE, 'w') as token:
    token.write(creds.to_json())
```

**Migration (one-time):** If `token.pickle` exists but `token.json` does not, load via pickle once, save as JSON, then delete the pickle file:
```python
import os, pickle
pickle_path = TOKEN_FILE.replace('.json', '.pickle')
if os.path.exists(pickle_path) and not os.path.exists(TOKEN_FILE):
    with open(pickle_path, 'rb') as f:
        old_creds = pickle.load(f)
    with open(TOKEN_FILE, 'w') as f:
        f.write(old_creds.to_json())
    os.remove(pickle_path)
```

Add this migration block into `auth_utils.get_authenticated_service()` so it runs automatically on first use.

**Verify:** `grep -r "pickle.load" youtube_shorts/` returns 0 results.

---

## Issue 2 — Wrong Auth File Paths in `downloader_keyword.py`

**Severity:** High
**File:** `youtube_shorts/downloader_keyword.py` lines 237–238
**Problem:** Auth paths are hardcoded relative to the script directory (inside the package), not using `constants.CLIENT_SECRETS_FILE` and `constants.TOKEN_FILE` which point to the correct `data/` directory. Authentication will fail or create orphan token files in the wrong location.

**Fix:**
```python
# BEFORE:
CLIENT_SECRETS_FILE = os.path.join(script_dir, 'client_secret.json')
TOKEN_FILE = os.path.join(script_dir, 'token.json')

# AFTER:
from youtube_shorts.utils.constants import CLIENT_SECRETS_FILE, TOKEN_FILE
```

---

## Issue 3 — Duplicated `get_authenticated_service()` in 5 Files

**Severity:** High
**Files:**
- `youtube_shorts/downloader_keyword.py:224` — has pickle vulnerability, uses wrong paths
- `youtube_shorts/downloader_channel.py:2690+` — has pickle vulnerability
- `youtube_shorts/uploader.py:540` — has pickle vulnerability
- `youtube_shorts/performance_tracker.py:222` — has pickle vulnerability
- `youtube_shorts/utils/auth_utils.py:101` — canonical version, NOT used by the main scripts

**Problem:** `auth_utils.py` already has the correct, well-structured implementation but none of the 4 main entry points import from it. Each has its own copy with subtle differences in paths and scopes. Any bug fix must be applied 5 times. Total ~500 lines of duplicated code.

**Fix:**
1. Fix the canonical `auth_utils.get_authenticated_service()` — replace its pickle usage, ensure it uses `constants.CLIENT_SECRETS_FILE` and `constants.TOKEN_FILE`
2. In each of the 4 main scripts, delete the local `get_authenticated_service()` and replace with:
```python
from youtube_shorts.utils.auth_utils import get_authenticated_service
```

**Verify:** `grep -rn "def get_authenticated_service" youtube_shorts/` returns exactly 1 result (in `auth_utils.py`).

---

## Issue 4 — 17 Duplicated `MinimalConstants` Classes

**Severity:** High
**Files:** Every module in `youtube_shorts/utils/` and all 4 main entry point scripts each define their own `class MinimalConstants` in an `except ImportError` block. Each defines a slightly different subset of constants with different default values.

**Problem:** When constants drift between copies, behavior silently diverges across modules. 17 copies to maintain.

**Fix:**
1. Create `youtube_shorts/utils/_fallback_constants.py`:
```python
class MinimalConstants:
    """Fallback constants used when constants.py cannot be imported."""
    # Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CONFIG_DIR = os.path.join(BASE_DIR, "config")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    # Files
    CLIENT_SECRETS_FILE = os.path.join(DATA_DIR, "client_secret.json")
    TOKEN_FILE = os.path.join(DATA_DIR, "token.json")
    SHORTS_DATA_FILE = os.path.join(DATA_DIR, "shorts_data.xlsx")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.txt")
    # ... all other fields from constants.py
```

2. Replace every inline `class MinimalConstants:` block across all 17 files with:
```python
from youtube_shorts.utils._fallback_constants import MinimalConstants
```

**Verify:** `grep -rn "class MinimalConstants" youtube_shorts/` returns exactly 1 result (in `_fallback_constants.py`).

---

## Issue 5 — Dual Config Systems (`config_utils.py` vs `secure_config.py`)

**Severity:** High
**Files:**
- `youtube_shorts/utils/config_utils.py` — `load_config()` with typed defaults and int/float/bool coercion
- `youtube_shorts/utils/secure_config.py` — `load_config()` returning all strings, with env-var overlay and sensitive-key masking

**Problem:** Both parse the same `config.txt` but with different semantics. Callers must know which to use. Type coercion done in `config_utils` is unavailable in `secure_config` callers. Duplicate logic.

**Fix:** Merge into one loader:
1. Keep `config_utils.load_config()` as primary — it has type coercion and DEFAULT_CONFIG
2. Add env-var overlay from `secure_config` as a parameter:
```python
def load_config(config_file=None, check_env=False):
    config = _load_from_file(config_file)
    if check_env:
        for key in config:
            env_val = os.environ.get(f"YT_SHORTS_{key}")
            if env_val is not None:
                config[key] = env_val
    return config
```
3. Move `secure_config`'s sensitive-key masking into a separate `mask_config(config)` utility function
4. Update all callers of `secure_config.load_config()` to use `config_utils.load_config(check_env=True)`
5. Keep `secure_config.py` but have it delegate to `config_utils`

**Verify:** All callers get correctly typed values; env vars like `YT_SHORTS_API_KEY` still override config.txt values.
