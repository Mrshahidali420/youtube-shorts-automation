# Phase 1: Critical Crash Fixes

These bugs cause immediate crashes or prevent the package from importing at all. Fix these first.

---

## Issue 1 — Broken `__init__.py` imports

**Severity:** Critical
**File:** `youtube_shorts/__init__.py` lines 11, 15
**Problem:** Imports `setup_workspace` and `downloader` which do not exist anywhere in the project. `python -c "import youtube_shorts"` raises `ImportError`. The CI workflow that checks `import youtube_shorts` would always fail.
**Fix:** Remove lines 11 and 15:
```python
# DELETE these lines:
from .setup_workspace import setup_workspace   # line 11
from . import downloader                        # line 15
```
The actual modules are `downloader_keyword` and `downloader_channel` — these do NOT need to be re-exported from `__init__.py`.

**Verify:** `python -c "import youtube_shorts"` exits with code 0.

---

## Issue 2 — Undefined variable `msg` (NameError)

**Severity:** Critical
**File:** `youtube_shorts/uploader.py` lines 697–698
**Problem:** `log_error_to_file(f"Warning: {msg}")` references `msg` which is not defined at that point. The variable `msg` is only set later in the except blocks (lines 711–713). This raises `NameError` whenever the profile path is invalid.
**Fix:**
```python
# BEFORE (broken):
log_error_to_file(f"Warning: {msg}")

# AFTER:
log_error_to_file(f"Warning: Failed to set Firefox profile path: {e}")
```
(The enclosing except block has `except Exception as e`.)

**Verify:** Pass an invalid Firefox profile path in config — no NameError, just a warning in the log.

---

## Issue 3 — Invalid `[...]` literal (TypeError)

**Severity:** Critical
**File:** `youtube_shorts/uploader.py` line 1391
**Problem:** `any(metadata.get("target_playlist") for metadata in [...])` — the literal `[...]` is an Ellipsis inside a list, not actual metadata. This raises `TypeError: argument of type 'ellipsis' is not iterable` at runtime.
**Fix:** Replace `[...]` with the actual list of loaded metadata dicts used in the upload loop. Likely should be:
```python
any(m.get("target_playlist") for m in videos_to_upload)
```
Where `videos_to_upload` is the list being iterated in the surrounding upload loop. Confirm variable name by reading the surrounding context.

**Verify:** Run uploader with at least one video that has `target_playlist` set — no TypeError.

---

## Issue 4 — Unreachable `time.sleep()` after `break`

**Severity:** Low
**File:** `youtube_shorts/uploader.py` line 500
**Problem:** `if not next_page_token: break; time.sleep(0.5)` — the sleep is on the same line after `break` and will never execute.
**Fix:**
```python
# BEFORE:
if not next_page_token: break; time.sleep(0.5)

# AFTER:
time.sleep(0.5)
if not next_page_token:
    break
```

**Verify:** Code review only — no functional regression expected.

---

## Verification Checklist

- [ ] `python -c "import youtube_shorts"` → exits 0
- [ ] `python -m youtube_shorts.uploader` → no NameError on startup
- [ ] Run uploader with invalid Firefox profile → warning in log, no crash
- [ ] Run uploader with a video that has `target_playlist` → no TypeError
