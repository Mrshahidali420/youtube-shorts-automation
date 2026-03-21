# Audit & Remediation Tasks

Full audit of the codebase completed 2026-03-21. 37 issues found across security, reliability, architecture, and code quality. Organized into 5 dependency-ordered phases.

## Files

| File | Contents |
|------|----------|
| `audit-phase1-crashes.md` | 4 critical crash fixes (broken imports, undefined vars, invalid literals) |
| `audit-phase2-security.md` | Pickle vulnerability fix, auth consolidation, MinimalConstants unification, config merge |
| `audit-phase3-reliability.md` | Browser reuse, file locking, quota bug, crash recovery, duplicate detection |
| `audit-phase4-quality.md` | Test framework setup + medium quality fixes (locators, timezone, constants, logging) |
| `audit-phase5-polish.md` | Low-priority polish (version pins, dead code, upload polling speed) |

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
```

Phase 4A (test framework) can start in parallel with Phase 3.

## Issue Count by Severity

| Severity | Count |
|----------|-------|
| Critical | 6 |
| High | 13 |
| Medium | 13 |
| Low | 5 |
| **Total** | **37** |

## Top 5 Highest Impact Fixes

1. **Pickle deserialization vulnerability** (Phase 2) — security critical
2. **Broken `__init__.py`** (Phase 1) — package can't import
3. **New browser per video** (Phase 3) — massive resource waste
4. **Browser crash kills entire run** (Phase 3) — lost work
5. **Auth duplicated in 5 files** (Phase 2) — ~500 lines removable
