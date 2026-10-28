# 描述事项生成诊断

## 目录形态

```json
{
  "memories": 1225,
  "entries": 725,
  "entries_per_memory": 0.5918,
  "singleton_entries": 538,
  "singleton_ratio": 0.7421,
  "mean_memories_per_entry": 1.7324,
  "max_memories_per_entry": 59,
  "duplicate_pairs": 5,
  "near_duplicate_pairs": 47,
  "unverified_entries": 6,
  "admission": {
    "selected": 494,
    "selected_pending": 1,
    "contrast_reused": 1,
    "contrast_reused_pending": 0,
    "new": 755,
    "unverified": 6
  },
  "quality": {
    "audit_threshold": 0.9,
    "repair_rounds": 1,
    "repaired_batches": 20,
    "flagged_batches": 20,
    "initial_by_rule": {
      "missing": 11,
      "missing_contrast": 34,
      "same_event_split": 10
    },
    "remaining_by_rule": {
      "missing_contrast": 6,
      "same_event_split": 2
    },
    "fallback_by_rule": {},
    "unverified_entries": [
      "D000314",
      "D000315",
      "D000316",
      "D000318",
      "D000319",
      "D000322"
    ]
  },
  "prompt_version": "1.2",
  "prompt_hash": "4b7a1dd9964a"
}
```
