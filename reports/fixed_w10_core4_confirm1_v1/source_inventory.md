# Source inventory

- yoyo HEAD `41e183cdba68a89fdef9d7f743837355a2a55612`
- fable HEAD `9170987dcfc032856aa6661e72f074adbde38b5a`

| source | records |
|---|---:|
| `easy_negatives_r1` | 1343 |
| `human_gold_owner_box` | 1345 |
| `labelstudio_yoyo_gold_v1` | 1 |
| `local_8768` | 24 |
| `owner_review_packs` | 1200 |
| `review_sheet_csv` | 2525 |
| `rule_verified_r1` | 1345 |
| `rule_verified_v3` | 1345 |
| `v3_2_reviewed` | 221 |
| `v3_dispute_final_labels` | 243 |

## Interval semantics
- **local_8768**: inclusive — core_end_bar is stored as the last included bar; HTML shade runs center(a)→center(b)
- **labelstudio_keypoints**: inclusive — START/END snap to bar index, core_end_bar = local_start + end_i
- **legacy_yolo_box**: pixel_box — bars whose candle centers fall in [x_min, x_max]
- **review_sheet_csv**: inclusive local bar_b0/bar_b1 on the old image; win_mode=end_incl; global restore needs window start
- **legacy_manifest_bar_range**: inclusive — V3 box_start_bar/box_end_bar and R1 core_global
- **human_gold_owner_box**: inclusive — source_owner_global [lo, hi], source_owner_bars = hi-lo+1

## 8768 annotator
- **script**: `tools/serve_gold_annotate.py`
- **port**: `8768`
- **save_path**: `datasets/gold_v1.jsonl`
- **storage**: `JSONL on disk, not localStorage, not sqlite`
- **decision**: `local_end_bar == decision_bar (last visible local bar, inclusive)`
- **old_window**: `W30 causal local + context 50 pre / 120 post`
- **renderer**: `yoyo.layers.l1_detection.render 1280x742 plus gold_render footer on local`
