# Interval semantics audit

- **local_8768**: inclusive — core_end_bar is stored as the last included bar; HTML shade runs center(a)→center(b)
- **labelstudio_keypoints**: inclusive — START/END snap to bar index, core_end_bar = local_start + end_i
- **legacy_yolo_box**: pixel_box — bars whose candle centers fall in [x_min, x_max]
- **review_sheet_csv**: inclusive local bar_b0/bar_b1 on the old image; win_mode=end_incl; global restore needs window start
- **legacy_manifest_bar_range**: inclusive — V3 box_start_bar/box_end_bar and R1 core_global
- **human_gold_owner_box**: inclusive — source_owner_global [lo, hi], source_owner_bars = hi-lo+1

8768 页面「起点 22 — 终点 26」在旧工具里是**闭区间** 22…26（5 根）。新合同解释为半开区间 [22, 26) = 22,23,24,25。迁移时按代码语义（闭区间）转半开，不把旧 22–26 静默当成 4 根。
