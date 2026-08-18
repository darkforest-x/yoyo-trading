"""Gold500 SHORT W10 classification builder source and safety contracts."""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

import pytest

from tools.build_owner_short_w10_cls_gold500 import (
    CUT1,
    CUT2,
    PURGE_BARS,
    load_confirmed_hard_negatives,
    load_gold500_short_ids,
    select_positive_rows,
    split_for_time,
)


def _write_sheet(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("box_id", "in_sample", "owner_side"))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _short_ids() -> set[str]:
    return {f"short_{index:03d}" for index in range(257)}


def _positive_rows(short_ids: set[str]) -> list[dict]:
    ordered = sorted(short_ids)
    rows = [
        {
            "sample_id": "positive_000",
            "owner_annotation_ids": ordered[:2],
            "source_owner_gold_confirmed": True,
        }
    ]
    rows.extend(
        {
            "sample_id": f"positive_{index:03d}",
            "owner_annotation_ids": [annotation_id],
            "source_owner_gold_confirmed": True,
        }
        for index, annotation_id in enumerate(ordered[2:], start=1)
    )
    assert len(rows) == 256
    return rows


def _hard_negative(event_id: str = "event-1", **overrides: object) -> dict:
    row = {
        "event_id": event_id,
        "owner_decision": "hard_negative",
        "owner_confirmed": True,
        "holdout_read": False,
        "future_data_in_causal_input": False,
        "future_data_in_training_image": False,
        "selection_future_used": False,
        "touches_owner_box_guard": False,
        "decision_time": "2025-10-31T12:00:00Z",
        "window_start_time": "2025-10-31T09:15:00Z",
        "symbol": "ETH_USDT_SWAP",
        "causal_input_sha256": f"sha256-{event_id}",
        # Future charts may have been available to the human reviewer; they were
        # not model inputs and therefore are not themselves a leakage flag.
        "owner_label_future_review_available": True,
        "future_review_only": True,
    }
    row.update(overrides)
    return row


def test_gold500_loader_filters_to_short_and_explicitly_deduplicates_ids(
    tmp_path: Path,
) -> None:
    expected = _short_ids()
    rows = [
        {"box_id": box_id, "in_sample": 1, "owner_side": "short"}
        for box_id in sorted(expected)
    ]
    # Duplicate source rows have an explicit set-like outcome: 258 matching
    # rows below still produce the 257 unique reviewed box IDs.
    rows.append({"box_id": "short_000", "in_sample": 1, "owner_side": "SHORT"})
    rows.extend(
        {"box_id": f"long_{index:03d}", "in_sample": 1, "owner_side": "long"}
        for index in range(235)
    )
    rows.extend(
        {"box_id": f"skip_{index:03d}", "in_sample": 1, "owner_side": "skip"}
        for index in range(7)
    )
    assert len(rows) == 500
    rows.extend(
        [
            {"box_id": "outside_short", "in_sample": 0, "owner_side": "short"},
            {"box_id": "inside_long", "in_sample": 0, "owner_side": "long"},
        ]
    )
    sheet = tmp_path / "review_sheet.csv"
    _write_sheet(sheet, rows)

    result = load_gold500_short_ids(sheet)

    assert result == expected
    assert len(result) == 257
    assert "outside_short" not in result


def test_positive_join_covers_257_ids_with_256_unique_rows() -> None:
    short_ids = _short_ids()
    rows = _positive_rows(short_ids)
    rows.extend(
        [
            {
                "sample_id": "unconfirmed_copy",
                "owner_annotation_ids": ["short_000"],
                "source_owner_gold_confirmed": False,
            },
            {
                "sample_id": "unrelated",
                "owner_annotation_ids": ["not-in-gold500"],
                "source_owner_gold_confirmed": True,
            },
        ]
    )

    selected = select_positive_rows(rows, short_ids)

    assert len(selected) == 256
    assert len({row["sample_id"] for row in selected}) == 256
    assert {
        annotation_id
        for row in selected
        for annotation_id in row["owner_annotation_ids"]
        if annotation_id in short_ids
    } == short_ids
    assert "unconfirmed_copy" not in {row["sample_id"] for row in selected}


def test_positive_join_rejects_missing_gold500_id() -> None:
    short_ids = _short_ids()
    rows = _positive_rows(short_ids)

    with pytest.raises(ValueError, match=r"missing=1"):
        select_positive_rows(rows[:-1], short_ids)


def test_positive_join_rejects_one_id_mapped_to_multiple_rows() -> None:
    short_ids = _short_ids()
    rows = _positive_rows(short_ids)
    rows.append(
        {
            "sample_id": "ambiguous_copy",
            "owner_annotation_ids": ["short_000"],
            "source_owner_gold_confirmed": True,
        }
    )

    with pytest.raises(ValueError, match=r"ambiguous=1"):
        select_positive_rows(rows, short_ids)


def test_positive_join_rejects_duplicate_selected_sample_id() -> None:
    short_ids = _short_ids()
    rows = _positive_rows(short_ids)
    rows[1]["sample_id"] = rows[0]["sample_id"]

    with pytest.raises(ValueError, match="duplicate selected positive sample_id"):
        select_positive_rows(rows, short_ids)


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (CUT1 - timedelta(hours=40, minutes=30), "train"),
        (CUT1 - timedelta(hours=40, minutes=29), None),
        (CUT1, None),
        (CUT1 + timedelta(hours=40, minutes=29), None),
        (CUT1 + timedelta(hours=40, minutes=30), "val"),
        (CUT2 - timedelta(hours=40, minutes=30), "val"),
        (CUT2 - timedelta(hours=40, minutes=29), None),
        (CUT2, None),
        (CUT2 + timedelta(hours=40, minutes=29), None),
        (CUT2 + timedelta(hours=40, minutes=30), "test"),
    ],
)
def test_time_split_has_open_162_bar_purge_bands(timestamp: object, expected: object) -> None:
    assert PURGE_BARS == 162
    assert split_for_time(timestamp) == expected


def test_time_split_uses_utc_and_refuses_holdout() -> None:
    assert split_for_time("2025-06-01T00:00:00") == "train"
    assert split_for_time("2025-12-15T08:00:00+08:00") == "val"
    assert split_for_time("2026-03-01T00:00:00Z") == "test"
    with pytest.raises(ValueError, match="holdout timestamp"):
        split_for_time("2026-05-04T00:00:00Z")


def test_hard_negative_loader_keeps_only_owner_confirmed_false_fires(tmp_path: Path) -> None:
    source = tmp_path / "owner_review.jsonl"
    ignored = _hard_negative(
        "not-a-negative",
        owner_decision="positive",
        owner_confirmed=False,
    )
    _write_jsonl(source, [_hard_negative(), ignored])

    rows = load_confirmed_hard_negatives([source])

    assert [row["event_id"] for row in rows] == ["event-1"]
    assert rows[0]["source_review_manifest"] == str(source)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("owner_confirmed", False),
        ("holdout_read", True),
        ("future_data_in_causal_input", True),
        ("future_data_in_training_image", True),
        ("selection_future_used", True),
        ("hard_negative_expansion_future_used", True),
        ("hard_negative_newblocks_future_used", True),
        ("touches_owner_box_guard", True),
    ],
)
def test_hard_negative_loader_rejects_unsafe_rows(
    tmp_path: Path, field: str, unsafe_value: object
) -> None:
    source = tmp_path / f"unsafe_{field}.jsonl"
    _write_jsonl(source, [_hard_negative(**{field: unsafe_value})])

    with pytest.raises(ValueError, match="unsafe Owner hard negative"):
        load_confirmed_hard_negatives([source])


def test_hard_negative_loader_rejects_decision_in_holdout(tmp_path: Path) -> None:
    source = tmp_path / "holdout.jsonl"
    _write_jsonl(
        source,
        [_hard_negative(decision_time="2026-05-04T00:00:00Z")],
    )

    with pytest.raises(ValueError, match="hard negative in holdout"):
        load_confirmed_hard_negatives([source])


def test_hard_negative_loader_rejects_duplicate_event_across_sources(tmp_path: Path) -> None:
    first = tmp_path / "review_a.jsonl"
    second = tmp_path / "review_b.jsonl"
    _write_jsonl(first, [_hard_negative("duplicate-event")])
    _write_jsonl(
        second,
        [
            _hard_negative(
                "duplicate-event",
                symbol="SOL_USDT_SWAP",
                window_start_time="2025-11-01T09:15:00Z",
                decision_time="2025-11-01T12:00:00Z",
                causal_input_sha256="different-image",
            )
        ],
    )

    with pytest.raises(ValueError, match="duplicate Owner hard negative"):
        load_confirmed_hard_negatives([first, second])
