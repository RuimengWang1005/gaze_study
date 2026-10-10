from __future__ import annotations

import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from bbox_teacher_audit.io_utils import load_sample_records, save_sample_record  # noqa: E402
from bbox_teacher_audit.data import select_valid_samples  # noqa: E402


def test_resume_record_is_atomic_and_not_duplicated(tmp_path):
    save_sample_record(tmp_path, "sufficiency", "sample-a", {"sample_id": "sample-a", "value": 1})
    records = load_sample_records(tmp_path, "sufficiency")
    assert records == {"sample-a": {"sample_id": "sample-a", "value": 1}}
    save_sample_record(tmp_path, "sufficiency", "sample-a", {"sample_id": "sample-a", "value": 2})
    records = load_sample_records(tmp_path, "sufficiency")
    assert len(records) == 1
    assert records["sample-a"]["value"] == 2


def test_zero_source_quota_does_not_validate_candidates():
    samples, exclusions, examined = select_valid_samples(
        {"unused": [{"this": "must not be inspected"}]},
        image_processor=None,
        merge_size=2,
        quotas={"unused": 0},
    )
    assert samples == []
    assert not exclusions
    assert examined == {"unused": 0}
