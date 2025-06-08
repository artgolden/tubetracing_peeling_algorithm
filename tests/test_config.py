import yaml
import pytest
from dataclasses import asdict
from argparse import Namespace
from pipeline_config import (
    load_config,
    merge_cli_overrides,
    GlobalConfig,
    TimeSeriesConfig,
    OnionRangeConfig,
    SURFACE_MODES,
    WBNS_THRESHOLDS
)

# The example YAML from before, using list-of-mappings for time_series
EXAMPLE_YAML = """
global:
  log_level: DEBUG
  force_cpu: false
  include_patterns:
    - "*.tif"
  exclude_patterns: []
  create_subfolders: true

  reuse_peeling: false
  only_first_timepoint: false
  voxel_size: [1, 1, 1]
  surface_detection_mode: wbns
  wbns_threshold: otsu
  do_save_mask: true

time_series:
  - id: "timelapseID-20241008-143038_SPC-0001"
    only_first_timepoint: true
    voxel_size: [2.34, 0.586, 0.586]
    surface_detection_mode: tubetracing
    wbns_threshold: mean
    onion_z_range:
      start: 0
      end: 15
    onion_layer_ranges:
      - start: 0
        end: 5
      - start: 6
        end: 10
      - start: 11
        end: 15

  - id: "timelapseID-20241009-154500_SPC-0002"
    reuse_peeling: true
    do_save_mask: false
    do_save_zmax_projection: true
    mask_dilation_radius: 3
"""

# A minimal global‐only YAML for CLI-override tests
MINIMAL_GLOBAL_YAML = """
global:
  log_level: INFO
  force_cpu: true
"""

# An invalid list‐style YAML (missing 'id')
INVALID_LIST_ENTRY_YAML = """
time_series:
  - reuse_peeling: true
"""

def test_load_example_and_roundtrip(tmp_path):
    """Load the example YAML and round-trip dump preserves keys."""
    cfg_file = tmp_path / "example.yaml"
    cfg_file.write_text(EXAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, GlobalConfig)

    # Check some global fields
    assert cfg.log_level == "DEBUG"
    assert cfg.force_cpu is False
    assert cfg.include_patterns == ["*.tif"]
    assert cfg.create_subfolders is True
    assert cfg.reuse_peeling is False
    assert cfg.surface_detection_mode == "wbns"
    assert cfg.wbns_threshold == "otsu"
    assert cfg.do_save_mask is True

    # Round-trip dump
    data = asdict(cfg)
    out = tmp_path / "out.yaml"
    with out.open("w") as f:
        yaml.safe_dump(data, f)
    txt = out.read_text()
    # Ensure keys are present in the dumped YAML
    assert "log_level" in txt
    assert "time_series_overrides" in txt

def test_series_overrides_for_example(tmp_path):
    """Verify per‐series settings from the example list‐of‐mappings YAML."""
    cfg_file = tmp_path / "example.yaml"
    cfg_file.write_text(EXAMPLE_YAML)
    cfg = load_config(str(cfg_file))

    # First series: timelapseID-20241008-143038_SPC-0001
    ts1 = cfg.get_series_config("timelapseID-20241008-143038_SPC-0001")
    assert isinstance(ts1, TimeSeriesConfig)
    assert ts1.only_first_timepoint is True
    assert ts1.voxel_size == (2.34, 0.586, 0.586)
    assert ts1.surface_detection_mode == "tubetracing"
    assert ts1.wbns_threshold == "mean"

    # Onion ranges
    assert isinstance(ts1.onion_z_range, OnionRangeConfig)
    assert (ts1.onion_z_range.start, ts1.onion_z_range.end) == (0, 15)
    assert isinstance(ts1.onion_layer_ranges, list)
    layer_ranges = [(r.start, r.end) for r in ts1.onion_layer_ranges]
    assert layer_ranges == [(0, 5), (6, 10), (11, 15)]

    # Second series: timelapseID-20241009-154500_SPC-0002
    ts2 = cfg.get_series_config("timelapseID-20241009-154500_SPC-0002")
    assert ts2.reuse_peeling is True
    assert ts2.do_save_mask is False
    assert ts2.do_save_zmax_projection is True
    assert ts2.mask_dilation_radius == 3
    # Falls back to globals for others
    assert ts2.only_first_timepoint is False
    assert ts2.surface_detection_mode == "wbns"

    # Unknown series → all global defaults
    ts_unknown = cfg.get_series_config("does-not-exist")
    assert ts_unknown.reuse_peeling is False
    assert ts_unknown.surface_detection_mode == "wbns"

def test_invalid_list_style_raises(tmp_path):
    """A list entry without 'id' must trigger a ValueError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(INVALID_LIST_ENTRY_YAML)
    with pytest.raises(ValueError) as exc:
        _ = load_config(str(bad))
    assert "must include an 'id' field" in str(exc.value)

def test_merge_cli_overrides(tmp_path):
    """Ensure merge_cli_overrides respects provided Namespace values."""
    cfg_file = tmp_path / "minimal.yaml"
    cfg_file.write_text(MINIMAL_GLOBAL_YAML)
    cfg = load_config(str(cfg_file))

    args = Namespace(
        log_level="WARNING",
        force_cpu=False,              # should *not* override (explicitly False)
        include_patterns=["a", "b"],
        exclude_patterns=None,        # skip override
        create_subfolders=True
    )
    merge_cli_overrides(cfg, args)

    assert cfg.log_level == "WARNING"
    # original YAML set force_cpu True, CLI False does not override
    assert cfg.force_cpu is True
    assert cfg.include_patterns == ["a", "b"]
    assert cfg.create_subfolders is True