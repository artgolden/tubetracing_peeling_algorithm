import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union

# Allowed values
SURFACE_MODES = {"wbns", "tubetracing"}
WBNS_THRESHOLDS = {None, 'otsu', 'yen', 'li', 'isodata', 'minimum', 'triangle', 'mean', 'sauvola'}

@dataclass
class OnionRangeConfig:
    start: int
    end: int

    def __post_init__(self):
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ValueError(f"OnionRangeConfig start/end must be ints, got {type(self.start)}/{type(self.end)}")
        if self.start > self.end:
            raise ValueError(f"OnionRangeConfig start ({self.start}) must be <= end ({self.end})")

@dataclass
class TimeSeriesConfig:
    reuse_peeling: Optional[bool] = None
    only_first_timepoint: Optional[bool] = None
    load_surface_voxels: Optional[bool] = None
    add_series_id_to_filename: Optional[bool] = None
    voxel_size: Optional[Tuple[float, float, float]] = None
    surface_detection_mode: Optional[str] = None
    wbns_threshold: Optional[str] = None
    do_inverse_peeling: Optional[bool] = None
    do_prune_voxels_after_wbns: Optional[bool] = None
    do_remove_outliers_after_wbns: Optional[bool] = None
    do_cylindrical_cartography: Optional[bool] = None
    do_distortion_maps: Optional[bool] = None
    do_save_points: Optional[bool] = None
    do_save_peeled_volume: Optional[bool] = None
    do_save_zmax_projection: Optional[bool] = None
    do_save_unpeeled_zmax_projection: Optional[bool] = None
    do_save_mask: Optional[bool] = None
    do_save_wbns_output: Optional[bool] = None
    do_save_distortion_map_vis: Optional[bool] = None
    mask_dilation_radius: Optional[int] = None
    do_onion_z_stack: Optional[bool] = None
    onion_z_range: Optional[OnionRangeConfig] = None
    onion_layer_ranges: Optional[List[OnionRangeConfig]] = None

    def validate(self):
        # booleans
        bool_fields = {
            'reuse_peeling','only_first_timepoint','load_surface_voxels',
            'add_series_id_to_filename','do_inverse_peeling',
            'do_prune_voxels_after_wbns','do_remove_outliers_after_wbns',
            'do_cylindrical_cartography','do_distortion_maps',
            'do_save_points','do_save_peeled_volume','do_save_zmax_projection',
            'do_save_unpeeled_zmax_projection','do_save_mask','do_save_wbns_output',
            'do_save_distortion_map_vis','do_onion_z_stack'
        }
        for attr in bool_fields:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, bool):
                raise ValueError(f"{attr} must be bool in TimeSeriesConfig, got {type(val)}")
        # voxel_size
        if self.voxel_size is not None:
            if (not isinstance(self.voxel_size, (list, tuple)) or len(self.voxel_size) != 3 
                or not all((isinstance(x, float) or isinstance(x, int)) and x > 0 for x in self.voxel_size)):
                raise ValueError(f"voxel_size must be 3 positive float, got {self.voxel_size}")
        # surface_detection_mode
        if self.surface_detection_mode is not None and self.surface_detection_mode not in SURFACE_MODES:
            raise ValueError(f"surface_detection_mode must be one of {SURFACE_MODES}, got {self.surface_detection_mode}")
        # wbns_threshold
        if self.wbns_threshold not in WBNS_THRESHOLDS:
            raise ValueError(f"wbns_threshold must be one of {WBNS_THRESHOLDS}, got {self.wbns_threshold}")
        # mask_dilation_radius
        if self.mask_dilation_radius is not None and not isinstance(self.mask_dilation_radius, int):
            raise ValueError(f"mask_dilation_radius must be int, got {type(self.mask_dilation_radius)}")
        # onion ranges
        if self.onion_z_range is not None and not isinstance(self.onion_z_range, OnionRangeConfig):
            raise ValueError("onion_z_range must be OnionRangeConfig instance or None")
        if self.onion_layer_ranges is not None:
            if not isinstance(self.onion_layer_ranges, list):
                raise ValueError("onion_layer_ranges must be list of OnionRangeConfig instances or None")
            for r in self.onion_layer_ranges:
                if not isinstance(r, OnionRangeConfig):
                    raise ValueError("Each entry in onion_layer_ranges must be OnionRangeConfig instance")

@dataclass
class GlobalConfig:
    # CLI-overridable
    log_level: str = 'INFO'
    force_cpu: bool = False
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    create_subfolders: bool = False
    # Defaults for TimeSeriesConfig fields
    reuse_peeling: bool = False
    only_first_timepoint: bool = False
    load_surface_voxels: bool = False
    add_series_id_to_filename: bool = False
    voxel_size: Tuple[float, float, float] = (2.34, 0.586, 0.586)
    surface_detection_mode: str = "wbns"
    wbns_threshold: Optional[str] = None
    do_inverse_peeling: bool = False
    do_prune_voxels_after_wbns: bool = True
    do_remove_outliers_after_wbns: bool = True
    do_cylindrical_cartography: bool = False
    do_distortion_maps: bool = False
    do_save_points: bool = True
    do_save_peeled_volume: bool = True
    do_save_zmax_projection: bool = True
    do_save_unpeeled_zmax_projection: bool = False
    do_save_mask: bool = True
    do_save_wbns_output: bool = True
    do_save_distortion_map_vis: bool = True
    mask_dilation_radius: int = 0
    do_onion_z_stack: bool = False
    onion_z_range: OnionRangeConfig = OnionRangeConfig(0, 0)
    onion_layer_ranges: List[OnionRangeConfig] = field(default_factory=list)
    # Per-series overrides
    time_series_overrides: Dict[str, TimeSeriesConfig] = field(default_factory=dict)

    def validate(self):
        # booleans
        for attr in ['force_cpu','create_subfolders','reuse_peeling','only_first_timepoint',
                     'load_surface_voxels','add_series_id_to_filename','do_inverse_peeling',
                     'do_prune_voxels_after_wbns','do_remove_outliers_after_wbns',
                     'do_cylindrical_cartography','do_distortion_maps','do_save_points',
                     'do_save_peeled_volume','do_save_zmax_projection','do_save_unpeeled_zmax_projection',
                     'do_save_mask','do_save_wbns_output','do_save_distortion_map_vis','do_onion_z_stack']:
            val = getattr(self, attr)
            if not isinstance(val, bool):
                raise ValueError(f"{attr} must be bool in GlobalConfig, got {type(val)}")
        # log_level
        if not isinstance(self.log_level, str):
            raise ValueError(f"log_level must be str, got {type(self.log_level)}")
        # include/exclude patterns
        for name in ('include_patterns','exclude_patterns'):
            lst = getattr(self, name)
            if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
                raise ValueError(f"{name} must be list of str, got {lst}")
        # voxel_size
        if (not isinstance(self.voxel_size, (list, tuple)) or len(self.voxel_size) != 3 
            or not all((isinstance(x, float) or isinstance(x, int)) and x > 0 for x in self.voxel_size)):
            raise ValueError(f"voxel_size must be 3 positive float, got {self.voxel_size}")
        # surface_detection_mode
        if self.surface_detection_mode not in SURFACE_MODES:
            raise ValueError(f"surface_detection_mode must be one of {SURFACE_MODES}, got {self.surface_detection_mode}")
        # wbns_threshold
        if self.wbns_threshold not in WBNS_THRESHOLDS:
            raise ValueError(f"wbns_threshold must be one of {WBNS_THRESHOLDS}, got {self.wbns_threshold}")
        # mask_dilation_radius
        if not isinstance(self.mask_dilation_radius, int):
            raise ValueError(f"mask_dilation_radius must be int, got {type(self.mask_dilation_radius)}")
        # onion ranges
        if not isinstance(self.onion_z_range, OnionRangeConfig):
            raise ValueError("onion_z_range must be OnionRangeConfig instance")
        if not isinstance(self.onion_layer_ranges, list):
            raise ValueError("onion_layer_ranges must be list of OnionRangeConfig instances")
        for r in self.onion_layer_ranges:
            if not isinstance(r, OnionRangeConfig):
                raise ValueError("Each entry in onion_layer_ranges must be OnionRangeConfig")
        # validate per-series overrides
        for sid, ts_conf in self.time_series_overrides.items():
            if not isinstance(sid, str):
                raise ValueError(f"Time series key must be str, got {type(sid)}")
            ts_conf.validate()

    def get_series_config(self, series_id: Union[str, Any]) -> TimeSeriesConfig:
        """
        Return a TimeSeriesConfig with all fields filled: use per-series override when present,
        otherwise take from global defaults. Accepts any series_id and normalizes it to string.
        """
        sid = str(series_id)
        override = self.time_series_overrides.get(sid)
        merged_values: Dict[str, Any] = {}
        for field_name in TimeSeriesConfig.__annotations__:
            if override:
                val = getattr(override, field_name)
                if val is not None:
                    merged_values[field_name] = val
                    continue
            # fallback to global attr
            if hasattr(self, field_name):
                merged_values[field_name] = getattr(self, field_name)
            else:
                merged_values[field_name] = None
        merged = TimeSeriesConfig(**merged_values)
        merged.validate()
        return merged


def load_config(yaml_path: str) -> GlobalConfig:
    """
    Load global and per-series pipeline configuration from a YAML file with validation.

    Supports two forms for 'time_series':
    1) Mapping: keys are series IDs (strings), even with hyphens, mapping to config dicts.
    2) List of mappings: each entry must have an 'id' field with the series ID.
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    # Build GlobalConfig from 'global' section
    global_map = raw.get('global', {}) or {}
    # handle onion ranges
    if 'onion_z_range' in global_map:
        global_map['onion_z_range'] = OnionRangeConfig(**global_map['onion_z_range'])
    if 'onion_layer_ranges' in global_map:
        global_map['onion_layer_ranges'] = [OnionRangeConfig(**r) for r in global_map['onion_layer_ranges']]
    if 'voxel_size' in global_map and global_map['voxel_size'] is not None:
        global_map['voxel_size'] = tuple(global_map['voxel_size'])
    gc = GlobalConfig(**{k: v for k, v in global_map.items() if k in GlobalConfig.__annotations__})

    # Per-series: support dict or list
    ts_section = raw.get('time_series', {}) or {}
    if isinstance(ts_section, dict):
        ts_items = ts_section.items()
    elif isinstance(ts_section, list):
        ts_items = []
        for entry in ts_section:
            if not isinstance(entry, dict):
                raise ValueError(f"Each time_series entry must be a mapping, got {type(entry)}")
            if 'id' not in entry:
                raise ValueError("Each time_series entry must include an 'id' field.")
            sid = str(entry['id'])
            params = {k: v for k, v in entry.items() if k != 'id'}
            ts_items.append((sid, params))
    else:
        raise ValueError("time_series section must be a mapping or a list of mappings")

    for sid_raw, params in ts_items:
        sid = str(sid_raw)
        # convert any nested onion ranges
        if 'onion_z_range' in params:
            params['onion_z_range'] = OnionRangeConfig(**params['onion_z_range'])
        if 'onion_layer_ranges' in params:
            params['onion_layer_ranges'] = [OnionRangeConfig(**r) for r in params['onion_layer_ranges']]
        if 'voxel_size' in params and params['voxel_size'] is not None:
            params['voxel_size'] = tuple(params['voxel_size'])
        # filter to known TimeSeriesConfig keys
        filtered = {k: v for k, v in params.items() if k in TimeSeriesConfig.__annotations__}
        ts_conf = TimeSeriesConfig(**filtered)
        gc.time_series_overrides[sid] = ts_conf

    gc.validate()
    return gc


def merge_cli_overrides(config: GlobalConfig, args: Any) -> None:
    """
    Override config fields with CLI args when provided.
    """
    for attr in ('log_level','force_cpu','include_patterns','exclude_patterns','create_subfolders'):
        if hasattr(args, attr):
            val = getattr(args, attr)
            if val is not None and val is not False:
                setattr(config, attr, val)
    config.validate()
