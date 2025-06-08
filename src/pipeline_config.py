import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union

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
    voxel_size: Optional[Tuple[int, int, int]] = None
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
        for attr, val in vars(self).items():
            if attr in ('reuse_peeling','only_first_timepoint','load_surface_voxels',
                        'add_series_id_to_filename','do_inverse_peeling',
                        'do_prune_voxels_after_wbns','do_remove_outliers_after_wbns',
                        'do_cylindrical_cartography','do_distortion_maps',
                        'do_save_points','do_save_peeled_volume','do_save_zmax_projection',
                        'do_save_unpeeled_zmax_projection','do_save_mask','do_save_wbns_output',
                        'do_save_distortion_map_vis','do_onion_z_stack'):
                if val is not None and not isinstance(val, bool):
                    raise ValueError(f"{attr} must be bool in TimeSeriesConfig, got {type(val)}")
        # voxel_size
        if self.voxel_size is not None:
            if (not isinstance(self.voxel_size, (list, tuple)) or len(self.voxel_size) != 3 
                or not all(isinstance(x, int) and x>0 for x in self.voxel_size)):
                raise ValueError(f"voxel_size must be 3 positive ints, got {self.voxel_size}")
        # surface_detection_mode
        if self.surface_detection_mode is not None and self.surface_detection_mode not in SURFACE_MODES:
            raise ValueError(f"surface_detection_mode must be one of {SURFACE_MODES}, got {self.surface_detection_mode}")
        # wbns_threshold
        if self.wbns_threshold not in WBNS_THRESHOLDS:
            raise ValueError(f"wbns_threshold must be one of {WBNS_THRESHOLDS}, got {self.wbns_threshold}")
        # mask_dilation_radius
        if self.mask_dilation_radius is not None and not isinstance(self.mask_dilation_radius, int):
            raise ValueError(f"mask_dilation_radius must be int, got {type(self.mask_dilation_radius)}")
        # onion ranges validated during init of OnionRangeConfig

@dataclass
class GlobalConfig:
    # CLI-overridable
    log_level: str = 'INFO'
    force_cpu: bool = False
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    create_subfolders: bool = False
    # Defaults
    reuse_peeling: bool = False
    only_first_timepoint: bool = False
    load_surface_voxels: bool = False
    add_series_id_to_filename: bool = False
    voxel_size: Tuple[int, int, int] = (1, 1, 1)
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
            or not all(isinstance(x, int) and x>0 for x in self.voxel_size)):
            raise ValueError(f"voxel_size must be 3 positive ints, got {self.voxel_size}")
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
        for series_id, ts_conf in self.time_series_overrides.items():
            if not isinstance(series_id, str):
                raise ValueError(f"Time series key must be str, got {type(series_id)}")
            ts_conf.validate()

    def get_series_config(self, series_id: str) -> 'EffectiveConfig':
        """
        Merge global defaults with any overrides for a given series.
        """
        base = self
        override = self.time_series_overrides.get(series_id, None)
        eff = EffectiveConfig()
        # copy all global attributes
        for field_name in vars(base):
            if field_name == 'time_series_overrides':
                continue
            setattr(eff, field_name, getattr(base, field_name))
        # apply overrides when not None
        if override:
            for key, val in vars(override).items():
                if val is not None:
                    setattr(eff, key, val)
        return eff

@dataclass
class EffectiveConfig(GlobalConfig, TimeSeriesConfig):
    """
    Final per-series config with all fields resolved.
    Inherits all fields from GlobalConfig and TimeSeriesConfig.
    """
    pass

def load_config(yaml_path: str) -> GlobalConfig:
    """
    Load global and per-series pipeline configuration from a YAML file with validation.
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    # Build GlobalConfig
    global_map = raw.get('global', {})
    gc = GlobalConfig(**{k: v for k, v in global_map.items() if k in GlobalConfig.__annotations__})

    # Per-series
    ts_map = raw.get('time_series', {})
    for sid, params in ts_map.items():
        # filter keys
        filtered = {}
        for k, v in params.items():
            if k == 'onion_z_range':
                filtered[k] = OnionRangeConfig(**v)
            elif k == 'onion_layer_ranges':
                filtered[k] = [OnionRangeConfig(**r) for r in v]
            else:
                filtered[k] = v
        tsc = TimeSeriesConfig(**{k: filtered[k] for k in filtered if k in TimeSeriesConfig.__annotations__})
        gc.time_series_overrides[sid] = tsc

    # Validate
    gc.validate()
    return gc


def merge_cli_overrides(config: GlobalConfig, args: Any) -> None:
    """
    Override config fields with CLI args when provided.
    """
    if hasattr(args, 'log_level') and args.log_level is not None:
        config.log_level = args.log_level
    if hasattr(args, 'force_cpu') and args.force_cpu:
        config.force_cpu = True
    if hasattr(args, 'include_patterns') and args.include_patterns:
        config.include_patterns = args.include_patterns
    if hasattr(args, 'exclude_patterns') and args.exclude_patterns:
        config.exclude_patterns = args.exclude_patterns
    if hasattr(args, 'create_subfolders') and args.create_subfolders:
        config.create_subfolders = True
    # after CLI overrides, re-validate
    config.validate()