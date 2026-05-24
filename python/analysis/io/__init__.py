"""HDF5 and repository path helpers."""

from .hdf5 import compute_channel_stat, read_dataset, read_waveform
from .inspect import get_h5_files, show_h5_structure
from .parameters import load_parameter_file, load_parameter_stack
from .paths import RunFiles, find_project_root, list_h5_files, pair_parameter_files, pair_raw_pulse_files
from .time import read_bin_time_span, read_hdf5_time_span, read_raw_pulse_event_time_mpl
from .writers import save_hdf5, save_mat_v73

__all__ = [
    "RunFiles",
    "compute_channel_stat",
    "find_project_root",
    "get_h5_files",
    "list_h5_files",
    "load_parameter_file",
    "load_parameter_stack",
    "pair_parameter_files",
    "pair_raw_pulse_files",
    "read_bin_time_span",
    "read_dataset",
    "read_hdf5_time_span",
    "read_raw_pulse_event_time_mpl",
    "read_waveform",
    "save_hdf5",
    "save_mat_v73",
    "show_h5_structure",
]
