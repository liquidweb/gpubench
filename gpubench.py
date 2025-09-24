#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GPUBench - A Performance Benchmarking Tool for AI/ML Servers
#
# Copyright (C) 2024 Liquid Web, LLC <deveng@liquidweb.com>
# Copyright (C) 2024 Ryan MacDonald <rmacdonald@liquidweb.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import time
import json
import argparse
import platform
import subprocess
import logging
import textwrap
import threading
import hashlib
import gzip
import importlib
import shutil
import socket
from datetime import datetime, timezone
import numpy as np
import psutil
import GPUtil
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import multiprocessing as mp


logger = logging.getLogger(__name__)


def configure_logging():
    """Configure project-wide logging for consistent output."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root_logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)


def parse_float(value):
    """Safely parse a value into a float."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def run_nvidia_smi_query(query_fields):
    """Query nvidia-smi for additional GPU telemetry information."""
    if not is_command_available('nvidia-smi'):
        return None, "`nvidia-smi` command is not available."

    command = [
        'nvidia-smi',
        f"--query-gpu={','.join(query_fields)}",
        '--format=csv,noheader,nounits'
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        telemetry = []
        for line in lines:
            values = [value.strip() for value in line.split(',')]
            telemetry.append(dict(zip(query_fields, values)))
        return telemetry, None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        message = f"Failed to query nvidia-smi: {exc}"
        logger.warning(message)
        return None, message


def is_command_available(command):
    """Return True if the given command is available on PATH."""
    return shutil.which(command) is not None


def check_python_package(module_name):
    """Return True if the given Python package can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_cuda_toolkit(torch_module):
    """Check if the CUDA toolkit or CUDA-enabled PyTorch runtime is available."""
    if is_command_available('nvcc'):
        return True

    cuda_version = getattr(torch_module.version, 'cuda', None)
    if cuda_version:
        return True

    return False


def run_preflight_checks(args, run_flags, gpu_available):
    """Run dependency checks and update execution flags as needed."""
    logger.info("Running pre-flight dependency checks...")

    updated_flags = run_flags.copy()
    skip_reasons = {}

    # Disk benchmarking requires fio.
    if run_flags.get('disk_io'):
        if not is_command_available('fio'):
            logger.warning(
                "`fio` is not installed or not on PATH. Disk I/O benchmarks will be skipped. "
                "Install it via `sudo apt-get install -y fio` or build from https://github.com/axboe/fio."
            )
            updated_flags['disk_io'] = False
            skip_reasons['disk_io'] = "Missing `fio`; install via `sudo apt-get install -y fio` or build from source."

    # GPU logging and GPU introspection need nvidia-smi.
    if run_flags.get('gpu_benchmarks') or args.log_gpu:
        if not is_command_available('nvidia-smi'):
            logger.warning(
                "`nvidia-smi` was not detected. GPU monitoring and some GPU benchmarks may fail. "
                "Install NVIDIA drivers and the CUDA toolkit from https://developer.nvidia.com/cuda-downloads."
            )
            if args.log_gpu:
                logger.warning("Disabling GPU telemetry logging because `nvidia-smi` is unavailable.")
                updated_flags['log_gpu'] = False
                skip_reasons['log_gpu'] = "Missing `nvidia-smi`; install NVIDIA drivers and the CUDA toolkit."

    # Ensure CUDA runtime/toolkit is available when GPU benchmarks are requested.
    if run_flags.get('gpu_benchmarks') and gpu_available:
        if not check_cuda_toolkit(torch):
            logger.warning(
                "A CUDA toolkit or CUDA-enabled PyTorch runtime was not detected. GPU benchmarks may not run. "
                "Install CUDA from https://developer.nvidia.com/cuda-downloads or use a GPU-enabled PyTorch build."
            )

    # Optional model-specific dependencies for inference.
    if run_flags.get('gpu_inference') and gpu_available:
        model_name = getattr(args, 'gpu_inference_model', 'custom')
        model_name_lower = model_name.lower()
        if model_name_lower == 'resnet50':
            if not check_python_package('torchvision'):
                warning_message = (
                    "`torchvision` is required for the ResNet50 inference benchmark. Install it with `pip install torchvision` "
                    "or choose a different model via `--gpu-inference-model`. Skipping GPU inference benchmarks."
                )
                logger.warning(warning_message)
                updated_flags['gpu_inference'] = False
                skip_reasons['gpu_inference'] = warning_message
        elif model_name_lower in {'bert', 'gpt2'}:
            if not check_python_package('transformers'):
                warning_message = (
                    f"The `transformers` package is required for the {model_name.upper()} inference benchmark. Install it with "
                    "`pip install transformers` or choose a different model via `--gpu-inference-model`. "
                    "Skipping GPU inference benchmarks."
                )
                logger.warning(warning_message)
                updated_flags['gpu_inference'] = False
                skip_reasons['gpu_inference'] = warning_message

    return updated_flags, skip_reasons



def safe_get_gpus(context):
    """Safely retrieve GPU information, logging friendly warnings on failure."""
    try:
        return GPUtil.getGPUs(), None
    except Exception as exc:
        gpu_not_found_error = getattr(GPUtil, 'GPUNotFound', None)
        gpu_query_error = getattr(GPUtil, 'GPUQueryError', None)
        gputil_general_error = getattr(GPUtil, 'GPUtilError', None)

        if gpu_not_found_error and isinstance(exc, gpu_not_found_error):
            message = ("No GPUs were detected while {}. GPU-specific functionality "
                       "will be skipped.").format(context)
        elif ((gpu_query_error and isinstance(exc, gpu_query_error)) or
              (gputil_general_error and isinstance(exc, gputil_general_error))):
            message = ("GPUtil could not query GPUs while {}: {}. GPU-specific "
                       "functionality will be skipped.".format(context, exc))
        elif isinstance(exc, (subprocess.SubprocessError, FileNotFoundError, OSError)):
            message = ("Failed to execute GPU query while {}: {}. Ensure NVIDIA drivers "
                       "and `nvidia-smi` are installed.".format(context, exc))
        else:
            message = ("Unexpected error while {}: {}. GPU-specific functionality will "
                       "be skipped.".format(context, exc))

        logger.warning(message)
        return [], message

# Set default tensor type based on precision
def set_default_tensor_type(precision):
    if precision == 'fp16':
        torch.set_default_dtype(torch.float16)
    elif precision == 'fp32':
        torch.set_default_dtype(torch.float32)
    elif precision == 'fp64':
        torch.set_default_dtype(torch.float64)
    elif precision == 'bf16':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch.set_default_dtype(torch.bfloat16)
        else:
            print("bfloat16 is not supported on this device. Falling back to float32.")
            torch.set_default_dtype(torch.float32)
    else:
        print(f"Unknown precision: {precision}. Using float32.")
        torch.set_default_dtype(torch.float32)

# Utility Functions
def get_system_info():
    """Collect a comprehensive snapshot of system configuration and health."""
    uname = platform.uname()
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time if boot_time else None

    cpu_freq = psutil.cpu_freq()
    cpu_stats = psutil.cpu_stats()
    load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (None, None, None)

    cpu_model = platform.processor() or uname.processor or "Unknown"

    cpu_info = {
        'model': cpu_model,
        'architecture': platform.machine(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'current_frequency_mhz': round(cpu_freq.current, 2) if cpu_freq else None,
        'max_frequency_mhz': round(cpu_freq.max, 2) if cpu_freq else None,
        'min_frequency_mhz': round(cpu_freq.min, 2) if cpu_freq else None,
        'load_average': load_avg,
        'context_switches': getattr(cpu_stats, 'ctx_switches', None),
        'interrupts': getattr(cpu_stats, 'interrupts', None),
        'soft_interrupts': getattr(cpu_stats, 'soft_interrupts', None),
        'syscalls': getattr(cpu_stats, 'syscalls', None),
    }

    svmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    ram_info = {
        'total_gb': round(svmem.total / (1024 ** 3), 2),
        'available_gb': round(svmem.available / (1024 ** 3), 2),
        'used_gb': round(svmem.used / (1024 ** 3), 2),
        'percent_used': svmem.percent,
    }

    swap_info = {
        'total_gb': round(swap.total / (1024 ** 3), 2),
        'used_gb': round(swap.used / (1024 ** 3), 2),
        'percent_used': swap.percent,
    }

    disk_usage = psutil.disk_usage('/')
    disk_info = {
        'root_total_gb': round(disk_usage.total / (1024 ** 3), 2),
        'root_used_percent': disk_usage.percent,
    }

    disk_partitions = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        disk_partitions.append({
            'device': partition.device,
            'mountpoint': partition.mountpoint,
            'fstype': partition.fstype,
            'total_gb': round(usage.total / (1024 ** 3), 2),
            'used_percent': usage.percent,
        })

    gpus, gpu_warning = safe_get_gpus("collecting system information")

    nvidia_query_fields = [
        'index',
        'name',
        'driver_version',
        'vbios_version',
        'temperature.gpu',
        'fan.speed',
        'power.draw',
        'power.limit',
        'utilization.gpu',
        'memory.total',
        'memory.used',
        'clocks.current.sm',
        'clocks.current.memory'
    ]
    nvidia_telemetry, nvidia_error = run_nvidia_smi_query(nvidia_query_fields)
    nvidia_telemetry_map = {}
    if nvidia_telemetry:
        for entry in nvidia_telemetry:
            gpu_index = entry.get('index')
            if gpu_index is not None:
                nvidia_telemetry_map[str(gpu_index)] = entry

    gpu_info_list = []
    for gpu in gpus:
        telemetry = nvidia_telemetry_map.get(str(gpu.id), {})
        gpu_entry = {
            'id': gpu.id,
            'name': gpu.name,
            'total_memory_gb': round(gpu.memoryTotal / 1024, 2),
            'memory_used_gb': round(gpu.memoryUsed / 1024, 2),
            'temperature_c': getattr(gpu, 'temperature', None),
            'fan_speed_percent': getattr(gpu, 'fanSpeed', None),
            'utilization_percent': getattr(gpu, 'load', None) * 100 if getattr(gpu, 'load', None) is not None else None,
            'driver_version': telemetry.get('driver_version') or gpu.driver,
            'vbios_version': telemetry.get('vbios_version'),
            'power_draw_watts': parse_float(telemetry.get('power.draw')),
            'power_limit_watts': parse_float(telemetry.get('power.limit')),
            'sm_clock_mhz': parse_float(telemetry.get('clocks.current.sm')),
            'memory_clock_mhz': parse_float(telemetry.get('clocks.current.memory')),
        }
        gpu_info_list.append(gpu_entry)

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_devices = []
    if cuda_available:
        for logical_id in range(cuda_device_count):
            try:
                props = torch.cuda.get_device_properties(logical_id)
                capability = f"{props.major}.{props.minor}"
                cuda_devices.append({
                    'logical_id': logical_id,
                    'name': props.name,
                    'total_memory_gb': round(props.total_memory / 1e9, 2),
                    'multi_processor_count': props.multi_processor_count,
                    'compute_capability': capability,
                    'max_threads_per_block': props.max_threads_per_block,
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("Unable to query CUDA device %s: %s", logical_id, exc)

    torch_info = {
        'version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': getattr(torch.backends.cudnn, 'version', lambda: None)(),
        'cuda_available': cuda_available,
        'device_count': cuda_device_count,
        'devices': cuda_devices,
    }

    network_info = []
    net_stats = psutil.net_if_stats()
    for interface, stats in net_stats.items():
        network_info.append({
            'interface': interface,
            'is_up': stats.isup,
            'speed_mbps': stats.speed,
            'mtu': stats.mtu,
        })

    environment = {
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }

    system_overview = {
        'hostname': socket.gethostname(),
        'os': platform.platform(),
        'kernel': uname.release,
        'uptime_seconds': round(uptime_seconds, 2) if uptime_seconds is not None else None,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    system_info = {
        'system': system_overview,
        'cpu': cpu_info,
        'memory': ram_info,
        'swap': swap_info,
        'disk': disk_info,
        'disk_partitions': disk_partitions,
        'network_interfaces': network_info,
        'gpu_info': gpu_info_list,
        'torch': torch_info,
        'environment': environment,
    }

    if gpu_warning:
        system_info['gpu_warning'] = gpu_warning
    if nvidia_error:
        system_info['nvidia_smi_warning'] = nvidia_error

    return system_info

def start_gpu_logging(log_file, log_metrics):
    metrics = log_metrics.split(',')
    query_fields = ','.join(metrics)
    log_cmd = [
        'nvidia-smi',
        f'--query-gpu={query_fields}',
        '--format=csv',
        '-l', '1',
        '-f', log_file
    ]

    try:
        # Start the logging process
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return log_process
    except Exception as e:
        _print_status("error", f"Error starting GPU logging: {e}")
        return None

def stop_gpu_logging(log_process):
    try:
        if log_process:
            # Send termination signal
            log_process.terminate()
            # Wait for the process to terminate
            log_process.wait(timeout=5)
    except Exception as e:
        _print_status("error", f"Error stopping GPU logging: {e}")

def _print_heading(title):
    """Render a consistent console heading banner using rounded outlines."""
    banner = tabulate([[title]], tablefmt='rounded_outline', colalign=('center',))
    print(f"\n{banner}")


def _print_status(label, message, *, newline_before=False):
    """Print a status line with a consistent prefix."""
    prefix = f"[{label.upper()}]"
    if newline_before:
        print()
    print(f"{prefix} {message}")


def _print_section_table(title, rows, headers, *, tablefmt='rounded_grid', colalign=None, allow_empty=False):
    """Print a tabular section where the title is folded into the header."""
    if not rows and not allow_empty:
        return

    display_headers = list(headers) if headers else []
    if display_headers:
        display_headers[0] = f"{title} ▸ {display_headers[0]}"

    print()
    print(tabulate(rows, headers=display_headers, tablefmt=tablefmt, colalign=colalign))


def _format_detail_value(value):
    """Format detailed benchmark values for compact tabular display."""
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, indent=2)

    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"

    if isinstance(value, float):
        formatted = f"{value:,.6f}".rstrip('0').rstrip('.')
        return formatted

    text = str(value)
    if '\n' in text:
        return text

    return textwrap.fill(text, width=70)


def print_detailed_results(results):
    if not results:
        return

    _print_heading("Detailed Benchmark Results")
    field_preference = [
        ("Input Parameters", "input_params"),
        ("Metrics", "metrics"),
        ("GFLOPS", "gflops"),
        ("Execution Time (s)", "execution_time"),
        ("Score", "score"),
    ]

    for result in results:
        if not result:
            continue

        task = result.get('task', 'Unknown Task')
        rows = []
        seen_keys = set()

        for label, key in field_preference:
            if key in result and key not in {'task', 'category'}:
                rows.append([label, _format_detail_value(result[key])])
                seen_keys.add(key)

        for key, value in sorted(result.items()):
            if key in seen_keys or key in {'task', 'category'}:
                continue
            friendly_key = key.replace('_', ' ').title()
            rows.append([friendly_key, _format_detail_value(value)])

        _print_section_table(
            task,
            rows,
            ["Field", "Value"],
            tablefmt='rounded_grid',
            colalign=('left', 'left'),
            allow_empty=True,
        )

def print_results_table(results, total_score, total_execution_time):
    """Render benchmark results grouped by category with consistent tables."""
    if not results and total_score is None and total_execution_time is None:
        return

    max_width_input = 30  # Maximum width for "Input" column
    max_width_metrics = 50  # Maximum width for "Metrics" column

    sections = [
        ("GPU Benchmarks", [res for res in results if res and res.get('category') == 'GPU']),
        ("System Benchmarks", [res for res in results if res and res.get('category') == 'System']),
        ("Other Benchmarks", [res for res in results if res and res.get('category') not in {'GPU', 'System'}]),
    ]

    headers = ["Category", "Task", "Input", "Metrics", "Exec Time (s)", "Score"]
    colalign = ("left", "left", "left", "left", "right", "right")

    rows = []
    for title, section_results in sections:
        if not section_results:
            continue

        first_in_section = True
        for result in section_results:
            task = result.get('task', 'Unknown Task')
            input_params = result.get('input_params', '')
            metrics = result.get('metrics', 'N/A')
            score = result.get('score', 'N/A')
            execution_time = result.get('execution_time', 'N/A')

            wrapped_input = '\n'.join(textwrap.wrap(str(input_params), width=max_width_input)) if input_params else ''
            wrapped_metric = '\n'.join(textwrap.wrap(str(metrics), width=max_width_metrics)) if metrics else ''

            execution_time_str = f"{float(execution_time):.2f}" if isinstance(execution_time, (int, float)) else 'N/A'
            score_str = f"{float(score):.1f}" if isinstance(score, (int, float)) else 'N/A'

            category_label = title if first_in_section else ""
            rows.append([category_label, task, wrapped_input, wrapped_metric, execution_time_str, score_str])
            first_in_section = False

    if total_execution_time is not None or total_score is not None:
        execution_time_str = f"{float(total_execution_time):.2f}" if isinstance(total_execution_time, (int, float)) else 'N/A'
        score_str = f"{float(total_score):.1f}" if isinstance(total_score, (int, float)) else 'N/A'
        rows.append([
            "Summary",
            "Aggregate Totals",
            "",
            "",
            execution_time_str,
            score_str,
        ])

    _print_section_table(
        "Benchmark Results",
        rows,
        headers,
        tablefmt='rounded_grid',
        colalign=colalign,
        allow_empty=True,
    )


def _format_value(value, unit="", precision=2):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{precision}f}" if not isinstance(value, int) else str(value)
    else:
        formatted = str(value)
    return f"{formatted} {unit}".strip()


def print_system_overview(system_info, detailed=False):
    """Pretty-print system overview information."""
    if not system_info:
        _print_status("info", "System information unavailable.")
        return

    if 'error' in system_info:
        _print_status("error", f"System information error: {system_info['error']}")
        return

    if 'gpu_warning' in system_info:
        _print_status("warning", system_info['gpu_warning'])

    if 'nvidia_smi_warning' in system_info:
        _print_status("warning", system_info['nvidia_smi_warning'])

    system_meta = system_info.get('system', {})
    environment = system_info.get('environment', {})
    cpu = system_info.get('cpu', {})
    memory = system_info.get('memory', {})
    swap = system_info.get('swap', {})
    disk = system_info.get('disk', {})
    gpu_info = system_info.get('gpu_info', [])
    torch_info = system_info.get('torch', {})
    disk_partitions = system_info.get('disk_partitions', [])
    network_interfaces = system_info.get('network_interfaces', [])

    def _fmt(value, unit="", precision=2, skip_zero=False):
        if value is None:
            return None
        if skip_zero and isinstance(value, (int, float)) and value == 0:
            return None
        formatted = _format_value(value, unit=unit, precision=precision)
        return formatted if formatted != 'N/A' else None

    def _lines_to_cell(lines):
        filtered = [line for line in lines if line]
        return "\n".join(filtered) if filtered else "N/A"

    def _format_bool(value, true_label="Yes", false_label="No"):
        if value is None:
            return "N/A"
        return true_label if value else false_label

    def _wrap_text(value, width=70):
        if value is None:
            return "N/A"
        if isinstance(value, (int, float, bool)):
            return str(value)
        return textwrap.fill(str(value), width=width)

    load_values = []
    for value in cpu.get('load_average', []) or []:
        if isinstance(value, (int, float)):
            load_values.append(f"{value:.2f}")
        elif value is not None:
            load_values.append(str(value))
    load_display = ', '.join(load_values) if load_values else None

    cuda_visible_raw = environment.get('cuda_visible_devices')
    if cuda_visible_raw:
        cuda_visible_display = str(cuda_visible_raw)
    else:
        cuda_visible_display = 'Not set'
        if gpu_info:
            cuda_visible_display = 'Not set (all GPUs visible)'

    summary_rows = []

    system_lines = [
        f"Host: {system_meta.get('hostname', 'Unknown')}",
        f"OS: {system_meta.get('os', 'Unknown')}",
        f"Kernel: {system_meta.get('kernel', 'Unknown')}",
    ]
    summary_rows.append(('System', _lines_to_cell(system_lines)))

    uptime_line = None
    uptime_seconds = system_meta.get('uptime_seconds')
    if uptime_seconds is not None:
        uptime_line = f"Uptime: {int(uptime_seconds)} s"
    timestamp_line = system_meta.get('timestamp_utc')
    clock_lines = [uptime_line, f"Timestamp: {timestamp_line}" if timestamp_line else None]
    summary_rows.append(('Clock', _lines_to_cell(clock_lines)))

    cpu_lines = []
    model = cpu.get('model') or 'Unknown'
    architecture = cpu.get('architecture')
    if architecture and architecture != model:
        cpu_lines.append(f"Model: {model} ({architecture})")
    else:
        cpu_lines.append(f"Model: {model}")
    core_parts = []
    if cpu.get('physical_cores') is not None:
        core_parts.append(f"{cpu.get('physical_cores')}P")
    if cpu.get('logical_cores') is not None:
        core_parts.append(f"{cpu.get('logical_cores')}L")
    if core_parts:
        cpu_lines.append(f"Cores: {' / '.join(core_parts)}")
    freq_parts = []
    for label, key in [('cur', 'current_frequency_mhz'), ('min', 'min_frequency_mhz'), ('max', 'max_frequency_mhz')]:
        freq_str = _fmt(cpu.get(key), unit='MHz', precision=0, skip_zero=True)
        if freq_str:
            freq_parts.append(f"{label} {freq_str}")
    if freq_parts:
        cpu_lines.append("Freq: " + ', '.join(freq_parts))
    if load_display:
        cpu_lines.append(f"Load avg: {load_display}")
    summary_rows.append(('CPU', _lines_to_cell(cpu_lines)))

    memory_lines = []
    ram_used = _fmt(memory.get('used_gb'), 'GB')
    ram_total = _fmt(memory.get('total_gb'), 'GB')
    if ram_used and ram_total:
        memory_lines.append(f"RAM: {ram_used} / {ram_total}")
    elif ram_total:
        memory_lines.append(f"RAM Total: {ram_total}")
    elif ram_used:
        memory_lines.append(f"RAM Used: {ram_used}")
    ram_available = _fmt(memory.get('available_gb'), 'GB')
    if ram_available:
        memory_lines.append(f"Available: {ram_available}")
    ram_percent = _fmt(memory.get('percent_used'), '%', precision=1)
    if ram_percent:
        memory_lines.append(f"Usage: {ram_percent}")
    swap_used = _fmt(swap.get('used_gb'), 'GB')
    swap_total = _fmt(swap.get('total_gb'), 'GB')
    swap_percent = _fmt(swap.get('percent_used'), '%', precision=1)
    if swap_used and swap_total:
        memory_lines.append(f"Swap: {swap_used} / {swap_total}")
    elif swap_total:
        memory_lines.append(f"Swap Total: {swap_total}")
    if swap_percent:
        memory_lines.append(f"Swap Usage: {swap_percent}")
    summary_rows.append(('Memory', _lines_to_cell(memory_lines)))

    storage_lines = []
    root_total = _fmt(disk.get('root_total_gb'), 'GB')
    root_used_percent = _fmt(disk.get('root_used_percent'), '%', precision=1)
    if root_used_percent and root_total:
        storage_lines.append(f"Root usage: {root_used_percent} of {root_total}")
    elif root_used_percent:
        storage_lines.append(f"Root usage: {root_used_percent}")
    elif root_total:
        storage_lines.append(f"Root capacity: {root_total}")
    partition_count = len(disk_partitions)
    if partition_count:
        storage_lines.append(f"Partitions: {partition_count}")
    summary_rows.append(('Storage', _lines_to_cell(storage_lines)))

    if gpu_info:
        for gpu in sorted(gpu_info, key=lambda g: g.get('id', 0)):
            gpu_lines = [
                f"Name: {gpu.get('name', 'Unknown')}",
                f"Driver: {gpu.get('driver_version', 'Unknown')}"
            ]
            vbios = gpu.get('vbios_version')
            if vbios:
                gpu_lines.append(f"VBIOS: {vbios}")
            mem_used_gpu = _fmt(gpu.get('memory_used_gb'), 'GB')
            mem_total_gpu = _fmt(gpu.get('total_memory_gb'), 'GB')
            if mem_used_gpu and mem_total_gpu:
                gpu_lines.append(f"Memory: {mem_used_gpu} / {mem_total_gpu}")
            elif mem_total_gpu:
                gpu_lines.append(f"Memory Total: {mem_total_gpu}")
            temp_display = _fmt(gpu.get('temperature_c'), '°C')
            if temp_display:
                gpu_lines.append(f"Temp: {temp_display}")
            util_display = _fmt(gpu.get('utilization_percent'), '%', precision=1)
            if util_display:
                gpu_lines.append(f"Util: {util_display}")
            power_draw = _fmt(gpu.get('power_draw_watts'), 'W', precision=1)
            power_limit = _fmt(gpu.get('power_limit_watts'), 'W', precision=1)
            if power_draw and power_limit:
                gpu_lines.append(f"Power: {power_draw} / {power_limit}")
            elif power_draw:
                gpu_lines.append(f"Power: {power_draw}")
            elif power_limit:
                gpu_lines.append(f"Power Limit: {power_limit}")
            fan_display = _fmt(gpu.get('fan_speed_percent'), '%', precision=0)
            if fan_display:
                gpu_lines.append(f"Fan: {fan_display}")
            sm_clock = _fmt(gpu.get('sm_clock_mhz'), 'MHz', precision=0)
            mem_clock = _fmt(gpu.get('memory_clock_mhz'), 'MHz', precision=0)
            clock_parts = []
            if sm_clock:
                clock_parts.append(f"SM {sm_clock}")
            if mem_clock:
                clock_parts.append(f"Mem {mem_clock}")
            if clock_parts:
                gpu_lines.append("Clocks: " + ' | '.join(clock_parts))
            summary_rows.append((f"GPU {gpu.get('id', 'N/A')}", _lines_to_cell(gpu_lines)))
    else:
        summary_rows.append(('GPU', 'No GPUs detected.'))

    torch_lines = [f"PyTorch: {torch_info.get('version', 'Unknown')}"]
    if torch_info.get('cuda_version'):
        torch_lines.append(f"CUDA: {torch_info.get('cuda_version')}")
    if torch_info.get('cudnn_version'):
        torch_lines.append(f"cuDNN: {torch_info.get('cudnn_version')}")
    torch_lines.append(f"CUDA available: {_format_bool(torch_info.get('cuda_available'))}")
    if torch_info.get('device_count') is not None:
        torch_lines.append(f"Devices detected: {torch_info.get('device_count')}")
    summary_rows.append(('PyTorch', _lines_to_cell(torch_lines)))

    env_lines = []
    python_version = environment.get('python_version')
    if python_version:
        env_lines.append(f"Python: {python_version}")
    python_exec = environment.get('python_executable')
    if python_exec:
        env_lines.append(f"Executable: {python_exec}")
    env_lines.append(f"CUDA_VISIBLE_DEVICES: {cuda_visible_display}")
    summary_rows.append(('Environment', _lines_to_cell(env_lines)))

    _print_section_table(
        'System Overview',
        summary_rows,
        ['Category', 'Details'],
        tablefmt='rounded_grid',
        colalign=('left', 'left'),
        allow_empty=True,
    )

    if detailed:
        partition_rows = []
        for part in sorted(disk_partitions, key=lambda p: (p.get('mountpoint') or '', p.get('device') or '')):
            partition_rows.append([
                part.get('device') or 'Unknown',
                part.get('mountpoint') or 'Unknown',
                part.get('fstype') or 'Unknown',
                _fmt(part.get('total_gb'), 'GB') or 'N/A',
                _fmt(part.get('used_percent'), '%', precision=1) or 'N/A',
            ])
        _print_section_table(
            'Disk Partitions',
            partition_rows,
            ['Device', 'Mount', 'FS', 'Total (GB)', 'Used %'],
            tablefmt='rounded_grid',
            colalign=('left', 'left', 'left', 'right', 'right'),
        )

        cpu_stats_rows = []
        for label, key in [
            ('Context Switches', 'context_switches'),
            ('Interrupts', 'interrupts'),
            ('Soft Interrupts', 'soft_interrupts'),
            ('System Calls', 'syscalls'),
        ]:
            stat_value = cpu.get(key)
            if stat_value is not None:
                cpu_stats_rows.append([label, stat_value])
        _print_section_table(
            'CPU Scheduler Stats',
            cpu_stats_rows,
            ['Metric', 'Value'],
            tablefmt='rounded_grid',
            colalign=('left', 'right'),
        )

        torch_devices = torch_info.get('devices', [])
        device_rows = []
        for device in torch_devices:
            device_rows.append([
                device.get('logical_id'),
                device.get('name', 'Unknown'),
                _fmt(device.get('total_memory_gb'), 'GB') or 'N/A',
                device.get('compute_capability') or 'Unknown',
                device.get('multi_processor_count'),
                device.get('max_threads_per_block'),
            ])
        _print_section_table(
            'CUDA Device Properties (PyTorch view)',
            device_rows,
            ['Logical ID', 'Name', 'Total Mem (GB)', 'Compute Capability', 'SMs', 'Max Threads/Block'],
            colalign=('right', 'left', 'right', 'left', 'right', 'right'),
        )

        gpu_rows = []
        for gpu in sorted(gpu_info, key=lambda g: g.get('id', 0)):
            name_display = _wrap_text(gpu.get('name', 'Unknown'), width=24)

            driver_lines = [f"Driver {gpu.get('driver_version', 'Unknown')}"]
            vbios = gpu.get('vbios_version')
            if vbios:
                driver_lines.append(f"VBIOS {vbios}")
            driver_info = _lines_to_cell(driver_lines)

            mem_used_gpu = _fmt(gpu.get('memory_used_gb'), 'GB')
            mem_total_gpu = _fmt(gpu.get('total_memory_gb'), 'GB')
            memory_lines = []
            if mem_used_gpu and mem_total_gpu:
                memory_lines.append(f"Used {mem_used_gpu}")
                memory_lines.append(f"Total {mem_total_gpu}")
            elif mem_total_gpu:
                memory_lines.append(f"Total {mem_total_gpu}")
            elif mem_used_gpu:
                memory_lines.append(f"Used {mem_used_gpu}")
            memory_info = _lines_to_cell(memory_lines)

            thermal_lines = []
            temp_display = _fmt(gpu.get('temperature_c'), '°C')
            if temp_display:
                thermal_lines.append(f"Temp {temp_display}")
            util_display = _fmt(gpu.get('utilization_percent'), '%', precision=1)
            if util_display:
                thermal_lines.append(f"Util {util_display}")
            fan_display = _fmt(gpu.get('fan_speed_percent'), '%', precision=0)
            if fan_display:
                thermal_lines.append(f"Fan {fan_display}")
            thermal_info = _lines_to_cell(thermal_lines)

            power_lines = []
            power_draw = _fmt(gpu.get('power_draw_watts'), 'W', precision=1)
            power_limit = _fmt(gpu.get('power_limit_watts'), 'W', precision=1)
            if power_draw:
                power_lines.append(f"Draw {power_draw}")
            if power_limit:
                power_lines.append(f"Limit {power_limit}")
            power_info = _lines_to_cell(power_lines)

            clock_lines = []
            sm_clock = _fmt(gpu.get('sm_clock_mhz'), 'MHz', precision=0)
            if sm_clock:
                clock_lines.append(f"SM {sm_clock}")
            mem_clock = _fmt(gpu.get('memory_clock_mhz'), 'MHz', precision=0)
            if mem_clock:
                clock_lines.append(f"Mem {mem_clock}")
            clock_info = _lines_to_cell(clock_lines)

            gpu_rows.append([
                gpu.get('id', 'N/A'),
                name_display,
                driver_info,
                memory_info,
                thermal_info,
                power_info,
                clock_info,
            ])

        _print_section_table(
            'GPU Telemetry (nvidia-smi view)',
            gpu_rows,
            ['ID', 'Name', 'Driver / Firmware', 'Memory', 'Thermals / Utilization', 'Power', 'Clocks'],
            colalign=('right', 'left', 'left', 'left', 'left', 'left', 'left'),
        )

        firmware_rows = []
        for gpu in sorted(gpu_info, key=lambda g: g.get('id', 0)):
            vbios = gpu.get('vbios_version')
            if vbios:
                firmware_rows.append([gpu.get('id', 'N/A'), vbios])
        _print_section_table(
            'GPU Firmware',
            firmware_rows,
            ['GPU ID', 'VBIOS Version'],
            colalign=('right', 'left'),
        )

        network_rows = []
        for nic in sorted(network_interfaces, key=lambda n: n.get('interface') or n.get('name') or ''):
            interface_name = nic.get('interface') or nic.get('name') or 'Unknown'
            network_rows.append([
                interface_name,
                _format_bool(nic.get('is_up')),  # Up status
                _fmt(nic.get('speed_mbps'), 'Mbps', precision=0) or 'N/A',
                nic.get('mtu', 'N/A')
            ])
        _print_section_table(
            'Network Interfaces',
            network_rows,
            ['Interface', 'Up', 'Speed', 'MTU'],
            colalign=('left', 'center', 'right', 'right'),
        )

        env_rows = []
        display_environment = dict(environment)
        display_environment['cuda_visible_devices'] = cuda_visible_display
        for key, value in sorted(display_environment.items()):
            env_rows.append([key.replace('_', ' ').title(), _wrap_text(value)])
        _print_section_table(
            'Environment',
            env_rows,
            ['Variable', 'Value'],
            colalign=('left', 'left'),
        )

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Comprehensive Benchmarking Script for GPU, CPU, and Memory Performance',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # General Arguments
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--json', action='store_true',
                               help='Output results in JSON format')
    general_group.add_argument('--detailed-output', action='store_true',
                               help='Show detailed benchmark results')
    general_group.add_argument('--num-iterations', type=int, default=1,
                               help='Number of times to run the benchmarks (default: 1)')
    general_group.add_argument('--log-gpu', action='store_true',
                               help='Enable GPU logging during benchmarks')
    general_group.add_argument('--gpu-log-file', type=str, default='gpu_log.csv',
                               help='Specify GPU log file name')
    general_group.add_argument('--gpu-log-metrics', type=str,
                               default='timestamp,pstate,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.max.graphics,power.draw,clocks_throttle_reasons.active',
                               help='Comma-separated list of GPU metrics to log')
    general_group.add_argument('--gpus', type=str, default=None,
                               help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')

    general_group.add_argument('--precision', type=str, default='fp16',
                               choices=['fp16', 'fp32', 'fp64', 'bf16'],
                               help='Precision to use for computations (default: fp16)')

    # Benchmark Selection
    benchmark_group = parser.add_argument_group('Benchmark Selection')
    benchmark_group.add_argument('--all', action='store_true', help='Run all benchmarks')

    # GPU Benchmarks
    gpu_group = parser.add_argument_group('GPU Benchmarks')
    gpu_group.add_argument('--gpu-data-gen', action='store_true', help='Run GPU Data Generation benchmark')
    gpu_group.add_argument('--gpu-to-cpu-transfer', action='store_true', help='Run GPU to CPU Transfer benchmark')
    gpu_group.add_argument('--gpu-to-gpu-transfer', action='store_true', help='Run GPU to GPU Transfer benchmark')
    gpu_group.add_argument('--gpu-memory-bandwidth', action='store_true', help='Run GPU Memory Bandwidth benchmark')
    gpu_group.add_argument('--gpu-tensor', action='store_true', help='Run GPU Tensor Core Performance benchmark')
    gpu_group.add_argument('--gpu-compute', action='store_true', help='Run GPU Computational Task benchmark')
    gpu_group.add_argument('--gpu-data-size-gb', type=float, default=5.0,
                           help='Data size in GB for GPU benchmarks (default: 5.0)')
    gpu_group.add_argument('--gpu-memory-size-gb', type=float, default=5.0,
                           help='Memory size in GB for GPU Memory Bandwidth benchmark (default: 5.0)')
    gpu_group.add_argument('--gpu-tensor-matrix-size', type=int, default=4096,
                           help='Matrix size for GPU Tensor Core benchmark (default: 4096)')
    gpu_group.add_argument('--gpu-tensor-iterations', type=int, default=1000,
                           help='Iterations for GPU Tensor Core benchmark (default: 1000)')
    gpu_group.add_argument('--gpu-comp-epochs', type=int, default=200,
                           help='Number of epochs for GPU computational task (default: 200)')
    gpu_group.add_argument('--gpu-comp-batch-size', type=int, default=2048,
                           help='Batch size for GPU computational task (default: 2048)')
    gpu_group.add_argument('--gpu-comp-input-size', type=int, default=4096,
                           help='Input size for GPU computational task (default: 4096)')
    gpu_group.add_argument('--gpu-comp-hidden-size', type=int, default=4096,
                           help='Hidden layer size for GPU computational task (default: 4096)')
    gpu_group.add_argument('--gpu-comp-output-size', type=int, default=2000,
                           help='Output size for GPU computational task (default: 2000)')

    # Custom Inference Benchmark
    inference_group = parser.add_argument_group('GPU Inference Benchmark')
    inference_group.add_argument('--gpu-inference', action='store_true', help='Run GPU Inference Performance benchmark')
    inference_group.add_argument('--gpu-inference-model', type=str, default='custom',
                                 help=textwrap.dedent('''\
                                     Model to use for inference benchmark (default: custom).
                                     Options: custom, resnet50, bert, gpt2'''))
    inference_group.add_argument('--model-size', type=int, default=5,
                                 help='Depth of the custom inference model (default: 5)')
    inference_group.add_argument('--batch-size', type=int, default=256,
                                 help='Batch size for inference benchmark (default: 256)')
    inference_group.add_argument('--input-size', type=int, default=224,
                                 help='Input size for inference benchmark (default: 224)')
    inference_group.add_argument('--output-size', type=int, default=1000,
                                 help='Output size for inference benchmark (default: 1000)')
    inference_group.add_argument('--iterations', type=int, default=100,
                                 help='Number of iterations for inference benchmark (default: 100)')

    # CPU Benchmarks
    cpu_group = parser.add_argument_group('CPU Benchmarks')
    cpu_group.add_argument('--cpu-single-thread', action='store_true', help='Run CPU Single-threaded Performance benchmark')
    cpu_group.add_argument('--cpu-multi-thread', action='store_true', help='Run CPU Multi-threaded Performance benchmark')
    cpu_group.add_argument('--cpu-to-disk-write', action='store_true', help='Run CPU to Disk Write benchmark')
    cpu_group.add_argument('--memory-bandwidth', action='store_true', help='Run Memory Bandwidth benchmark')
    cpu_group.add_argument('--cpu-num-threads', type=int, default=psutil.cpu_count(logical=True),
                           help='Number of threads to use for multi-threaded CPU benchmark (default: all logical cores)')
    cpu_group.add_argument('--data-size-gb-cpu', type=float, default=5.0,
                           help='Data size in GB for CPU to Disk Write benchmark (default: 5.0)')
    cpu_group.add_argument('--memory-size-mb-cpu', type=int, default=1024,
                           help='Memory size in MB for CPU Memory Bandwidth benchmark (default: 1024)')

    # Disk I/O Benchmark
    disk_group = parser.add_argument_group('Disk I/O Benchmark')
    disk_group.add_argument('--disk-io', action='store_true', help='Run Disk I/O Performance benchmark')
    disk_group.add_argument('--disk-data-size', type=float, default=2.0,
                            help='Data size in GB for disk I/O benchmark (default: 2.0)')
    disk_group.add_argument('--disk-block-size', type=int, default=4,
                            help='Block size in KB for disk I/O benchmark (default: 4)')
    disk_group.add_argument('--disk-io-depth', type=int, default=16,
                            help='IO depth for disk I/O benchmark (default: 16)')
    disk_group.add_argument('--disk-num-jobs', type=int, default=8,
                            help='Number of concurrent jobs for disk I/O benchmark (default: 8)')

    return parser.parse_args()

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def hash_data(data_block):
    return hashlib.sha256(data_block).hexdigest()

def compress_decompress(data_block):
    compressed = gzip.compress(data_block)
    decompressed = gzip.decompress(compressed)
    return decompressed

# CPU Benchmark Functions
def benchmark_cpu_to_disk_write(file_path, data_size_gb, reference_metrics):
    """
    Writes data from CPU to disk to benchmark disk write performance.
    """
    try:
        # Calculate total number of elements based on precision
        dtype = torch.float32  # Default dtype
        element_size = 4       # Default element size in bytes
        if torch.get_default_dtype() == torch.float16:
            dtype = torch.float16
            element_size = 2
        elif torch.get_default_dtype() == torch.float64:
            dtype = torch.float64
            element_size = 8
        elif torch.get_default_dtype() == torch.bfloat16:
            dtype = torch.bfloat16
            element_size = 2

        num_elements = int((data_size_gb * 1e9) / element_size)

        # Generate data on CPU
        cpu_data = torch.randn(num_elements, dtype=dtype)

        # Write data to disk
        start = time.time()
        with open(file_path, 'wb') as f:
            f.write(cpu_data.numpy().tobytes())
        end = time.time()

        write_time = end - start
        data_size_bytes = num_elements * element_size
        data_size_gb_actual = data_size_bytes / 1e9
        write_bandwidth = data_size_gb_actual / write_time if write_time > 0 else float('inf')

        input_params = f'Data Size: {data_size_gb} GB'
        metrics = f"Bandwidth: {write_bandwidth:.2f} GB/s"

        result = {
            'task': 'CPU to Disk Write',
            'category': 'System',
            'input_params': input_params,
            'metrics': metrics,
            'data_size_gb': data_size_gb_actual,
            'time_seconds': write_time,
            'bandwidth_gb_per_second': write_bandwidth,
            'execution_time': write_time,
            'score': (write_bandwidth / reference_metrics['cpu_to_disk_write_bandwidth']) * 100
        }

        # Cleanup
        del cpu_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optionally, remove the file after benchmarking
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        print(f"Error during CPU to Disk Write benchmarking: {e}")
        return None

def benchmark_cpu_single_thread(reference_metrics):
    """
    Performs single-threaded CPU benchmarks covering computational, cryptographic, and data processing tasks.
    """
    try:
        total_time = 0.0

        # Computational Task: Fibonacci Calculation (Iterative)
        n = 500000  # Adjusted for desired computation time
        start_time = time.time()
        fib_result = fibonacci(n)
        comp_time = time.time() - start_time
        total_time += comp_time

        # Cryptographic Task: SHA-256 Hashing
        data_size_mb = 100
        data = os.urandom(data_size_mb * 1024 * 1024)  # Generate random data
        start_time = time.time()
        hash_result = hashlib.sha256(data).hexdigest()
        crypto_time = time.time() - start_time
        total_time += crypto_time

        # Data Processing Task: Gzip Compression/Decompression
        start_time = time.time()
        compressed_data = gzip.compress(data)
        decompressed_data = gzip.decompress(compressed_data)
        data_proc_time = time.time() - start_time
        total_time += data_proc_time

        # Calculate performance metrics
        comp_perf = n / comp_time  # Fibonacci numbers per second
        crypto_perf = (data_size_mb / crypto_time)  # MB hashed per second
        data_proc_perf = (data_size_mb / data_proc_time)  # MB processed per second

        input_params = "Single-threaded CPU Benchmark"
        metrics = (f"Comp Perf: {comp_perf:.2f} fib/sec, "
                   f"Crypto Perf: {crypto_perf:.2f} MB/s, "
                   f"Data Proc Perf: {data_proc_perf:.2f} MB/s")

        result = {
            'task': 'CPU Single-threaded Performance',
            'category': 'System',
            'input_params': input_params,
            'metrics': metrics,
            'fib_number': n,
            'comp_time_seconds': comp_time,
            'comp_perf': comp_perf,
            'crypto_data_size_mb': data_size_mb,
            'crypto_time_seconds': crypto_time,
            'crypto_perf_mb_per_sec': crypto_perf,
            'data_proc_time_seconds': data_proc_time,
            'data_proc_perf_mb_per_sec': data_proc_perf,
            'execution_time': total_time,
            'score': ((comp_perf / reference_metrics['cpu_single_thread_comp_perf']) +
                      (crypto_perf / reference_metrics['cpu_single_thread_crypto_perf']) +
                      (data_proc_perf / reference_metrics['cpu_single_thread_data_proc_perf'])) * 100 / 3
        }

        # Cleanup
        del data
        del compressed_data
        del decompressed_data

        return result

    except Exception as e:
        print(f"Error during single-threaded CPU benchmarking: {e}")
        return None

from concurrent.futures import ThreadPoolExecutor

def benchmark_cpu_multi_thread(reference_metrics, num_threads):
    """
    Performs multi-threaded CPU benchmarks covering computational, cryptographic, and data processing tasks.
    """
    try:
        total_time = 0.0

        n = 500000  # Adjusted for desired computation time

        # Use multiprocessing Pool for CPU-bound tasks
        with mp.Pool(processes=num_threads) as pool:
            # Computational Task
            start_time = time.time()
            fib_results = pool.map(fibonacci, [n] * num_threads)
            comp_time = time.time() - start_time
        total_time += comp_time

        # Cryptographic Task: SHA-256 Hashing
        data_size_mb = 100
        data_blocks = [os.urandom(data_size_mb * 1024 * 1024) for _ in range(num_threads)]
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            hash_results = list(executor.map(hash_data, data_blocks))
            crypto_time = time.time() - start_time
        total_time += crypto_time

        # Data Processing Task: Gzip Compression/Decompression
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            decompressed_results = list(executor.map(compress_decompress, data_blocks))
            data_proc_time = time.time() - start_time
        total_time += data_proc_time

        # Calculate performance metrics
        comp_perf = (n * num_threads) / comp_time  # Fibonacci numbers per second
        crypto_perf = (data_size_mb * num_threads) / crypto_time  # MB hashed per second
        data_proc_perf = (data_size_mb * num_threads) / data_proc_time  # MB processed per second

        input_params = f"Multi-threaded CPU Benchmark with {num_threads} threads"
        metrics = (f"Comp Perf: {comp_perf:.2f} fib/sec, "
                   f"Crypto Perf: {crypto_perf:.2f} MB/s, "
                   f"Data Proc Perf: {data_proc_perf:.2f} MB/s")

        result = {
            'task': 'CPU Multi-threaded Performance',
            'category': 'System',
            'input_params': input_params,
            'metrics': metrics,
            'fib_number': n,
            'comp_time_seconds': comp_time,
            'comp_perf': comp_perf,
            'crypto_data_size_mb': data_size_mb * num_threads,
            'crypto_time_seconds': crypto_time,
            'crypto_perf_mb_per_sec': crypto_perf,
            'data_proc_time_seconds': data_proc_time,
            'data_proc_perf_mb_per_sec': data_proc_perf,
            'execution_time': total_time,
            'score': ((comp_perf / reference_metrics['cpu_multi_thread_comp_perf']) +
                      (crypto_perf / reference_metrics['cpu_multi_thread_crypto_perf']) +
                      (data_proc_perf / reference_metrics['cpu_multi_thread_data_proc_perf'])) * 100 / 3
        }

        return result

    except Exception as e:
        print(f"Error during multi-threaded CPU benchmarking: {e}")
        return None

def benchmark_memory_bandwidth(memory_size_mb, reference_metrics):
    """
    Measures memory bandwidth by performing large memory copy operations.
    """
    try:
        data_size = memory_size_mb * 1024 * 1024  # Convert MB to bytes
        dtype = np.float32  # Default dtype
        element_size = 4    # Default element size in bytes

        if torch.get_default_dtype() == torch.float16:
            dtype = np.float16
            element_size = 2
        elif torch.get_default_dtype() == torch.float64:
            dtype = np.float64
            element_size = 8
        elif torch.get_default_dtype() == torch.bfloat16:
            dtype = np.float16  # numpy does not support bfloat16
            element_size = 2

        array_size = data_size // element_size  # Number of elements

        # Generate data
        src_array = np.random.rand(array_size).astype(dtype)

        # Warm-up
        dest_array = np.copy(src_array)

        # Measure memory copy bandwidth
        start_time = time.time()
        dest_array = np.copy(src_array)
        end_time = time.time()

        copy_time = end_time - start_time
        bandwidth_gb_per_sec = (data_size / copy_time) / 1e9

        input_params = f"Memory Size: {memory_size_mb} MB"
        metrics = f"Bandwidth: {bandwidth_gb_per_sec:.2f} GB/s"

        result = {
            'task': 'Memory Bandwidth',
            'category': 'System',
            'input_params': input_params,
            'metrics': metrics,
            'memory_size_mb': memory_size_mb,
            'copy_time_seconds': copy_time,
            'bandwidth_gb_per_sec': bandwidth_gb_per_sec,
            'execution_time': copy_time,
            'score': (bandwidth_gb_per_sec / reference_metrics['memory_bandwidth_gb_per_sec']) * 100
        }

        # Cleanup
        del src_array
        del dest_array

        return result

    except Exception as e:
        print(f"Error during memory bandwidth benchmarking: {e}")
        return None

def benchmark_disk_io(file_path, data_size_gb, block_size_kb, io_depth, num_jobs, reference_metrics):
    """
    Measures disk read/write throughput and IOPS for sequential and random access patterns.
    """
    try:
        # Convert sizes to appropriate units
        block_size_bytes = block_size_kb * 1024

        # Calculate data size in bytes
        data_size_bytes = int(data_size_gb * 1e9)

        # Ensure that data size is at least equal to block size
        if data_size_bytes < block_size_bytes:
            data_size_bytes = block_size_bytes
            print(f"Adjusted data size to {data_size_bytes} bytes to be at least equal to block size.")

        data_size = f'{data_size_bytes}'
        block_size = f'{block_size_kb}K'

        # Define tests to run
        tests = [
            {'name': 'sequential_read', 'rw': 'read', 'iodepth': io_depth, 'numjobs': num_jobs},
            {'name': 'sequential_write', 'rw': 'write', 'iodepth': io_depth, 'numjobs': num_jobs},
            {'name': 'random_read', 'rw': 'randread', 'iodepth': io_depth, 'numjobs': num_jobs},
            {'name': 'random_write', 'rw': 'randwrite', 'iodepth': io_depth, 'numjobs': num_jobs},
        ]

        results = {}
        total_execution_time = 0.0

        for test in tests:
            print(f"Running Disk {test['name'].replace('_', ' ').title()} benchmark...")
            fio_cmd = [
                'fio',
                f'--name={test["name"]}',
                f'--filename={file_path}',
                '--ioengine=libaio',
                f'--rw={test["rw"]}',
                f'--size={data_size}',
                f'--bs={block_size}',
                f'--iodepth={test["iodepth"]}',
                f'--numjobs={test["numjobs"]}',
                '--direct=1',
                '--runtime=30',
                '--time_based',
                '--group_reporting',
                f'--output={test["name"]}_output.json',
                '--output-format=json'
            ]

            start_time = time.time()
            fio_process = subprocess.run(fio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            end_time = time.time()
            test_execution_time = end_time - start_time
            total_execution_time += test_execution_time

            if fio_process.returncode != 0:
                print(f"Fio {test['name']} failed: {fio_process.stderr}")
                return None

            # Parse fio output
            with open(f'{test["name"]}_output.json', 'r') as f:
                fio_output = json.load(f)

            if 'jobs' not in fio_output or len(fio_output['jobs']) == 0:
                print(f"Fio output for {test['name']} does not contain expected job information.")
                return None

            job = fio_output['jobs'][0]

            if test['rw'] in ['read', 'randread']:
                read_bw = job['read']['bw'] / 1024  # Convert to MB/s
                read_iops = job['read']['iops']
                results[f'{test["name"]}_throughput_mb_per_sec'] = read_bw
                results[f'{test["name"]}_iops'] = read_iops
            else:
                write_bw = job['write']['bw'] / 1024  # Convert to MB/s
                write_iops = job['write']['iops']
                results[f'{test["name"]}_throughput_mb_per_sec'] = write_bw
                results[f'{test["name"]}_iops'] = write_iops

            # Cleanup output file
            if os.path.exists(f'{test["name"]}_output.json'):
                os.remove(f'{test["name"]}_output.json')

        input_params = (f"Data Size: {data_size_gb} GB, Block Size: {block_size_kb} KB, "
                        f"IO Depth: {io_depth}, Num Jobs: {num_jobs}")

        metrics = (
            f"Seq Read: {results.get('sequential_read_throughput_mb_per_sec', 0):.2f} MB/s, "
            f"Seq Write: {results.get('sequential_write_throughput_mb_per_sec', 0):.2f} MB/s, "
            f"Rand Read IOPS: {int(results.get('random_read_iops', 0))}, "
            f"Rand Write IOPS: {int(results.get('random_write_iops', 0))}"
        )

        result = {
            'task': 'Disk I/O Performance',
            'category': 'System',
            'input_params': input_params,
            'metrics': metrics,
            'execution_time': total_execution_time,
            **results
        }

        # Calculate scores based on sequential read/write throughput and random read/write IOPS
        seq_read_throughput_score = (results['sequential_read_throughput_mb_per_sec'] / reference_metrics['sequential_read_throughput_mb_per_sec']) * 100
        seq_write_throughput_score = (results['sequential_write_throughput_mb_per_sec'] / reference_metrics['sequential_write_throughput_mb_per_sec']) * 100
        rand_read_iops_score = (results['random_read_iops'] / reference_metrics['random_read_iops']) * 100
        rand_write_iops_score = (results['random_write_iops'] / reference_metrics['random_write_iops']) * 100

        # Average the scores
        result['score'] = (seq_read_throughput_score + seq_write_throughput_score + rand_read_iops_score + rand_write_iops_score) / 4

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        print(f"Error during disk I/O benchmarking: {e}")
        return None

# GPU Benchmark Functions

def validate_gpu_ids(gpu_ids):
    available_gpus, warning_message = safe_get_gpus("validating requested GPU IDs")
    available_gpu_ids = [gpu.id for gpu in available_gpus]
    valid_gpu_ids = []
    for gpu_id in gpu_ids:
        if gpu_id in available_gpu_ids:
            valid_gpu_ids.append(gpu_id)
        else:
            print(f"Warning: GPU ID {gpu_id} is not available. It will be skipped.")
    return valid_gpu_ids, warning_message

def map_physical_to_logical_gpu_ids(gpu_ids):
    """
    Maps physical GPU IDs to logical IDs after setting CUDA_VISIBLE_DEVICES.
    """
    physical_to_logical = {}
    for logical_id, physical_id in enumerate(gpu_ids):
        physical_to_logical[physical_id] = logical_id
    return physical_to_logical

def resolve_requested_gpus(requested_ids):
    """Resolve requested GPU IDs into physical IDs and logical ID mapping."""
    if requested_ids is None:
        if torch.cuda.is_available():
            available_gpus, gpu_query_message = safe_get_gpus("resolving requested GPUs")
            physical_gpu_ids = [gpu.id for gpu in available_gpus]
        else:
            physical_gpu_ids = []
            gpu_query_message = None
    else:
        physical_gpu_ids, gpu_query_message = validate_gpu_ids(requested_ids)

    physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)

    status_message = gpu_query_message
    if not physical_gpu_ids:
        if status_message is None and requested_ids is None:
            if torch.cuda.is_available():
                status_message = "No GPUs detected."
            else:
                status_message = "CUDA is not available. No GPUs detected."
        elif status_message is None:
            status_message = f"No valid GPUs available from requested IDs: {requested_ids}."

    return physical_gpu_ids, physical_to_logical, status_message

def get_torch_dtype_from_precision(precision):
    if precision == 'fp16':
        return torch.float16
    elif precision == 'fp32':
        return torch.float32
    elif precision == 'fp64':
        return torch.float64
    elif precision == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32  # Default

def run_data_generation_on_gpu(logical_gpu_id, num_elements_per_gpu, dtype, return_dict):
    try:
        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)
        start_time = time.time()
        tensor = torch.randn(num_elements_per_gpu, device=device, dtype=dtype)
        torch.cuda.synchronize()
        end_time = time.time()
        gen_time = end_time - start_time
        return_dict[logical_gpu_id] = gen_time
        del tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error on logical GPU {logical_gpu_id} during data generation: {e}")
        return_dict[logical_gpu_id] = None

def run_gpu_to_cpu_transfer_on_gpu(logical_gpu_id, num_elements_per_gpu, dtype, return_dict):
    try:
        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)
        data = torch.randn(num_elements_per_gpu, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start_time = time.time()
        cpu_data = data.cpu()
        torch.cuda.synchronize()
        end_time = time.time()
        transfer_time = end_time - start_time
        return_dict[logical_gpu_id] = transfer_time
        del data
        del cpu_data
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error on logical GPU {logical_gpu_id} during GPU to CPU transfer: {e}")
        return_dict[logical_gpu_id] = None

def run_gpu_to_gpu_transfer(logical_gpu0_id, logical_gpu1_id, num_elements, iterations, dtype, return_dict):
    try:
        device0 = torch.device(f'cuda:{logical_gpu0_id}')
        device1 = torch.device(f'cuda:{logical_gpu1_id}')
        torch.cuda.set_device(device0)
        src_tensor = torch.randn(num_elements, device=device0, dtype=dtype)
        dest_tensor = torch.empty(num_elements, device=device1, dtype=dtype)

        # Warm-up
        torch.cuda.synchronize()
        dest_tensor.copy_(src_tensor)
        torch.cuda.synchronize()

        # Measure copy bandwidth
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            dest_tensor.copy_(src_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        total_copy_time = end_time - start_time
        average_copy_time = total_copy_time / iterations
        return_dict['copy_time'] = average_copy_time
        del src_tensor
        del dest_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during GPU to GPU transfer: {e}")
        return_dict['copy_time'] = None

def benchmark_gpu_data_generation(data_size_gb, reference_metrics, precision, physical_gpu_ids=None):
    """
    Generates large tensors of random numbers directly on the GPUs to benchmark GPU memory bandwidth.
    """
    try:
        dtype = get_torch_dtype_from_precision(precision)

        physical_gpu_ids, physical_to_logical, status_message = resolve_requested_gpus(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            message = status_message or "No valid GPUs available."
            print(f"{message} Cannot run GPU Data Generation benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        logical_gpu_ids = [physical_to_logical[physical_id] for physical_id in physical_gpu_ids]

        print(f"Using GPUs: {physical_gpu_ids} (logical IDs: {logical_gpu_ids}) for GPU Data Generation")

        # Calculate total number of elements per GPU
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements_per_gpu = int(((data_size_gb * 1e9) / element_size) / num_gpus)

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for physical_id in physical_gpu_ids:
            logical_id = physical_to_logical[physical_id]
            p = mp.Process(target=run_data_generation_on_gpu, args=(logical_id, num_elements_per_gpu, dtype, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect per-GPU times
        gen_times = [t for t in return_dict.values() if t is not None]
        if not gen_times:
            print("No valid generation times collected.")
            return None

        # Calculate per-GPU bandwidths
        per_gpu_bandwidths = []
        data_size_bytes_per_gpu = num_elements_per_gpu * element_size
        data_size_gb_per_gpu = data_size_bytes_per_gpu / 1e9
        for gen_time in gen_times:
            bw = data_size_gb_per_gpu / gen_time if gen_time > 0 else 0
            per_gpu_bandwidths.append(bw)

        # Total bandwidth is sum of per-GPU bandwidths
        total_bandwidth = sum(per_gpu_bandwidths)

        # Total execution time is the maximum of the per-GPU times (since processes run in parallel)
        total_time = max(gen_times)

        # Total data size generated
        data_size_bytes_total = data_size_bytes_per_gpu * num_gpus
        data_size_gb_actual = data_size_bytes_total / 1e9

        metrics = f"Bandwidth: {total_bandwidth:.2f} GB/s"
        input_params = f"Data Size: {data_size_gb} GB, Precision: {precision}"

        result = {
            'task': 'GPU Data Generation',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'data_size_gb': data_size_gb_actual,
            'time_seconds': total_time,
            'bandwidth_gb_per_second': total_bandwidth,
            'execution_time': total_time,
            'score': (total_bandwidth / reference_metrics['gpu_data_generation_bandwidth']) * 100
        }

        return result

    except RuntimeError as e:
        print(f"Error during GPU data generation: {e}")
        return None

def benchmark_gpu_to_cpu_transfer(data_size_gb, reference_metrics, precision, physical_gpu_ids=None):
    """
    Transfers large tensors from the GPUs to the CPU to benchmark PCIe bandwidth.
    """
    try:
        dtype = get_torch_dtype_from_precision(precision)

        physical_gpu_ids, physical_to_logical, status_message = resolve_requested_gpus(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            message = status_message or "No valid GPUs available."
            print(f"{message} Cannot run GPU to CPU Transfer benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        logical_gpu_ids = [physical_to_logical[physical_id] for physical_id in physical_gpu_ids]

        print(f"Using GPUs: {physical_gpu_ids} (logical IDs: {logical_gpu_ids}) for GPU to CPU Transfer")

        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements_per_gpu = int(((data_size_gb * 1e9) / element_size) / num_gpus)

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for physical_id in physical_gpu_ids:
            logical_id = physical_to_logical[physical_id]
            p = mp.Process(target=run_gpu_to_cpu_transfer_on_gpu, args=(logical_id, num_elements_per_gpu, dtype, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        transfer_times = [t for t in return_dict.values() if t is not None]
        if not transfer_times:
            print("No valid transfer times collected.")
            return None

        max_transfer_time = max(transfer_times)
        data_size_bytes = num_elements_per_gpu * element_size * num_gpus
        data_size_gb_actual = data_size_bytes / 1e9
        transfer_bandwidth = data_size_gb_actual / max_transfer_time if max_transfer_time > 0 else float('inf')

        metrics = f"Bandwidth: {transfer_bandwidth:.2f} GB/s"
        input_params = f"Data Size: {data_size_gb} GB, Precision: {precision}"

        result = {
            'task': 'GPU to CPU Transfer',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'data_size_gb': data_size_gb_actual,
            'time_seconds': max_transfer_time,
            'bandwidth_gb_per_second': transfer_bandwidth,
            'execution_time': max_transfer_time,
            'score': (transfer_bandwidth / reference_metrics['gpu_to_cpu_transfer_bandwidth']) * 100
        }

        return result

    except RuntimeError as e:
        print(f"Error during GPU to CPU transfer: {e}")
        return None

def benchmark_gpu_to_gpu_transfer(data_size_gb, reference_metrics, precision, physical_gpu_ids=None):
    """
    Measures GPU to GPU data transfer bandwidth.
    """
    try:
        dtype = get_torch_dtype_from_precision(precision)

        physical_gpu_ids, physical_to_logical, status_message = resolve_requested_gpus(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)

        # Ensure at least two GPUs are available
        if num_gpus < 2:
            if num_gpus == 0:
                message = status_message or "No valid GPUs available."
                print(f"{message} GPU to GPU Transfer benchmark requires at least two GPUs.")
            else:
                print("At least two GPUs are required for GPU to GPU Transfer benchmark.")
            return {
                'task': 'GPU to GPU Transfer',
                'category': 'GPU',
                'input_params': f"Data Size: {data_size_gb} GB, Precision: {precision}",
                'metrics': "Not Applicable",
                'bandwidth_gb_per_second': 0.0,
                'execution_time': 0.0,
                'score': 0,  # Return a score of 0 if the test cannot be run
                'error': 'Less than two GPUs detected.'
            }

        # Map physical GPU IDs to logical IDs
        physical_gpu0_id, physical_gpu1_id = physical_gpu_ids[:2]
        logical_gpu0_id = physical_to_logical[physical_gpu0_id]
        logical_gpu1_id = physical_to_logical[physical_gpu1_id]

        print(f"Using GPUs: {physical_gpu0_id} and {physical_gpu1_id} (logical IDs: {logical_gpu0_id} and {logical_gpu1_id}) for GPU to GPU Transfer")

        # Generate data on GPU 0
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = int((data_size_gb * 1e9) / element_size)
        iterations = 10  # Number of times to repeat the copy

        manager = mp.Manager()
        return_dict = manager.dict()
        p = mp.Process(target=run_gpu_to_gpu_transfer, args=(logical_gpu0_id, logical_gpu1_id, num_elements, iterations, dtype, return_dict))
        p.start()
        p.join()

        average_copy_time = return_dict.get('copy_time', None)
        if average_copy_time is None:
            print("Failed to collect copy time.")
            return {
                'task': 'GPU to GPU Transfer',
                'category': 'GPU',
                'input_params': f"Data Size: {data_size_gb} GB, Precision: {precision}",
                'metrics': "Error collecting copy time",
                'bandwidth_gb_per_second': 0.0,
                'execution_time': 0.0,
                'score': 0
            }

        data_size_bytes = num_elements * element_size
        data_size_gb_actual = data_size_bytes / 1e9
        bandwidth_gb_per_second = data_size_gb_actual / average_copy_time if average_copy_time > 0 else float('inf')

        input_params = f"Data Size: {data_size_gb} GB, Precision: {precision}"
        metrics = f"Bandwidth: {bandwidth_gb_per_second:.2f} GB/s"

        result = {
            'task': 'GPU to GPU Transfer',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'bandwidth_gb_per_second': bandwidth_gb_per_second,
            'execution_time': average_copy_time * iterations,
            'score': (bandwidth_gb_per_second / reference_metrics['gpu_to_gpu_transfer_bandwidth']) * 100
        }

        return result

    except Exception as e:
        print(f"Error during GPU to GPU transfer benchmarking: {e}")
        return {
            'task': 'GPU to GPU Transfer',
            'category': 'GPU',
            'input_params': f"Data Size: {data_size_gb} GB, Precision: {precision}",
            'metrics': f"Error: {str(e)}",
            'bandwidth_gb_per_second': 0.0,
            'execution_time': 0.0,
            'score': 0
        }


def benchmark_gpu_memory_bandwidth(data_size_gb, reference_metrics, precision):
    """
    Measures GPU memory bandwidth by performing large memory copy operations on the GPU.
    Accepts data size in gigabytes (GB) instead of megabytes (MB).
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA is not available. Cannot benchmark GPU Memory Bandwidth.")
            return None

        dtype = get_torch_dtype_from_precision(precision)
        device = torch.device('cuda:0')  # Assuming single GPU for this benchmark

        # Convert GB to bytes
        data_size_bytes = data_size_gb * 1024 * 1024 * 1024  # GB to bytes conversion
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = int(data_size_bytes // element_size)

        # Generate data on GPU
        src_tensor = torch.randn((num_elements,), device=device, dtype=dtype)
        src_tensor = torch.randn((num_elements,), device=device, dtype=dtype)

        # Warm-up
        dest_tensor = src_tensor.clone()

        # Synchronize to ensure all operations are completed
        torch.cuda.synchronize()

        # Measure copy bandwidth
        start_time = time.time()
        dest_tensor = src_tensor.clone()
        torch.cuda.synchronize()  # Ensure the operation is completed
        end_time = time.time()

        copy_time = end_time - start_time

        if copy_time == 0:
            bandwidth_gb_per_second = float('inf')
        else:
            bandwidth_gb_per_second = data_size_gb / copy_time

        input_params = f"Data Size: {data_size_gb} GB, Precision: {precision}"
        metrics = f"Bandwidth: {bandwidth_gb_per_second:.2f} GB/s"

        result = {
            'task': 'GPU Memory Bandwidth',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'bandwidth_gb_per_second': bandwidth_gb_per_second,
            'execution_time': copy_time,
            'score': (bandwidth_gb_per_second / reference_metrics['gpu_memory_bandwidth_gb_per_sec']) * 100
        }

        # Cleanup
        del src_tensor
        del dest_tensor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Error during GPU Memory Bandwidth benchmarking: {e}")
        return None

def benchmark_gpu_tensor_cores(matrix_size, num_iterations, reference_metrics, precision):
    """
    Benchmarks GPU Tensor Core performance using mixed-precision matrix multiplication.
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA is not available. Cannot benchmark GPU Tensor Cores.")
            return None

        dtype = get_torch_dtype_from_precision(precision)
        device = torch.device('cuda:0')  # Assuming single GPU for this benchmark

        # Generate random matrices
        A = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        B = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

        # Warm-up
        torch.cuda.synchronize()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Measure time for multiple iterations
        start_time = time.time()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time

        # Calculate GFLOPS
        total_flops = 2 * matrix_size ** 3 * num_iterations
        gflops = total_flops / total_time / 1e9

        input_params = f"Matrix Size: {matrix_size}, Iterations: {num_iterations}, Precision: {precision}"
        metrics = f"GFLOPS: {gflops:.2f}"

        result = {
            'task': 'GPU Tensor Core Performance',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'gflops': gflops,
            'execution_time': total_time,
            'score': (gflops / reference_metrics['tensor_core_gflops']) * 100
        }

        # Cleanup
        del A
        del B
        del C
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Error during GPU Tensor Core benchmarking: {e}")
        return None

def benchmark_gpu_computational_task(epochs, batch_size, input_size, hidden_size, output_size, reference_metrics, physical_gpu_ids=None, precision='fp16'):
    """
    Performs a computationally intensive task to benchmark GPU computational performance across multiple GPUs.
    """
    try:
        dtype = get_torch_dtype_from_precision(precision)

        physical_gpu_ids, physical_to_logical, status_message = resolve_requested_gpus(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            message = status_message or "No valid GPUs available."
            print(f"{message} Cannot run GPU Computational Task benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        logical_gpu_ids = [physical_to_logical[physical_id] for physical_id in physical_gpu_ids]

        def _distribute_batch_size(total_size, num_partitions):
            base = total_size // num_partitions
            remainder = total_size % num_partitions
            return [base + 1 if idx < remainder else base for idx in range(num_partitions)]

        per_gpu_batch_sizes = _distribute_batch_size(batch_size, num_gpus)
        active_indices = [idx for idx, size in enumerate(per_gpu_batch_sizes) if size > 0]

        if not active_indices:
            print("Batch size results in no work for any GPU. Cannot run GPU Computational Task benchmark.")
            return None

        active_physical_ids = [physical_gpu_ids[idx] for idx in active_indices]
        active_logical_ids = [logical_gpu_ids[idx] for idx in active_indices]
        active_per_gpu_batch_sizes = [per_gpu_batch_sizes[idx] for idx in active_indices]

        device_list = [torch.device(f'cuda:{logical_id}') for logical_id in active_logical_ids]
        primary_device = device_list[0]
        torch.cuda.set_device(primary_device)

        num_active_gpus = len(device_list)

        if num_active_gpus > 1:
            print(f"Using GPUs: {active_physical_ids} (logical IDs: {active_logical_ids}) for GPU Computational Task")
            per_gpu_summary = ", ".join(
                f"GPU {phys_id} (logical {log_id}): batch {batch_sz}"
                for phys_id, log_id, batch_sz in zip(active_physical_ids, active_logical_ids, active_per_gpu_batch_sizes)
            )
            print(f"Per-GPU batch sizes: {per_gpu_summary}")
        elif num_gpus > 1:
            print(
                f"Batch size {batch_size} only provides work for a single GPU. "
                f"Limiting computation to GPU {active_physical_ids[0]} (logical ID {active_logical_ids[0]})."
            )

        class SimpleModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        base_model = SimpleModel(input_size, hidden_size, output_size).to(primary_device, dtype=dtype)

        if num_active_gpus > 1:
            model = nn.DataParallel(base_model, device_ids=active_logical_ids, output_device=active_logical_ids[0])
        else:
            model = base_model

        criterion = nn.MSELoss().to(primary_device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Generate random data sized to cover the combined per-GPU workload
        total_batch_size = sum(active_per_gpu_batch_sizes)
        inputs = torch.randn(total_batch_size, input_size, device=primary_device, dtype=dtype)
        targets = torch.randn(total_batch_size, output_size, device=primary_device, dtype=dtype)

        # Warm-up
        model(inputs)

        # Training loop
        for device in device_list:
            torch.cuda.synchronize(device=device)
        start_time = time.time()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        for device in device_list:
            torch.cuda.synchronize(device=device)
        end_time = time.time()

        total_time = end_time - start_time

        # Calculate GFLOPS
        flops_per_sample = 2 * (input_size * hidden_size + hidden_size * output_size)
        per_gpu_flops = [flops_per_sample * size * epochs for size in active_per_gpu_batch_sizes]
        total_flops = sum(per_gpu_flops)
        gflops = total_flops / total_time / 1e9

        input_params = (f"Epochs: {epochs}, Batch Size: {batch_size}, Input Size: {input_size}, "
                        f"Hidden Size: {hidden_size}, Output Size: {output_size}, Precision: {precision}")
        metrics = f"GFLOPS: {gflops:.2f} across {num_active_gpus} GPU(s)"

        result = {
            'task': 'GPU Computational Task',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'performance_gflops': gflops,
            'execution_time': total_time,
            'score': (gflops / reference_metrics['computational_task_gflops']) * 100,
            'num_gpus_used': num_active_gpus,
            'per_gpu_batch_sizes': {
                physical_id: batch_sz
                for physical_id, batch_sz in zip(active_physical_ids, active_per_gpu_batch_sizes)
            }
        }

        # Cleanup
        del model
        del inputs
        del targets
        for device in device_list:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Error during GPU Computational Task benchmarking: {e}")
        return None

def run_inference_on_gpu(logical_gpu_id, model_name, model_size, batch_size_per_gpu, input_size, output_size, iterations, dtype, return_dict):
    try:
        if not torch.cuda.is_available():
            print(f"CUDA is not available on logical GPU {logical_gpu_id}.")
            return

        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)

        if model_name.lower() == 'custom':
            # Define custom model
            class ConvNet(nn.Module):
                def __init__(self, input_channels, num_classes, depth):
                    super(ConvNet, self).__init__()
                    layers = []
                    channels = input_channels
                    for _ in range(depth):
                        layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1))
                        layers.append(nn.ReLU())
                        layers.append(nn.MaxPool2d(2))
                        channels *= 2
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    self.features = nn.Sequential(*layers)
                    self.classifier = nn.Linear(channels, num_classes)

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            input_channels = 3
            num_classes = output_size
            depth = model_size

            model = ConvNet(input_channels, num_classes, depth).to(device, dtype=dtype)
        else:
            # Load model based on the provided model name
            if model_name.lower() == 'resnet50':
                from torchvision.models import resnet50
                model = resnet50(pretrained=False).to(device, dtype=dtype)
            elif model_name.lower() == 'bert':
                from transformers import BertModel, BertConfig
                config = BertConfig()
                model = BertModel(config).to(device, dtype=dtype)
            elif model_name.lower() == 'gpt2':
                from transformers import GPT2Model, GPT2Config
                config = GPT2Config()
                model = GPT2Model(config).to(device, dtype=dtype)
            else:
                print(f"Unsupported model name: {model_name}")
                return

        model.eval()

        # Generate input data
        if model_name.lower() in ['custom', 'resnet50']:
            inputs = torch.randn(batch_size_per_gpu, 3, input_size, input_size, device=device, dtype=dtype)
        elif model_name.lower() in ['bert', 'gpt2']:
            seq_length = input_size  # For BERT and GPT-2, use input_size as sequence length
            vocab_size = getattr(model.config, 'vocab_size', 30522)
            inputs = torch.randint(0, vocab_size, (batch_size_per_gpu, seq_length), device=device, dtype=torch.long)
        else:
            print(f"Unsupported model name: {model_name}")
            return

        # Handle dtype for inputs if necessary
        if dtype != torch.float32 and model_name.lower() in ['custom', 'resnet50']:
            inputs = inputs.to(dtype=dtype)

        # Warm-up
        with torch.no_grad():
            model(inputs)

        torch.cuda.synchronize()

        # Measure inference time
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        total_inference_time = sum(times)
        throughput = (batch_size_per_gpu * iterations) / total_inference_time  # Samples per second per GPU

        return_dict[logical_gpu_id] = {'throughput': throughput, 'time': total_inference_time}

    except Exception as e:
        print(f"Error on logical GPU {logical_gpu_id} during inference benchmarking: {e}")
        return_dict[logical_gpu_id] = None

def benchmark_inference_performance_multi_gpu(model_name, model_size, batch_size, input_size, output_size, iterations, reference_metrics, precision, physical_gpu_ids=None):
    """
    Measures inference performance by running separate processes on each GPU.
    """
    try:
        dtype = get_torch_dtype_from_precision(precision)

        physical_gpu_ids, physical_to_logical, status_message = resolve_requested_gpus(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            message = status_message or "No valid GPUs available."
            print(f"{message} Cannot run GPU Inference Performance benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        logical_gpu_ids = [physical_to_logical[physical_id] for physical_id in physical_gpu_ids]

        print(f"Using GPUs: {physical_gpu_ids} (logical IDs: {logical_gpu_ids}) for GPU Inference Performance")

        # Determine batch size per GPU
        batch_size_per_gpu = batch_size // num_gpus
        if batch_size_per_gpu == 0:
            print(f"Batch size {batch_size} is too small for {num_gpus} GPUs.")
            return None

        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for physical_id in physical_gpu_ids:
            logical_id = physical_to_logical[physical_id]
            p = mp.Process(target=run_inference_on_gpu, args=(
                logical_id,
                model_name,
                model_size,
                batch_size_per_gpu,
                input_size,
                output_size,
                iterations,
                dtype,
                return_dict
            ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Aggregate results
        valid_results = [v for v in return_dict.values() if v is not None]
        if not valid_results:
            _print_status("warning", "No valid inference results collected.")
            return None

        total_throughput = sum([v['throughput'] for v in valid_results])
        total_time = max([v['time'] for v in valid_results])  # Use max time since processes run in parallel
        avg_latency = total_time / iterations

        input_params = (f"Model: {model_name}, Model Size: {model_size}, Batch Size: {batch_size}, "
                        f"Input Size: {input_size}, Output Size: {output_size}, Precision: {precision}")
        metrics = f"Throughput: {total_throughput:.2f} samples/s"

        result = {
            'task': 'GPU Inference Performance',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'average_latency_seconds': avg_latency,
            'throughput_samples_per_second': total_throughput,
            'execution_time': total_time,
            'score': (total_throughput / reference_metrics['inference_throughput']) * 100
        }

        return result

    except Exception as e:
        print(f"Error during multi-GPU inference benchmarking: {e}")
        return None

# Main Function
def main():
    configure_logging()
    args = parse_arguments()

    # Set the default tensor type based on precision
    set_default_tensor_type(args.precision)
    dtype = get_torch_dtype_from_precision(args.precision)

    # Validate requested GPUs and manage CUDA visibility without clobbering the user's environment.
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested_gpu_ids = None
    if args.gpus is not None:
        requested_gpu_ids = [int(id.strip()) for id in args.gpus.split(',') if id.strip()]

    physical_gpu_ids, _, status_message = resolve_requested_gpus(requested_gpu_ids)
    physical_gpu_id_list = sorted(set(physical_gpu_ids))

    if physical_gpu_id_list:
        gpu_available = True
        visible_devices = ','.join(map(str, physical_gpu_id_list))
        if args.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            _print_status("gpu", f"Using GPU IDs {visible_devices}")
        else:
            if original_cuda_visible:
                _print_status(
                    "gpu",
                    f"CUDA_VISIBLE_DEVICES preset to {original_cuda_visible}; detected GPU IDs {visible_devices}",
                )
            else:
                _print_status("gpu", f"Detected GPU IDs {visible_devices}")
    else:
        if args.gpus is not None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        warning_message = status_message
        if not warning_message:
            if args.gpus is not None:
                warning_message = "No valid GPUs specified."
            else:
                warning_message = "No GPUs detected."
        _print_status("warning", f"{warning_message} GPU benchmarks will be skipped.")
        gpu_available = False

    # Determine which benchmarks to run
    benchmarks_specified = any([
        args.gpu_inference,
        args.disk_io,
        args.gpu_data_gen,
        args.gpu_to_cpu_transfer,
        args.cpu_to_disk_write,
        args.gpu_compute,
        args.cpu_single_thread,
        args.cpu_multi_thread,
        args.memory_bandwidth,
        args.gpu_tensor,
        args.gpu_memory_bandwidth,
        args.gpu_to_gpu_transfer
    ])

    if args.all or not benchmarks_specified:
        # No specific benchmarks specified, so run default set
        run_gpu_inference = True
        run_disk_io = True
        run_gpu_data_gen = True
        run_gpu_to_cpu_transfer = True
        run_cpu_to_disk_write = True
        run_gpu_compute = True
        run_cpu_single_thread = True
        run_cpu_multi_thread = True
        run_memory_bandwidth = True
        run_gpu_tensor = True
        run_gpu_memory_bandwidth = True
        run_gpu_to_gpu_transfer = True
    else:
        run_gpu_inference = args.gpu_inference
        run_disk_io = args.disk_io
        run_gpu_data_gen = args.gpu_data_gen
        run_gpu_to_cpu_transfer = args.gpu_to_cpu_transfer
        run_cpu_to_disk_write = args.cpu_to_disk_write
        run_gpu_compute = args.gpu_compute
        run_cpu_single_thread = args.cpu_single_thread
        run_cpu_multi_thread = args.cpu_multi_thread
        run_memory_bandwidth = args.memory_bandwidth
        run_gpu_tensor = args.gpu_tensor
        run_gpu_memory_bandwidth = args.gpu_memory_bandwidth
        run_gpu_to_gpu_transfer = args.gpu_to_gpu_transfer

    run_flags = {
        'disk_io': run_disk_io,
        'gpu_inference': run_gpu_inference,
        'gpu_benchmarks': any([
            run_gpu_inference,
            run_gpu_data_gen,
            run_gpu_to_cpu_transfer,
            run_gpu_to_gpu_transfer,
            run_gpu_memory_bandwidth,
            run_gpu_tensor,
            run_gpu_compute,
        ]),
        'log_gpu': args.log_gpu,
    }

    preflight_flags, skip_reasons = run_preflight_checks(args, run_flags, gpu_available)
    run_disk_io = preflight_flags.get('disk_io', run_disk_io)
    run_gpu_inference = preflight_flags.get('gpu_inference', run_gpu_inference)
    args.log_gpu = preflight_flags.get('log_gpu', args.log_gpu)

    gpu_benchmarks_requested = any([
        run_gpu_inference,
        run_gpu_data_gen,
        run_gpu_to_cpu_transfer,
        run_gpu_to_gpu_transfer,
        run_gpu_memory_bandwidth,
        run_gpu_tensor,
        run_gpu_compute,
    ])

    if skip_reasons:
        logger.info('Pre-flight skipped workloads due to missing dependencies:')
        for workload, reason in skip_reasons.items():
            logger.info('  %s: %s', workload, reason)

    # Start total execution timer
    total_start_time = time.time()

    # System Information
    try:
        system_info = get_system_info()
    except Exception as e:
        print(f"Error retrieving system information: {e}")
        system_info = {'error': str(e)}

    # Start GPU logging if enabled
    log_process = None
    if args.log_gpu:
        if gpu_available:
            log_process = start_gpu_logging(args.gpu_log_file, args.gpu_log_metrics)
            if log_process:
                _print_status("log", f"GPU logging started → {args.gpu_log_file}")
            else:
                _print_status("warning", "Failed to start GPU logging.")
        else:
            _print_status("warning", "GPU logging requested but no GPUs are available; skipping GPU logging.")

    # Reference Metrics (adjusted as needed)
    # 2024-09-26 1/4 of dual a16 / 12 vCore / 128G RAM / 700G NVMe
    reference_metrics = {
        'gpu_data_generation_bandwidth': 20.0,    # GB/s
        'gpu_to_cpu_transfer_bandwidth': 2.5,     # GB/s
        'gpu_to_gpu_transfer_bandwidth': 5.0,     # GB/s
        'cpu_to_disk_write_bandwidth': 0.25,      # GB/s
        'computational_task_gflops': 2500.0,      # GFLOPS
        'inference_throughput': 4000.0,           # Samples per second
        'sequential_read_throughput_mb_per_sec': 250.0,     # MB/s
        'sequential_write_throughput_mb_per_sec': 250.0,    # MB/s
        'random_read_iops': 10000.0,               # IOPS
        'random_write_iops': 10000.0,              # IOPS
        'cpu_single_thread_comp_perf': 175000.0,  # Fibonacci numbers per second
        'cpu_single_thread_crypto_perf': 200.0,   # MB/s
        'cpu_single_thread_data_proc_perf': 20.0, # MB/s
        'cpu_multi_thread_comp_perf': 600000.0,   # Fibonacci numbers per second
        'cpu_multi_thread_crypto_perf': 1000.0,    # MB/s
        'cpu_multi_thread_data_proc_perf': 100.0, # MB/s
        'memory_bandwidth_gb_per_sec': 3.0,       # GB/s
        'tensor_core_gflops': 5000.0,             # GFLOPS
        'gpu_memory_bandwidth_gb_per_sec': 40.0,  # GB/s
    }

    # Run benchmarks
    all_results = []
    for iteration in range(args.num_iterations):
        _print_status("iteration", f"{iteration + 1}/{args.num_iterations}", newline_before=True)
        results = []

        if run_gpu_data_gen:
            if gpu_available:
                _print_status("run", "GPU Data Generation benchmark")
                result = benchmark_gpu_data_generation(
                    args.gpu_data_size_gb,
                    reference_metrics,
                    args.precision,
                    physical_gpu_ids=physical_gpu_id_list
                )
                results.append(result)
            else:
                _print_status("skip", "GPU Data Generation benchmark: no GPUs detected.")

        if run_gpu_to_cpu_transfer:
            if gpu_available:
                _print_status("run", "GPU to CPU Transfer benchmark")
                result = benchmark_gpu_to_cpu_transfer(
                    args.gpu_data_size_gb,
                    reference_metrics,
                    args.precision,
                    physical_gpu_ids=physical_gpu_id_list
                )
                results.append(result)
            else:
                _print_status("skip", "GPU to CPU Transfer benchmark: no GPUs detected.")

        if run_gpu_to_gpu_transfer:
            if gpu_available:
                _print_status("run", "GPU to GPU Transfer benchmark")
                result = benchmark_gpu_to_gpu_transfer(
                    args.gpu_data_size_gb,
                    reference_metrics,
                    args.precision,
                    physical_gpu_ids=physical_gpu_id_list
                )
                results.append(result)
            else:
                _print_status("skip", "GPU to GPU Transfer benchmark: no GPUs detected.")

        if run_gpu_tensor:
            if gpu_available:
                _print_status("run", "GPU Tensor Core Performance benchmark")
                tensor_core_result = benchmark_gpu_tensor_cores(
                    matrix_size=args.gpu_tensor_matrix_size,
                    num_iterations=args.gpu_tensor_iterations,
                    reference_metrics=reference_metrics,
                    precision=args.precision
                )
                results.append(tensor_core_result)
            else:
                _print_status("skip", "GPU Tensor Core Performance benchmark: no GPUs detected.")

        if run_gpu_compute:
            if gpu_available:
                _print_status("run", "GPU Computational Task benchmark")
                computational_result = benchmark_gpu_computational_task(
                    epochs=args.gpu_comp_epochs,
                    batch_size=args.gpu_comp_batch_size,
                    input_size=args.gpu_comp_input_size,
                    hidden_size=args.gpu_comp_hidden_size,
                    output_size=args.gpu_comp_output_size,
                    reference_metrics=reference_metrics,
                    physical_gpu_ids=physical_gpu_id_list,
                    precision=args.precision
                )
                results.append(computational_result)
            else:
                _print_status("skip", "GPU Computational Task benchmark: no GPUs detected.")

        if run_gpu_inference:
            if gpu_available:
                _print_status("run", "GPU Inference Performance benchmark")
                inference_result = benchmark_inference_performance_multi_gpu(
                    model_name=args.gpu_inference_model,
                    model_size=args.model_size,
                    batch_size=args.batch_size,
                    input_size=args.input_size,
                    output_size=args.output_size,
                    iterations=args.iterations,
                    reference_metrics=reference_metrics,
                    precision=args.precision,
                    physical_gpu_ids=physical_gpu_id_list
                )
                results.append(inference_result)
            else:
                _print_status("skip", "GPU Inference Performance benchmark: no GPUs detected.")

        if run_gpu_memory_bandwidth:
            if gpu_available:
                _print_status("run", "GPU Memory Bandwidth benchmark")
                gpu_mem_bw_result = benchmark_gpu_memory_bandwidth(
                    data_size_gb=args.gpu_memory_size_gb,
                    reference_metrics=reference_metrics,
                    precision=args.precision
                )
                results.append(gpu_mem_bw_result)
            else:
                _print_status("skip", "GPU Memory Bandwidth benchmark: no GPUs detected.")

        if run_cpu_single_thread:
            _print_status("run", "CPU Single-threaded Performance benchmark")
            cpu_single_thread_result = benchmark_cpu_single_thread(reference_metrics)
            results.append(cpu_single_thread_result)

        if run_cpu_multi_thread:
            _print_status("run", "CPU Multi-threaded Performance benchmark")
            cpu_multi_thread_result = benchmark_cpu_multi_thread(reference_metrics, args.cpu_num_threads)
            results.append(cpu_multi_thread_result)

        if run_memory_bandwidth:
            _print_status("run", "Memory Bandwidth benchmark")
            memory_bandwidth_result = benchmark_memory_bandwidth(args.memory_size_mb_cpu, reference_metrics)
            results.append(memory_bandwidth_result)

        if run_cpu_to_disk_write:
            _print_status("run", "CPU to Disk Write benchmark")
            output_file = f'benchmark_output_{iteration}.bin'
            result = benchmark_cpu_to_disk_write(output_file, args.data_size_gb_cpu, reference_metrics)
            results.append(result)

        if run_disk_io:
            _print_status("run", "Disk I/O Performance benchmark")
            disk_file_path = f'disk_io_test_file_{iteration}.dat'
            disk_result = benchmark_disk_io(
                file_path=disk_file_path,
                data_size_gb=args.disk_data_size,
                block_size_kb=args.disk_block_size,
                io_depth=args.disk_io_depth,
                num_jobs=args.disk_num_jobs,
                reference_metrics=reference_metrics
            )
            results.append(disk_result)

        all_results.extend(results)

    # Stop GPU logging if it was started
    if args.log_gpu:
        stop_gpu_logging(log_process)
        _print_status("log", "GPU logging stopped.")

    # End total execution timer
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # Calculate total score
    total_score = sum([result['score'] for result in all_results if result and 'score' in result])

    # Print detailed results if requested
    if args.detailed_output:
        print_detailed_results(all_results)

    # Print results table
    print_results_table(all_results, total_score, total_execution_time)

    # System Information and Execution Time
    print_system_overview(system_info, detailed=args.detailed_output)
    _print_status("summary", f"Total execution time {total_execution_time:.2f} s", newline_before=True)

    if args.json:
        # Prepare output data
        output_data = {
            'results': all_results,
            'system_info': system_info,
            'total_execution_time_seconds': total_execution_time,
            'total_score': total_score,
            'skipped_workloads': skip_reasons
        }
        # Output results as JSON
        json_output = json.dumps(output_data, indent=4)
        _print_heading("JSON Output")
        print(json_output)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
