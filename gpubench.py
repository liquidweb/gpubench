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
import textwrap
import threading
import hashlib
import gzip
import numpy as np
import psutil
import GPUtil
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import multiprocessing as mp

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
    # CPU Information
    cpu_info = {
        'cpu_model': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
    }

    # RAM Information
    svmem = psutil.virtual_memory()
    ram_info = {
        'total_ram_gb': round(svmem.total / (1024 ** 3), 2),
    }

    # Disk Information
    disk_usage = psutil.disk_usage('/')
    disk_info = {
        'total_disk_gb': round(disk_usage.total / (1024 ** 3), 2),
    }

    # GPU Information
    gpus = GPUtil.getGPUs()
    gpu_info_list = []
    for gpu in gpus:
        gpu_info = {
            'id': gpu.id,
            'name': gpu.name,
            'total_memory_gb': round(gpu.memoryTotal / 1024, 2),
            'driver_version': gpu.driver,
            'cuda_version': torch.version.cuda,
        }
        gpu_info_list.append(gpu_info)

    system_info = {
        'cpu_info': cpu_info,
        'ram_info': ram_info,
        'disk_info': disk_info,
        'gpu_info': gpu_info_list,
    }

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
        print(f"Error starting GPU logging: {e}")
        return None

def stop_gpu_logging(log_process):
    try:
        if log_process:
            # Send termination signal
            log_process.terminate()
            # Wait for the process to terminate
            log_process.wait(timeout=5)
    except Exception as e:
        print(f"Error stopping GPU logging: {e}")

def print_detailed_results(results):
    print("\n=== Detailed Benchmark Results ===")
    for result in results:
        if result is None:
            continue
        task = result.get('task', 'Unknown Task')
        print(f"\n--- {task} ---")
        for key, value in result.items():
            if key not in ['task', 'category']:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")

def print_results_table(results, total_score, total_execution_time):
    # Prepare table data
    table_data = []
    max_width_input = 30  # Maximum width for "Input" column
    max_width_metrics = 50  # Maximum width for "Metrics" column

    # Separate GPU and System benchmarks
    gpu_results = [res for res in results if res.get('category') == 'GPU']
    system_results = [res for res in results if res.get('category') == 'System']

    # Function to process results and append to table data
    def process_results(result_list, header):
        table_data.append([header, '', '', '', ''])
        for result in result_list:
            if result is None:
                continue
            task = result.get('task', 'Unknown Task')
            input_params = result.get('input_params', '')
            score = result.get('score', 'N/A')
            execution_time = result.get('execution_time', 'N/A')

            # Build metric string with relevant metrics
            metrics = result.get('metrics', 'N/A')

            # Wrap text in "Input" and "Metrics" columns
            wrapped_input = '\n'.join(textwrap.wrap(input_params, width=max_width_input))
            wrapped_metric = '\n'.join(textwrap.wrap(metrics, width=max_width_metrics))

            # Format execution time and score
            execution_time_str = f"{float(execution_time):.2f}" if isinstance(execution_time, (int, float)) else 'N/A'
            score_str = f"{float(score):.1f}" if isinstance(score, (int, float)) else 'N/A'

            table_data.append([task, wrapped_input, wrapped_metric, execution_time_str, score_str])

    # Process GPU and System results
    process_results(gpu_results, '=== GPU Benchmarks ===')
    process_results(system_results, '=== System Benchmarks ===')

    # Add total score and total execution time
    total_execution_time_str = f"{float(total_execution_time):.2f}"
    total_score_str = f"{float(total_score):.1f}"
    table_data.append(['Total Score / Exec. Time', '', '', total_execution_time_str, total_score_str])

    headers = ["Task", "Input", "Metrics", "Exec Time (s)", "Score"]

    # Set column alignment
    colalign = ("left", "left", "left", "right", "right")

    # Print the table
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", colalign=colalign))

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
    gpu_group.add_argument('--gpu-memory-size-mb', type=int, default=1024,
                           help='Memory size in MB for GPU Memory Bandwidth benchmark (default: 1024)')
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

        # Use ThreadPoolExecutor for more efficient thread management
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Computational Task
            start_time = time.time()
            fib_results = list(executor.map(fibonacci, [n] * num_threads))
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
    available_gpus = [gpu.id for gpu in GPUtil.getGPUs()]
    valid_gpu_ids = []
    for gpu_id in gpu_ids:
        if gpu_id in available_gpus:
            valid_gpu_ids.append(gpu_id)
        else:
            print(f"Warning: GPU ID {gpu_id} is not available. It will be skipped.")
    return valid_gpu_ids

def map_physical_to_logical_gpu_ids(gpu_ids):
    """
    Maps physical GPU IDs to logical IDs after setting CUDA_VISIBLE_DEVICES.
    """
    physical_to_logical = {}
    for logical_id, physical_id in enumerate(gpu_ids):
        physical_to_logical[physical_id] = logical_id
    return physical_to_logical

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

        if physical_gpu_ids is None:
            if torch.cuda.is_available():
                physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
            else:
                physical_gpu_ids = []
        else:
            physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            print("No valid GPUs available for GPU Data Generation benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)
        logical_gpu_ids = list(range(num_gpus))

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

        if physical_gpu_ids is None:
            if torch.cuda.is_available():
                physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
            else:
                physical_gpu_ids = []
        else:
            physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            print("No valid GPUs available for GPU to CPU Transfer benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)
        logical_gpu_ids = list(range(num_gpus))

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

        if physical_gpu_ids is None:
            physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        else:
            physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)

        # Ensure at least two GPUs are available
        if len(physical_gpu_ids) < 2:
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
        physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)

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


def benchmark_gpu_memory_bandwidth(data_size_mb, reference_metrics, precision):
    """
    Measures GPU memory bandwidth by performing large memory copy operations on the GPU.
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA is not available. Cannot benchmark GPU Memory Bandwidth.")
            return None

        dtype = get_torch_dtype_from_precision(precision)
        device = torch.device('cuda:0')  # Assuming single GPU for this benchmark

        data_size = data_size_mb * 1024 * 1024  # Convert MB to bytes
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = data_size // element_size  # Number of elements

        # Generate data on GPU
        src_tensor = torch.randn(num_elements, device=device, dtype=dtype)

        # Warm-up
        dest_tensor = src_tensor.clone()

        # Measure copy bandwidth
        torch.cuda.synchronize()
        start_time = time.time()
        dest_tensor = src_tensor.clone()
        torch.cuda.synchronize()
        end_time = time.time()

        copy_time = end_time - start_time

        if copy_time == 0:
            bandwidth_gb_per_second = float('inf')
        else:
            bandwidth_gb_per_second = (data_size / copy_time) / 1e9

        input_params = f"Data Size: {data_size_mb} MB, Precision: {precision}"
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

        if physical_gpu_ids is None:
            if torch.cuda.is_available():
                physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
            else:
                physical_gpu_ids = []
        else:
            physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            print("No valid GPUs available for GPU Computational Task benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)
        logical_gpu_ids = list(range(num_gpus))

        device = torch.device(f'cuda:{logical_gpu_ids[0]}')

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

        model = SimpleModel(input_size, hidden_size, output_size).to(device, dtype=dtype)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Generate random data
        inputs = torch.randn(batch_size, input_size, device=device, dtype=dtype)
        targets = torch.randn(batch_size, output_size, device=device, dtype=dtype)

        # Warm-up
        model(inputs)

        # Training loop
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time

        # Calculate GFLOPS
        flops_per_sample = 2 * (input_size * hidden_size + hidden_size * output_size)
        total_flops = flops_per_sample * batch_size * epochs
        gflops = total_flops / total_time / 1e9

        input_params = (f"Epochs: {epochs}, Batch Size: {batch_size}, Input Size: {input_size}, "
                        f"Hidden Size: {hidden_size}, Output Size: {output_size}, Precision: {precision}")
        metrics = f"GFLOPS: {gflops:.2f}"

        result = {
            'task': 'GPU Computational Task',
            'category': 'GPU',
            'input_params': input_params,
            'metrics': metrics,
            'performance_gflops': gflops,
            'execution_time': total_time,
            'score': (gflops / reference_metrics['computational_task_gflops']) * 100
        }

        # Cleanup
        del model
        del inputs
        del targets
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

        if physical_gpu_ids is None:
            # Get available GPU IDs
            physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        else:
            physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)

        num_gpus = len(physical_gpu_ids)
        if num_gpus == 0:
            print("No valid GPUs available for GPU Inference Performance benchmark.")
            return None

        # Map physical GPU IDs to logical IDs
        physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)
        logical_gpu_ids = list(range(num_gpus))

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
            print("No valid inference results collected.")
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
    args = parse_arguments()

    # Set the default tensor type based on precision
    set_default_tensor_type(args.precision)
    dtype = get_torch_dtype_from_precision(args.precision)

    # Validate and set CUDA_VISIBLE_DEVICES
    if args.gpus is not None:
        physical_gpu_ids = [int(id.strip()) for id in args.gpus.split(',')]
        physical_gpu_ids = validate_gpu_ids(physical_gpu_ids)
        if not physical_gpu_ids:
            print("No valid GPUs specified. Exiting.")
            sys.exit(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, physical_gpu_ids))
        print(f"Using GPUs: {', '.join(map(str, physical_gpu_ids))}")
    else:
        physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        if not physical_gpu_ids:
            print("No GPUs found. Exiting.")
            sys.exit(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, physical_gpu_ids))
        print(f"Using all available GPUs: {', '.join(map(str, physical_gpu_ids))}")

    # After setting CUDA_VISIBLE_DEVICES, logical GPU IDs start from 0
    # Map physical GPU IDs to logical GPU IDs
    physical_to_logical = map_physical_to_logical_gpu_ids(physical_gpu_ids)
    logical_gpu_ids = list(range(len(physical_gpu_ids)))

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
        log_process = start_gpu_logging(args.gpu_log_file, args.gpu_log_metrics)
        if log_process:
            print(f"GPU logging started, writing to {args.gpu_log_file}")
        else:
            print("Failed to start GPU logging.")

    # Reference Metrics (adjusted as needed)
    reference_metrics = {
        'gpu_data_generation_bandwidth': 50.0,    # GB/s
        'gpu_to_cpu_transfer_bandwidth': 12.0,    # GB/s
        'gpu_to_gpu_transfer_bandwidth': 50.0,    # GB/s
        'cpu_to_disk_write_bandwidth': 0.25,      # GB/s
        'computational_task_gflops': 1000.0,      # GFLOPS
        'inference_throughput': 1000.0,           # Samples per second
        'sequential_read_throughput_mb_per_sec': 100.0,    # MB/s
        'sequential_write_throughput_mb_per_sec': 90.0,    # MB/s
        'random_read_iops': 5000.0,               # IOPS
        'random_write_iops': 4500.0,              # IOPS
        'cpu_single_thread_comp_perf': 500000.0,  # Fibonacci numbers per second
        'cpu_single_thread_crypto_perf': 100.0,   # MB/s
        'cpu_single_thread_data_proc_perf': 100.0,# MB/s
        'cpu_multi_thread_comp_perf': 500000.0,   # Fibonacci numbers per second
        'cpu_multi_thread_crypto_perf': 400.0,    # MB/s
        'cpu_multi_thread_data_proc_perf': 400.0, # MB/s
        'memory_bandwidth_gb_per_sec': 2.0,       # GB/s
        'tensor_core_gflops': 5000.0,             # GFLOPS
        'gpu_memory_bandwidth_gb_per_sec': 500.0, # GB/s
    }

    # Run benchmarks
    all_results = []
    for iteration in range(args.num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.num_iterations} ===")
        results = []

        if run_gpu_data_gen:
            print("Running GPU Data Generation benchmark...")
            result = benchmark_gpu_data_generation(
                args.gpu_data_size_gb,
                reference_metrics,
                args.precision,
                physical_gpu_ids=physical_gpu_ids
            )
            results.append(result)

        if run_gpu_to_cpu_transfer:
            print("Running GPU to CPU Transfer benchmark...")
            result = benchmark_gpu_to_cpu_transfer(
                args.gpu_data_size_gb,
                reference_metrics,
                args.precision,
                physical_gpu_ids=physical_gpu_ids
            )
            results.append(result)

        if run_gpu_to_gpu_transfer:
            print("Running GPU to GPU Transfer benchmark...")
            result = benchmark_gpu_to_gpu_transfer(
                args.gpu_data_size_gb,
                reference_metrics,
                args.precision,
                physical_gpu_ids=physical_gpu_ids
            )
            results.append(result)

        if run_gpu_tensor:
            print("Running GPU Tensor Core Performance benchmark...")
            tensor_core_result = benchmark_gpu_tensor_cores(
                matrix_size=args.gpu_tensor_matrix_size,
                num_iterations=args.gpu_tensor_iterations,
                reference_metrics=reference_metrics,
                precision=args.precision
            )
            results.append(tensor_core_result)

        if run_gpu_compute:
            print("Running GPU Computational Task benchmark...")
            computational_result = benchmark_gpu_computational_task(
                epochs=args.gpu_comp_epochs,
                batch_size=args.gpu_comp_batch_size,
                input_size=args.gpu_comp_input_size,
                hidden_size=args.gpu_comp_hidden_size,
                output_size=args.gpu_comp_output_size,
                reference_metrics=reference_metrics,
                physical_gpu_ids=physical_gpu_ids,
                precision=args.precision
            )
            results.append(computational_result)

        if run_gpu_inference:
            print("Running GPU Inference Performance benchmark...")
            inference_result = benchmark_inference_performance_multi_gpu(
                model_name=args.gpu_inference_model,
                model_size=args.model_size,
                batch_size=args.batch_size,
                input_size=args.input_size,
                output_size=args.output_size,
                iterations=args.iterations,
                reference_metrics=reference_metrics,
                precision=args.precision,
                physical_gpu_ids=physical_gpu_ids
            )
            results.append(inference_result)

        if run_gpu_memory_bandwidth:
            print("Running GPU Memory Bandwidth benchmark...")
            gpu_mem_bw_result = benchmark_gpu_memory_bandwidth(
                data_size_mb=args.gpu_memory_size_mb,
                reference_metrics=reference_metrics,
                precision=args.precision
            )
            results.append(gpu_mem_bw_result)

        if run_cpu_single_thread:
            print("Running CPU Single-threaded Performance benchmark...")
            cpu_single_thread_result = benchmark_cpu_single_thread(reference_metrics)
            results.append(cpu_single_thread_result)

        if run_cpu_multi_thread:
            print("Running CPU Multi-threaded Performance benchmark...")
            cpu_multi_thread_result = benchmark_cpu_multi_thread(reference_metrics, args.cpu_num_threads)
            results.append(cpu_multi_thread_result)

        if run_memory_bandwidth:
            print("Running Memory Bandwidth benchmark...")
            memory_bandwidth_result = benchmark_memory_bandwidth(args.memory_size_mb_cpu, reference_metrics)
            results.append(memory_bandwidth_result)

        if run_cpu_to_disk_write:
            print("Running CPU to Disk Write benchmark...")
            output_file = f'benchmark_output_{iteration}.bin'
            result = benchmark_cpu_to_disk_write(output_file, args.data_size_gb_cpu, reference_metrics)
            results.append(result)

        if run_disk_io:
            print("Running Disk I/O Performance benchmark...")
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
        print("GPU logging stopped.")

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
    if args.detailed_output:
        print("\n=== System Information ===")
        if 'error' in system_info:
            print(f"System Information Error: {system_info['error']}")
        else:
            print(f"CPU Model: {system_info['cpu_info']['cpu_model']}")
            print(f"CPU Cores: {system_info['cpu_info']['cpu_cores']}")
            print(f"CPU Threads: {system_info['cpu_info']['cpu_threads']}")
            print(f"Total RAM (GB): {system_info['ram_info']['total_ram_gb']}")
            print(f"Total Disk (GB): {system_info['disk_info']['total_disk_gb']}")

            for gpu in system_info['gpu_info']:
                print(f"GPU {gpu['id']}:")
                print(f"  Name: {gpu['name']}")
                print(f"  Total Memory (GB): {gpu['total_memory_gb']}")
                print(f"  Driver Version: {gpu['driver_version']}")
                print(f"  CUDA Version: {gpu['cuda_version']}")

        print(f"\nTotal Execution Time: {total_execution_time:.2f} seconds")

    if args.json:
        # Prepare output data
        output_data = {
            'results': all_results,
            'system_info': system_info,
            'total_execution_time_seconds': total_execution_time,
            'total_score': total_score
        }
        # Output results as JSON
        json_output = json.dumps(output_data, indent=4)
        print("\nJSON Output:")
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
