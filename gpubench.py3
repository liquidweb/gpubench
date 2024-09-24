#!/usr/bin/env python3

import os
import sys
import time
import json
import torch
import argparse
import platform
import subprocess
import torch.nn as nn
import torch.optim as optim
import psutil
import GPUtil
import textwrap
import hashlib
import gzip
import threading
import concurrent.futures
import numpy as np
from tabulate import tabulate

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

def benchmark_gpu_data_generation(data_size_gb, reference_metrics):
    """
    Generates large tensors of random numbers directly on the GPUs to benchmark GPU memory bandwidth.
    - Allocates tensors of size `data_size_gb` GB divided equally across available GPUs.
    - Measures the time taken to generate the data.
    """
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for GPU Data Generation")
        else:
            print("Using 1 GPU for GPU Data Generation")

        # Calculate total number of elements per GPU
        num_elements_per_gpu = int(((data_size_gb * 1e9) / 4) / num_gpus)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()

        tensors = []
        for i in range(num_gpus):
            with torch.cuda.device(i):
                tensor = torch.randn(num_elements_per_gpu, device=f'cuda:{i}', dtype=torch.float32)
                tensors.append(tensor)

        torch.cuda.synchronize()
        end = time.time()

        gen_time = end - start
        data_size_bytes = num_elements_per_gpu * 4 * num_gpus  # float32
        data_size_gb_actual = data_size_bytes / 1e9
        gen_bandwidth = data_size_gb_actual / gen_time if gen_time > 0 else float('inf')

        result = {
            'task': 'GPU Data Generation',
            'input_params': f'Data Size: {data_size_gb} GB',
            'data_size_gb': data_size_gb_actual,
            'time_seconds': gen_time,
            'bandwidth_gb_per_second': gen_bandwidth,
            'execution_time': gen_time,
            'score': (gen_bandwidth / reference_metrics['gpu_data_generation_bandwidth']) * 100
        }

        # Cleanup
        del tensors
        torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        print(f"Error during GPU data generation: {e}")
        return None

def benchmark_gpu_to_cpu_transfer(data_size_gb, reference_metrics):
    """
    Transfers large tensors from the GPUs to the CPU to benchmark PCIe bandwidth.
    - Generates tensors of size `data_size_gb` GB divided equally across GPUs.
    - Transfers the tensors to the CPU.
    - Measures the time taken for the transfer.
    """
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for GPU to CPU Transfer")
        else:
            print("Using 1 GPU for GPU to CPU Transfer")

        num_elements_per_gpu = int(((data_size_gb * 1e9) / 4) / num_gpus)

        gpu_data = []
        for i in range(num_gpus):
            with torch.cuda.device(i):
                data = torch.randn(num_elements_per_gpu, device=f'cuda:{i}', dtype=torch.float32)
                gpu_data.append(data)

        torch.cuda.synchronize()
        start = time.time()

        cpu_data = []
        for data in gpu_data:
            cpu_data.append(data.cpu())

        torch.cuda.synchronize()
        end = time.time()

        transfer_time = end - start
        data_size_bytes = num_elements_per_gpu * 4 * num_gpus  # float32
        data_size_gb_actual = data_size_bytes / 1e9
        transfer_bandwidth = data_size_gb_actual / transfer_time if transfer_time > 0 else float('inf')

        result = {
            'task': 'GPU to CPU Transfer',
            'input_params': f'Data Size: {data_size_gb} GB',
            'data_size_gb': data_size_gb_actual,
            'time_seconds': transfer_time,
            'bandwidth_gb_per_second': transfer_bandwidth,
            'execution_time': transfer_time,
            'score': (transfer_bandwidth / reference_metrics['gpu_to_cpu_transfer_bandwidth']) * 100
        }

        # Cleanup
        del gpu_data
        del cpu_data
        torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        print(f"Error during GPU to CPU transfer: {e}")
        return None

def benchmark_cpu_to_disk_write(file_path, data_size_gb, reference_metrics):
    """
    Writes data from CPU to disk to benchmark disk write performance.
    - Generates a large tensor on the CPU.
    - Writes the tensor to disk.
    - Measures the time taken for the write operation.
    """
    try:
        # Calculate total number of elements
        num_elements = int((data_size_gb * 1e9) / 4)

        # Generate data on CPU
        cpu_data = torch.randn(num_elements, dtype=torch.float32)

        # Write data to disk
        start = time.time()
        with open(file_path, 'wb') as f:
            f.write(cpu_data.numpy().tobytes())
        end = time.time()

        write_time = end - start
        data_size_bytes = num_elements * 4  # float32
        data_size_gb_actual = data_size_bytes / 1e9
        write_bandwidth = data_size_gb_actual / write_time if write_time > 0 else float('inf')

        input_params = f'Data Size: {data_size_gb} GB'

        result = {
            'task': 'CPU to Disk Write',
            'input_params': input_params,
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

def benchmark_computational_task(epochs, batch_size, input_size, hidden_size, output_size, reference_metrics):
    """
    Performs a computationally intensive task to benchmark GPU computational performance.
    - Defines a deep neural network with multiple layers.
    - Uses DataParallel to utilize multiple GPUs if available.
    - Trains the network for a number of epochs using synthetic data.
    - Measures the time taken for the training.
    - Calculates the approximate GFLOPS achieved during training.
    """
    try:
        # Initialize CUDA context
        torch.cuda.current_device()

        # Define the neural network
        class ComplexNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(ComplexNet, self).__init__()
                self.layers = nn.ModuleList()
                # Add multiple layers to increase depth
                self.layers.append(nn.Linear(input_size, hidden_size))
                for _ in range(10):  # Increase the number of layers
                    self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.Linear(hidden_size, output_size))
                self.relu = nn.ReLU()

            def forward(self, x):
                for layer in self.layers[:-1]:
                    x = self.relu(layer(x))
                x = self.layers[-1](x)
                return x

        model = ComplexNet(input_size, hidden_size, output_size)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for Computationally Intensive Task")
            model = nn.DataParallel(model).cuda()
        else:
            print("Using 1 GPU for Computationally Intensive Task")
            model = model.cuda()

        criterion = nn.MSELoss().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Generate random input and target tensors
        inputs = torch.randn(batch_size, input_size, device='cuda')
        targets = torch.randn(batch_size, output_size, device='cuda')

        # Warm-up
        torch.cuda.synchronize()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # Measure time for multiple epochs
        torch.cuda.synchronize()
        start = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        end = time.time()

        total_time = end - start

        # Adjusted FLOPS calculation
        # Total FLOPs per epoch (approximate): 2 * number of operations in forward pass
        total_operations = 0
        model_layers = model.module.layers if num_gpus > 1 else model.layers
        for layer in model_layers:
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                # For Linear layers: 2 * in_features * out_features multiplications/additions per sample
                total_operations += 2 * in_features * out_features
        total_operations *= batch_size  # Multiply by batch size
        total_flops = total_operations * epochs * 2  # Multiply by 2 for forward and backward passes

        gflops = total_flops / total_time / 1e9

        input_params = (f"Epochs: {epochs}, Batch Size: {batch_size}, Input Size: {input_size}, "
                        f"Hidden Size: {hidden_size}, Output Size: {output_size}")

        result = {
            'task': 'Computationally Intensive Task',
            'input_params': input_params,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers': len(model_layers),
            'gpu_time_seconds': total_time,
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

    except RuntimeError as e:
        print(f"Error during computational task: {e}")
        return None

def benchmark_inference_performance(model_size, batch_size, input_size, output_size, iterations, reference_metrics):
    """
    Measures the inference latency and throughput of a neural network model.
    - Uses a convolutional neural network with specified depth.
    - Uses DataParallel to utilize multiple GPUs if available.
    - Runs inference for a number of iterations and measures latency and throughput.
    """
    try:
        # Calculate the maximum depth based on input size to prevent output size from becoming zero
        max_depth = int(torch.log2(torch.tensor(input_size / 4)))  # Adjusted to prevent output size from being zero
        if model_size > max_depth:
            print(f"Adjusted model size from {model_size} to {max_depth} to prevent output size from becoming zero.")
            model_size = max_depth

        # Define a convolutional neural network for inference benchmarking
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

        input_channels = 3  # Simulating RGB images
        num_classes = output_size
        depth = model_size

        model = ConvNet(input_channels, num_classes, depth)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for Inference Performance")
            model = nn.DataParallel(model).cuda()
        else:
            print("Using 1 GPU for Inference Performance")
            model = model.cuda()

        model.eval()  # Set model to evaluation mode

        # Generate random input tensor simulating high-resolution images
        inputs = torch.randn(batch_size, input_channels, input_size, input_size, device='cuda')

        # Warm-up
        torch.cuda.synchronize()
        with torch.no_grad():
            model(inputs)
        torch.cuda.synchronize()

        # Measure inference time over multiple iterations
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
        avg_latency = total_inference_time / iterations
        throughput = (batch_size * iterations) / total_inference_time  # Samples per second

        input_params = (f"Model Size: {model_size}, Batch Size: {batch_size}, Input Size: {input_size}, "
                        f"Output Size: {output_size}, Iterations: {iterations}")

        result = {
            'task': 'Inference Performance',
            'input_params': input_params,
            'average_latency_seconds': avg_latency,
            'throughput_samples_per_second': throughput,
            'execution_time': total_inference_time,
            'score': (throughput / reference_metrics['inference_throughput']) * 100
        }

        # Cleanup
        del model
        del inputs
        torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        print(f"Error during inference performance benchmarking: {e}")
        return None

def benchmark_disk_io(file_path, data_size_gb, block_size_kb, io_depth, num_jobs, reference_metrics):
    """
    Measures disk read/write throughput and IOPS for sequential and random access patterns.
    - Uses fio for sequential and random I/O benchmarking.
    - Allows customization of data size, block size, IO depth, and concurrency.
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
            print(f"Running {test['name'].replace('_', ' ').title()} benchmark...")
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
                '--runtime=60',
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

        input_params = (f"Data Size: {data_size_bytes} bytes, Block Size: {block_size_kb} KB, "
                        f"IO Depth: {io_depth}, Num Jobs: {num_jobs}")

        result = {
            'task': 'Disk I/O Performance',
            'input_params': input_params,
            'execution_time': total_execution_time,
            **results
        }

        # Calculate scores based on sequential read throughput and random read IOPS
        seq_read_throughput_score = (results['sequential_read_throughput_mb_per_sec'] / reference_metrics['sequential_read_throughput_mb_per_sec']) * 100
        rand_read_iops_score = (results['random_read_iops'] / reference_metrics['random_read_iops']) * 100

        # Average the scores
        result['score'] = (seq_read_throughput_score + rand_read_iops_score) / 2

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        print(f"Error during disk I/O benchmarking: {e}")
        return None

def benchmark_cpu_single_thread(reference_metrics):
    """
    Performs single-threaded CPU benchmarks covering computational, cryptographic, and data processing tasks.
    - Computational: Calculates large Fibonacci numbers using recursion.
    - Cryptographic: Computes SHA-256 hashes of large data blocks.
    - Data Processing: Compresses and decompresses data using gzip.
    """
    try:
        total_time = 0.0
        # Computational Task: Fibonacci Calculation
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n - 1) + fibonacci(n - 2)
        n = 35  # Adjust for desired computation time
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
        comp_perf = n / comp_time  # Fibonacci number per second
        crypto_perf = (data_size_mb / crypto_time)  # MB hashed per second
        data_proc_perf = (data_size_mb / data_proc_time)  # MB processed per second

        input_params = "Single-threaded CPU Benchmark"

        result = {
            'task': 'CPU Single-threaded Performance',
            'input_params': input_params,
            'fib_number': n,
            'fib_result': fib_result,
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

def benchmark_cpu_multi_thread(reference_metrics, num_threads):
    """
    Performs multi-threaded CPU benchmarks covering computational, cryptographic, and data processing tasks.
    - Computational: Calculates large Fibonacci numbers using recursion.
    - Cryptographic: Computes SHA-256 hashes of large data blocks.
    - Data Processing: Compresses and decompresses data using gzip.
    """
    try:
        total_time = 0.0

        # Computational Task: Fibonacci Calculation
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n - 1) + fibonacci(n - 2)

        n = 35  # Adjust for desired computation time
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            futures = [executor.submit(fibonacci, n) for _ in range(num_threads)]
            fib_results = [f.result() for f in futures]
            comp_time = time.time() - start_time
        total_time += comp_time

        # Cryptographic Task: SHA-256 Hashing
        data_size_mb = 100
        data_blocks = [os.urandom(data_size_mb * 1024 * 1024) for _ in range(num_threads)]
        def hash_data(data_block):
            return hashlib.sha256(data_block).hexdigest()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            hash_results = list(executor.map(hash_data, data_blocks))
            crypto_time = time.time() - start_time
        total_time += crypto_time

        # Data Processing Task: Gzip Compression/Decompression
        def compress_decompress(data_block):
            compressed = gzip.compress(data_block)
            decompressed = gzip.decompress(compressed)
            return decompressed

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            decompressed_results = list(executor.map(compress_decompress, data_blocks))
            data_proc_time = time.time() - start_time
        total_time += data_proc_time

        # Calculate performance metrics
        comp_perf = (n * num_threads) / comp_time  # Fibonacci numbers per second
        crypto_perf = (data_size_mb * num_threads) / crypto_time  # MB hashed per second
        data_proc_perf = (data_size_mb * num_threads) / data_proc_time  # MB processed per second

        input_params = f"Multi-threaded CPU Benchmark with {num_threads} threads"

        result = {
            'task': 'CPU Multi-threaded Performance',
            'input_params': input_params,
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

        # Cleanup
        del data_blocks
        del decompressed_results

        return result

    except Exception as e:
        print(f"Error during multi-threaded CPU benchmarking: {e}")
        return None

def benchmark_memory_bandwidth(memory_size_mb, reference_metrics):
    """
    Measures memory bandwidth by performing large memory copy operations.
    - Allocates large numpy arrays and measures the time to copy them.
    """
    try:
        data_size = memory_size_mb * 1024 * 1024  # Convert MB to bytes
        array_size = data_size // 8  # Number of elements (float64)

        # Generate data
        src_array = np.random.rand(array_size)

        # Warm-up
        dest_array = np.copy(src_array)

        # Measure memory copy bandwidth
        start_time = time.time()
        dest_array = np.copy(src_array)
        end_time = time.time()

        copy_time = end_time - start_time
        bandwidth_gb_per_sec = (data_size / copy_time) / 1e9

        input_params = f"Memory Size: {memory_size_mb} MB"

        result = {
            'task': 'Memory Bandwidth',
            'input_params': input_params,
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
    for result in results:
        if result is None:
            continue
        task = result.get('task', 'Unknown Task')
        print(f"\n=== {task} ===")
        for key, value in result.items():
            if key != 'task':
                print(f"{key.replace('_', ' ').capitalize()}: {value}")

def print_results_table(results, total_score, total_execution_time):
    # Adjusted function to condense table output within 120 characters
    table_data = []
    max_width_input = 30  # Maximum width for "Input" column
    max_width_metrics = 50  # Maximum width for "Metrics" column

    for result in results:
        if result is None:
            continue
        task = result.get('task', 'Unknown Task')
        input_params = result.get('input_params', '')
        score = result.get('score', 'N/A')
        execution_time = result.get('execution_time', 'N/A')

        # Build metric string with relevant metrics
        if task == 'GPU Data Generation':
            metric = f"Bandwidth: {result['bandwidth_gb_per_second']:.2f} GB/s"
        elif task == 'GPU to CPU Transfer':
            metric = f"Bandwidth: {result['bandwidth_gb_per_second']:.2f} GB/s"
        elif task == 'CPU to Disk Write':
            metric = f"Bandwidth: {result['bandwidth_gb_per_second']:.2f} GB/s"
        elif task == 'Computationally Intensive Task':
            metric = f"GFLOPS: {result['performance_gflops']:.2f}"
        elif task == 'Inference Performance':
            metric = f"Throughput: {result['throughput_samples_per_second']:.2f} samples/s"
        elif task == 'Disk I/O Performance':
            metric = (
                f"Seq Read: {result.get('sequential_read_throughput_mb_per_sec', 0):.2f} MB/s, "
                f"Seq Write: {result.get('sequential_write_throughput_mb_per_sec', 0):.2f} MB/s, "
                f"Rand Read IOPS: {int(result.get('random_read_iops', 0))}, "
                f"Rand Write IOPS: {int(result.get('random_write_iops', 0))}"
            )
        elif task == 'CPU Single-threaded Performance':
            metric = (
                f"Comp Perf: {result['comp_perf']:.2f} fib/sec, "
                f"Crypto Perf: {result['crypto_perf_mb_per_sec']:.2f} MB/s, "
                f"Data Proc Perf: {result['data_proc_perf_mb_per_sec']:.2f} MB/s"
            )
        elif task == 'CPU Multi-threaded Performance':
            metric = (
                f"Comp Perf: {result['comp_perf']:.2f} fib/sec, "
                f"Crypto Perf: {result['crypto_perf_mb_per_sec']:.2f} MB/s, "
                f"Data Proc Perf: {result['data_proc_perf_mb_per_sec']:.2f} MB/s"
            )
        elif task == 'Memory Bandwidth':
            metric = f"Bandwidth: {result['bandwidth_gb_per_sec']:.2f} GB/s"
        else:
            metric = 'N/A'

        # Wrap text in "Input" and "Metrics" columns
        wrapped_input = '\n'.join(textwrap.wrap(input_params, width=max_width_input))
        wrapped_metric = '\n'.join(textwrap.wrap(metric, width=max_width_metrics))

        # Format execution time and score
        execution_time_str = f"{float(execution_time):.2f}" if isinstance(execution_time, (int, float)) else 'N/A'
        score_str = f"{float(score):.1f}" if isinstance(score, (int, float)) else 'N/A'

        table_data.append([task, wrapped_input, wrapped_metric, execution_time_str, score_str])

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

def main():
    parser = argparse.ArgumentParser(
        description='GPU Benchmarking Script with CPU and Memory Bandwidth Tests',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--json', action='store_true',
                        help='Output results in JSON format')
    parser.add_argument('--detailed-output', action='store_true',
                        help='Show detailed benchmark results (default: False)')
    parser.add_argument('--data-size-gb', type=float, default=5.0,
                        help='Data size in GB for benchmarks (default: 5.0)\n'
                             'Used in GPU data generation, GPU to CPU transfer, and CPU to disk write tests.')
    parser.add_argument('--comp-batch-size', type=int, default=2048,
                        help='Batch size for computational task (default: 2048)\n'
                             'Determines the number of samples processed at once during training.')
    parser.add_argument('--comp-hidden-size', type=int, default=4096,
                        help='Hidden layer size for computational task (default: 4096)\n'
                             'Number of neurons in each hidden layer of the neural network.')
    parser.add_argument('--num-iterations', type=int, default=1,
                        help='Number of times to run the benchmarks (default: 1)')
    parser.add_argument('--comp-epochs', type=int, default=200,
                        help='Number of epochs for computational task (default: 200)\n'
                             'Defines how many times the entire dataset is passed through the network.')
    parser.add_argument('--comp-input-size', type=int, default=4096,
                        help='Input size for computational task (default: 4096)\n'
                             'Number of features in the input data.')
    parser.add_argument('--comp-output-size', type=int, default=2000,
                        help='Output size for computational task (default: 2000)\n'
                             'Number of output classes or regression targets.')
    parser.add_argument('--log-gpu', action='store_true',
                        help='Enable GPU logging during benchmarks (default log file: gpu_log.csv)')
    parser.add_argument('--gpu-log-file', type=str, default='gpu_log.csv',
                        help='Specify GPU log file name (default: gpu_log.csv)')
    parser.add_argument('--gpu-log-metrics', type=str,
                        default='timestamp,pstate,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.max.graphics,power.draw,clocks_throttle_reasons.active',
                        help='Comma-separated list of GPU metrics to log (default metrics are shown below):\n'
                             'timestamp,pstate,temperature.gpu,utilization.gpu,clocks.current.graphics,\n'
                             'clocks.max.graphics,power.draw,clocks_throttle_reasons.active')
    # Arguments for benchmarks
    parser.add_argument('--inference', action='store_true',
                        help='Run inference performance benchmark')
    parser.add_argument('--inf-model-size', type=int, default=5,
                        help='Depth of the inference model (number of convolutional layers, default: 5)')
    parser.add_argument('--inf-batch-size', type=int, default=256,
                        help='Batch size for inference benchmark (default: 256)')
    parser.add_argument('--inf-input-size', type=int, default=224,
                        help='Input size (height and width) for inference benchmark (default: 224)')
    parser.add_argument('--inf-output-size', type=int, default=1000,
                        help='Output size for inference benchmark (default: 1000)')
    parser.add_argument('--inf-iterations', type=int, default=100,
                        help='Number of iterations for inference benchmark (default: 100)')

    parser.add_argument('--disk-io', action='store_true',
                        help='Run disk I/O performance benchmark')
    parser.add_argument('--disk-data-size', type=float, default=4.0,
                        help='Data size in GB for disk I/O benchmark (default: 4.0)')
    parser.add_argument('--disk-block-size', type=int, default=4,
                        help='Block size in KB for disk I/O benchmark (default: 4)')
    parser.add_argument('--disk-io-depth', type=int, default=16,
                        help='IO depth for disk I/O benchmark (default: 16)')
    parser.add_argument('--disk-num-jobs', type=int, default=8,
                        help='Number of concurrent jobs for disk I/O benchmark (default: 8)')

    # New flags for other benchmarks
    parser.add_argument('--gpu-data-gen', action='store_true', help='Run GPU Data Generation benchmark')
    parser.add_argument('--gpu-to-cpu-transfer', action='store_true', help='Run GPU to CPU Transfer benchmark')
    parser.add_argument('--cpu-to-disk-write', action='store_true', help='Run CPU to Disk Write benchmark')
    parser.add_argument('--computational-task', action='store_true', help='Run Computationally Intensive Task benchmark')

    # New flags for CPU and memory benchmarks
    parser.add_argument('--cpu-single-thread', action='store_true', help='Run CPU Single-threaded Performance benchmark')
    parser.add_argument('--cpu-multi-thread', action='store_true', help='Run CPU Multi-threaded Performance benchmark')
    parser.add_argument('--cpu-num-threads', type=int, default=psutil.cpu_count(logical=True),
                        help='Number of threads to use for multi-threaded CPU benchmark (default: all logical cores)')

    parser.add_argument('--memory-bandwidth', action='store_true', help='Run Memory Bandwidth benchmark')
    parser.add_argument('--memory-size-mb', type=int, default=1024,
                        help='Memory size in MB for memory bandwidth benchmark (default: 1024)')

    # Argument to specify GPUs
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3"). If not set, use all available GPUs.')

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES before any CUDA operations
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
    else:
        print("Using all available GPUs")

    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner

    # Determine which benchmarks to run
    if not any([
        args.inference,
        args.disk_io,
        args.gpu_data_gen,
        args.gpu_to_cpu_transfer,
        args.cpu_to_disk_write,
        args.computational_task,
        args.cpu_single_thread,
        args.cpu_multi_thread,
        args.memory_bandwidth
    ]):
        # No specific benchmarks specified, so run all
        run_inference = True
        run_disk_io = True
        run_gpu_data_gen = True
        run_gpu_to_cpu_transfer = True
        run_cpu_to_disk_write = True
        run_computational_task = True
        run_cpu_single_thread = True
        run_cpu_multi_thread = True
        run_memory_bandwidth = True
    else:
        run_inference = args.inference
        run_disk_io = args.disk_io
        run_gpu_data_gen = args.gpu_data_gen
        run_gpu_to_cpu_transfer = args.gpu_to_cpu_transfer
        run_cpu_to_disk_write = args.cpu_to_disk_write
        run_computational_task = args.computational_task
        run_cpu_single_thread = args.cpu_single_thread
        run_cpu_multi_thread = args.cpu_multi_thread
        run_memory_bandwidth = args.memory_bandwidth

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

    # Define reference metrics for scoring
    reference_metrics = {
        'gpu_data_generation_bandwidth': 110.0,    # GB/s
        'gpu_to_cpu_transfer_bandwidth': 0.5,      # GB/s
        'cpu_to_disk_write_bandwidth': 0.2,        # GB/s
        'computational_task_gflops': 7000.0,       # GFLOPS
        'inference_throughput': 13000.0,           # Samples per second
        'sequential_read_throughput_mb_per_sec': 110.0,    # MB/s
        'random_read_iops': 24000.0,               # IOPS
        'cpu_single_thread_comp_perf': 1.0,        # Fibonacci numbers per second (example value)
        'cpu_single_thread_crypto_perf': 50.0,     # MB/s (example value)
        'cpu_single_thread_data_proc_perf': 50.0,  # MB/s (example value)
        'cpu_multi_thread_comp_perf': 10.0,        # Fibonacci numbers per second (example value)
        'cpu_multi_thread_crypto_perf': 500.0,     # MB/s (example value)
        'cpu_multi_thread_data_proc_perf': 500.0,  # MB/s (example value)
        'memory_bandwidth_gb_per_sec': 10.0,       # GB/s (example value)
    }

    # Run benchmarks for N iterations
    all_results = []
    for iteration in range(args.num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.num_iterations} ===")
        results = []

        if run_gpu_data_gen:
            print("Running GPU Data Generation benchmark...")
            # Benchmark GPU Data Generation
            result = benchmark_gpu_data_generation(args.data_size_gb, reference_metrics)
            results.append(result)

        if run_gpu_to_cpu_transfer:
            print("Running GPU to CPU Transfer benchmark...")
            # Benchmark GPU to CPU Transfer
            result = benchmark_gpu_to_cpu_transfer(args.data_size_gb, reference_metrics)
            results.append(result)

        if run_cpu_to_disk_write:
            print("Running CPU to Disk Write benchmark...")
            # Benchmark CPU to Disk Write
            output_file = f'benchmark_output_{iteration}.bin'
            result = benchmark_cpu_to_disk_write(output_file, args.data_size_gb, reference_metrics)
            results.append(result)

        if run_computational_task:
            print("Running Computationally Intensive Task benchmark...")
            # Benchmark Computationally Intensive Task
            computational_result = benchmark_computational_task(
                epochs=args.comp_epochs,
                batch_size=args.comp_batch_size,
                input_size=args.comp_input_size,
                hidden_size=args.comp_hidden_size,
                output_size=args.comp_output_size,
                reference_metrics=reference_metrics
            )
            results.append(computational_result)

        # Run inference performance benchmark if enabled
        if run_inference:
            print("Running Inference Performance benchmark...")
            inference_result = benchmark_inference_performance(
                model_size=args.inf_model_size,
                batch_size=args.inf_batch_size,
                input_size=args.inf_input_size,
                output_size=args.inf_output_size,
                iterations=args.inf_iterations,
                reference_metrics=reference_metrics
            )
            results.append(inference_result)

        # Run disk I/O performance benchmark if enabled
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
            memory_bandwidth_result = benchmark_memory_bandwidth(args.memory_size_mb, reference_metrics)
            results.append(memory_bandwidth_result)

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

            for idx, gpu in enumerate(system_info['gpu_info']):
                print(f"GPU {idx}:")
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
