
# GPUBench

## Overview
GPUBench is a flexible tool built to test the performance of key hardware components including GPUs, CPUs, memory, and disk storage. It helps evaluate how well a system can handle AI and machine learning workloads, making it a valuable resource for engineers, data scientists, and system admins who want to optimize their computing setup. Ideal for ensuring everything runs smoothly in demanding environments, especially where high-performance matters most.

With comparative scoring, users can benchmark similar systems to ensure consistency, compare newer or dissimilar hardware to assess performance differences, and evaluate PaaS/IaaS providers against each other to achieve the optimal performance:$ ratio.

### Key Features:
- **GPU Memory Bandwidth**: Measure memory allocation and bandwidth across multiple GPUs.
- **GPU to CPU Transfer**: Test PCIe transfer speeds between GPU and CPU.
- **GPU to GPU Transfer**: Evaluate data transfer rates between GPUs.
- **Disk I/O**: Benchmark read/write performance of the system's storage.
- **Computationally Intensive Tasks**: Run deep learning models and synthetic tasks to test compute performance.
- **Model Inference**: Benchmark common AI models like ResNet, BERT, GPT-2 for inference throughput and latency.
- **CPU Performance**: Evaluate both single-threaded and multi-threaded CPU performance.
- **Memory Bandwidth**: Measure system memory performance.
- **Tensor Core Performance**: Benchmark GPU Tensor Core capabilities.

## Requirements and Setup

### System Requirements
- **Operating System**: Ubuntu 22.04/24.04 or Rocky/Alma Linux 9
- **Disk space**: At least 10GB of free disk space for benchmarking operations.
- **fio**: Flexible I/O Tester, used for disk I/O benchmarks.
- **nvidia-smi**: NVIDIA System Management Interface, used for GPU monitoring (typically installed with CUDA).
- **CUDA libraries**: Required for GPU operations (installed with CUDA toolkit).

### Python Dependencies
The following Python libraries are required:
- `torch`: PyTorch framework for deep learning operations.
- `numpy`: For numerical operations.
- `psutil`: For system and process utilities.
- `GPUtil`: To monitor GPU usage.
- `tabulate`: For formatting output as tables.
- `transformers`: For transformer models like BERT and GPT inference.
- `torchvision`: For ResNet and other image-related tasks.

### Installation Instructions

#### Rocky/Alma Linux 9

1. Install Python and Pip:
    ```bash
    sudo dnf install python3 python3-pip -y
    ```

2. Install CUDA:
   Follow the [CUDA Installation Guide for Rocky Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

3. Install Python dependencies:
    ```bash
    pip3 install torch numpy psutil GPUtil tabulate transformers torchvision
    ```

#### Ubuntu 22.04/24.04

1. Install Python and Pip:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip -y
    ```

2. Install CUDA:
   Follow the [CUDA Installation Guide for Ubuntu](https://developer.nvidia.com/cuda-downloads).

3. Install the required Python packages:
    ```bash
    pip3 install torch numpy psutil GPUtil tabulate transformers torchvision
    ```

## Command-Line Options

### General Options:
- `--json`: Output results in JSON format.
- `--detailed-output`: Show detailed benchmark results.
- `--num-iterations N`: Number of times to run the benchmarks (default: 1).
- `--log-gpu`: Enable GPU logging during benchmarks.
- `--gpu-log-file FILE`: Specify GPU log file name (default: 'gpu_log.csv').
- `--gpu-log-metrics METRICS`: Comma-separated list of GPU metrics to log.
- `--gpus GPU_IDS`: Comma-separated list of GPU IDs to use (e.g., "0,1,2,3").
- `--precision {fp16,fp32,fp64,bf16}`: Precision to use for computations (default: fp16).

### GPU Benchmarks:
- `--gpu-data-gen`: Run GPU Data Generation benchmark.
- `--gpu-to-cpu-transfer`: Run GPU to CPU Transfer benchmark.
- `--gpu-to-gpu-transfer`: Run GPU to GPU Transfer benchmark.
- `--gpu-memory-bandwidth`: Run GPU Memory Bandwidth benchmark.
- `--gpu-tensor`: Run GPU Tensor Core Performance benchmark.
- `--gpu-compute`: Run GPU Computational Task benchmark.
- `--gpu-data-size-gb N`: Data size in GB for GPU benchmarks (default: 5.0).
- `--gpu-memory-size-gb N`: Memory size in GB for GPU Memory Bandwidth benchmark (default: 5.0).
- `--gpu-tensor-matrix-size N`: Matrix size for GPU Tensor Core benchmark (default: 4096).
- `--gpu-tensor-iterations N`: Iterations for GPU Tensor Core benchmark (default: 1000).
- `--gpu-comp-epochs N`: Number of epochs for GPU computational task (default: 200).
- `--gpu-comp-batch-size N`: Batch size for GPU computational task (default: 2048).
- `--gpu-comp-input-size N`: Input size for GPU computational task (default: 4096).
- `--gpu-comp-hidden-size N`: Hidden layer size for GPU computational task (default: 4096).
- `--gpu-comp-output-size N`: Output size for GPU computational task (default: 2000).

### CPU Benchmarks:
- `--cpu-single-thread`: Run CPU Single-threaded Performance benchmark.
- `--cpu-multi-thread`: Run CPU Multi-threaded Performance benchmark.
- `--cpu-to-disk-write`: Run CPU to Disk Write benchmark.
- `--memory-bandwidth`: Run Memory Bandwidth benchmark.
- `--cpu-num-threads N`: Number of threads to use for multi-threaded CPU benchmark (default: all logical cores).
- `--data-size-gb-cpu N`: Data size in GB for CPU to Disk Write benchmark (default: 5.0).
- `--memory-size-mb-cpu N`: Memory size in MB for CPU Memory Bandwidth benchmark (default: 1024).

### Disk I/O Benchmarks:
- `--disk-io`: Run Disk I/O Performance benchmark.
- `--disk-data-size N`: Data size in GB for disk I/O benchmark (default: 2.0).
- `--disk-block-size N`: Block size in KB for disk I/O benchmark (default: 4).
- `--disk-io-depth N`: IO depth for disk I/O benchmark (default: 16).
- `--disk-num-jobs N`: Number of concurrent jobs for disk I/O benchmark (default: 8).

### Inference Benchmarks:
- `--gpu-inference`: Run GPU Inference Performance benchmark.
- `--gpu-inference-model {custom,resnet50,bert,gpt2}`: Model to use for inference benchmark (default: custom).
- `--model-size N`: Depth of the custom inference model (default: 5).
- `--batch-size N`: Batch size for inference benchmark (default: 256).
- `--input-size N`: Input size for inference benchmark (default: 224).
- `--output-size N`: Output size for inference benchmark (default: 1000).
- `--iterations N`: Number of iterations for inference benchmark (default: 100).

### Full Suite of Benchmarks:
To run all benchmarks:
```bash
python3 gpubench.py --all
```
## Example Usage:

#### GPU Memory Bandwidth Test:
```bash
python3 gpubench.py --gpu-memory-bandwidth --memory-size-mb 1024
```

#### CPU Multi-thread Performance Benchmark:
```bash
python3 gpubench.py --cpu-multi-thread --cpu-num-threads 8
```
#### Example Output:
- system: 12 vCPUs, 128G RAM, 700 GB NVMe, 2x A16
- executed: `python3 gpubench.py` (no options)

```
Benchmark Results:
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| Task                            | Input                          | Metrics                                           |   Exec Time (s) |   Score |
+=================================+================================+===================================================+=================+=========+
| === GPU Benchmarks ===          |                                |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU Data Generation             | Data Size: 5.0 GB, Precision:  | Bandwidth: 54.07 GB/s                             |            0.37 |   270.4 |
|                                 | fp16                           |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU to CPU Transfer             | Data Size: 5.0 GB, Precision:  | Bandwidth: 3.51 GB/s                              |            1.43 |   140.3 |
|                                 | fp16                           |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU to GPU Transfer             | Data Size: 5.0 GB, Precision:  | Bandwidth: 6.24 GB/s                              |            8.01 |   124.8 |
|                                 | fp16                           |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU Tensor Core Performance     | Matrix Size: 4096, Iterations: | GFLOPS: 14119.95                                  |            9.73 |   282.4 |
|                                 | 1000, Precision: fp16          |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU Computational Task          | Epochs: 200, Batch Size: 2048, | GFLOPS: 5342.96                                   |            3.83 |   213.7 |
|                                 | Input Size: 4096, Hidden Size: |                                                   |                 |         |
|                                 | 4096, Output Size: 2000,       |                                                   |                 |         |
|                                 | Precision: fp16                |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU Inference Performance       | Model: custom, Model Size: 5,  | Throughput: 8068.83 samples/s                     |            3.18 |   201.7 |
|                                 | Batch Size: 256, Input Size:   |                                                   |                 |         |
|                                 | 224, Output Size: 1000,        |                                                   |                 |         |
|                                 | Precision: fp16                |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| GPU Memory Bandwidth            | Data Size: 5.0 GB, Precision:  | Bandwidth: 80.00 GB/s                             |            0.01 |   200.0 |
|                                 | fp16                           |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| === System Benchmarks ===       |                                |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| CPU Single-threaded Performance | Single-threaded CPU Benchmark  | Comp Perf: 240821.10 fib/sec, Crypto Perf: 378.97 |            5.96 |   155.1 |
|                                 |                                | MB/s, Data Proc Perf: 27.61 MB/s                  |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| CPU Multi-threaded Performance  | Multi-threaded CPU Benchmark   | Comp Perf: 1755824.78 fib/sec, Crypto Perf:       |           11.71 |   279.3 |
|                                 | with 12 threads                | 3952.12 MB/s, Data Proc Perf: 150.15 MB/s         |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| Memory Bandwidth                | Memory Size: 1024 MB           | Bandwidth: 3.61 GB/s                              |            0.30 |   120.5 |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| CPU to Disk Write               | Data Size: 5.0 GB              | Bandwidth: 0.78 GB/s                              |            6.45 |   310.2 |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| Disk I/O Performance            | Data Size: 2.0 GB, Block Size: | Seq Read: 2099.06 MB/s, Seq Write: 2242.33 MB/s,  |          123.21 |  1485.3 |
|                                 | 4 KB, IO Depth: 16, Num Jobs:  | Rand Read IOPS: 219517, Rand Write IOPS: 200931   |                 |         |
|                                 | 8                              |                                                   |                 |         |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
| Total Score / Exec. Time        |                                |                                                   |          282.03 |  3783.7 |
+---------------------------------+--------------------------------+---------------------------------------------------+-----------------+---------+
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). 

```
Copyright (C) 2024 Liquid Web, LLC <deveng@liquidweb.com>
Copyright (C) 2024 Ryan MacDonald <rmacdonald@liquidweb.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

## Contributing

Contributions to GPUBench are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

