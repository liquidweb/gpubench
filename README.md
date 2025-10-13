
# GPUBench

## Overview
Modern AI/ML workloads demand reliable, high-performance infrastructure, but comparing GPUs, CPUs, memory, and storage across different servers or providers is often inconsistent and unclear. GPUBench solves this by providing a unified, reproducible benchmarking tool that measures real-world performance across compute, memory, disk, and inference tasks. This makes it easier to validate hardware, compare providers, and ensure systems deliver the performance required for demanding AI/ML applications.

Built as a flexible benchmarking suite, GPUBench tests the performance of key hardware components including GPUs, CPUs, memory, and disk storage. It helps evaluate how well a system can handle AI and machine learning workloads, making it a valuable resource for engineers, data scientists, and system admins who want to optimize their computing setup. Ideal for ensuring everything runs smoothly in demanding environments, GPUBench also enables comparative scoring to benchmark similar systems for consistency, assess performance differences across new or dissimilar hardware, and evaluate PaaS/IaaS providers against each other for the best performance-to-cost ratio

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
- **System Overview Snapshot**: Capture OS, CPU, GPU telemetry, storage, and environment metadata for reproducible benchmarking.

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

Refer to `requirements.txt` for the complete, version-pinned dependency set.

### Installation Instructions

The project follows standard Python virtual environment practices so that the
benchmark and its dependencies stay isolated from system packages. The steps
below install the system prerequisites, create a virtual environment, and
install the packages pinned in `requirements.txt`.

#### Rocky Linux 9 / AlmaLinux 9

1. Install Python tooling, Git, and `fio` (for disk benchmarks):
   ```bash
   sudo dnf install -y python3 python3-pip git fio
   ```

2. Install the CUDA drivers and toolkit that match your GPUs by following the
   official [CUDA Installation Guide for Rocky Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
   Confirm that `nvidia-smi` works before running GPUBench.

3. Create and activate a virtual environment in the repository:
   ```bash
   cd gpubench
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

4. Install the Python dependencies from the provided requirements file:
   ```bash
   python -m pip install -r requirements.txt
   ```

#### Ubuntu 22.04 / Ubuntu 24.04

1. Install Python tooling, Git, `fio`, and the virtual environment module:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip git fio
   ```

2. Install the NVIDIA driver and CUDA toolkit using the
   [CUDA Installation Guide for Ubuntu](https://developer.nvidia.com/cuda-downloads),
   then verify `nvidia-smi` reports your GPUs.

3. Create and activate a virtual environment inside the repository:
   ```bash
   cd gpubench
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

4. Install the Python dependencies listed in `requirements.txt`:
   ```bash
   python -m pip install -r requirements.txt
   ```

> **Optional packages:** GPU inference benchmarks that use ResNet, BERT, or GPT-2
> rely on `torchvision` and `transformers`. These are included in
> `requirements.txt` so inference workloads function out of the box. If you are
> curating a minimal environment, remove them from the requirements file before
> installing.

After activation, the `gpubench.py` CLI can be executed with `python gpubench.py`.
Reactivate the environment later with `source .venv/bin/activate`.

## Command-Line Options

### General Options:
- `--json`: Output results in JSON format.
- `--detailed-output`: Show detailed benchmark results and print an expanded system overview (disk partitions, network links, environment variables).
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

### GPU Inference Benchmarks:
- `--gpu-inference`: Run GPU inference throughput and latency benchmarks.
- `--gpu-inference-model {custom,resnet50,bert,gpt2}`: Select the model to benchmark (default: `custom`).
- `--model-size N`: Depth of the custom inference model (default: 5).
- `--batch-size N`: Batch size for the inference benchmark (default: 256).
- `--input-size N`: Input feature size for inference benchmark (default: 224).
- `--output-size N`: Output dimension for inference benchmark (default: 1000).
- `--iterations N`: Number of inference iterations to execute (default: 100).

### CPU Benchmarks:
- `--cpu-single-thread`: Run CPU single-threaded performance benchmark.
- `--cpu-multi-thread`: Run CPU multi-threaded performance benchmark.
- `--cpu-to-disk-write`: Run CPU to disk write throughput benchmark.
- `--memory-bandwidth`: Run memory bandwidth benchmark.
- `--cpu-num-threads N`: Threads used for multi-threaded CPU benchmark (default: all logical cores).
- `--data-size-gb-cpu N`: Data size in GB for CPU to disk write benchmark (default: 5.0).
- `--memory-size-mb-cpu N`: Memory size in MB for CPU memory bandwidth benchmark (default: 1024).

### Disk I/O Benchmarks:
- `--disk-io`: Run disk I/O benchmark via `fio`.
- `--disk-data-size N`: Data size in GB for disk I/O benchmark (default: 2.0).
- `--disk-block-size N`: Block size in KB for disk I/O benchmark (default: 4).
- `--disk-io-depth N`: Queue depth for disk I/O benchmark (default: 16).
- `--disk-num-jobs N`: Number of concurrent `fio` jobs to run (default: 8).
- `--disk-path PATH`: Target directory for the disk benchmark scratch files (default: current directory).

### Full Suite of Benchmarks:
To run all benchmarks:
```bash
python3 gpubench.py --all
```
## Example Usage:

#### GPU Memory Bandwidth Test:
```bash
python3 gpubench.py --gpu-memory-bandwidth --gpu-memory-size-gb 6
```

#### CPU Multi-thread Performance Benchmark:
```bash
python3 gpubench.py --cpu-multi-thread --cpu-num-threads 8
```

#### Run GPU Inference with ResNet50:
```bash
python3 gpubench.py --gpu-inference --gpu-inference-model resnet50 --batch-size 128
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

### System Overview Output

After the benchmark table, GPUBench prints a consolidated system summary that now captures:

- Hostname, operating system, kernel, uptime, and timestamp (UTC) for reproducibility.
- CPU architecture, physical/logical cores, frequency range, and load averages.
- Memory and swap utilisation, root disk usage, and (with `--detailed-output`) a breakdown of every mounted partition.
- GPU inventory with live telemetry from `nvidia-smi` when available (temperature, power draw/limits, SM & memory clocks, fan speeds, utilisation, and memory use).
- GPU driver and firmware (VBIOS) details plus per-device PyTorch properties (logical ID mapping, compute capability, SM count, thread limits).
- PyTorch runtime metadata (CUDA/cuDNN versions) alongside captured environment variables such as `CUDA_VISIBLE_DEVICES`.
- Optional network interface status (link state, speed, MTU) when `--detailed-output` is supplied.
- CPU scheduler counters (context switches, interrupts, syscalls) to aid in diagnosing contention-heavy runs.

This enhanced inventory helps share benchmark results with complete hardware and software context, enabling easier cross-environment comparisons and troubleshooting.

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

