
# GPUBench

## Overview
GPUBench is a flexible tool built to test the performance of key hardware components including GPUs, CPUs, memory, and disk storage. It helps evaluate how well a system can handle AI and machine learning workloads, making it a valuable resource for engineers, data scientists, and system admins who want to optimize their computing setup. Ideal for ensuring everything runs smoothly in demanding environments, especially where high-performance matters most.

With comparative scoring, users can benchmark similar systems to ensure consistency, compare newer or dissimilar hardware to assess performance differences, and evaluate PaaS/IaaS providers against each other to achieve the optimal performance:$ ratio.

### Key Features:
- **GPU Memory Bandwidth**: Measure memory allocation and bandwidth across multiple GPUs.
- **GPU to CPU Transfer**: Test PCIe transfer speeds between GPU and CPU.
- **Disk I/O**: Benchmark read/write performance of the system's storage.
- **Computationally Intensive Tasks**: Run deep learning models and synthetic tasks to test compute performance.
- **Model Inference**: Benchmark common AI models like ResNet, BERT, GPT-2 for inference throughput and latency.

## Dependencies and Setup

### Python Dependencies
The following Python libraries are required:
- `torch` (PyTorch framework)
- `numpy` (for numerical operations)
- `psutil` (for system and process utilities)
- `GPUtil` (to monitor GPU usage)
- `tabulate` (for formatting output as tables)
- `transformers` (optional, for transformer models like BERT and GPT inference)
- `torchvision` (optional, for ResNet and other image-related tasks)

### System Requirements
- **Python 3.x**: The script requires Python 3.x.
- **CUDA**: Ensure CUDA is properly installed to take full advantage of GPU benchmarks.
  
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

GPUBench provides a variety of options to customize benchmarking for your specific use case. Below is a detailed list of available command-line arguments.

### General Options:
- `--json`: Output results in JSON format.
- `--detailed-output`: Show detailed benchmark results.
- `--num-iterations N`: Number of times to run the benchmarks (default: 1).
- `--log-gpu`: Enable GPU logging during benchmarks.
- `--gpu-log-file FILE`: Specify GPU log file name (default: 'gpu_log.csv').
- `--gpu-log-metrics METRICS`: Comma-separated list of GPU metrics to log (default: `timestamp,pstate,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.max.graphics,power.draw,clocks_throttle_reasons.active`).
- `--gpus GPU_IDS`: Comma-separated list of GPU IDs to use (e.g., `0,1,2,3`).

### GPU Benchmarks:
- `--gpu-data-gen`: Run GPU Data Generation benchmark.
- `--gpu-to-cpu-transfer`: Run GPU to CPU Transfer benchmark.
- `--gpu-memory-bandwidth`: Run GPU Memory Bandwidth benchmark.
- `--tensor-core`: Run Tensor Core Performance benchmark.
- `--data-size-gb N`: Data size in GB for GPU benchmarks (default: 5.0).
- `--memory-size-mb N`: Memory size in MB for GPU Memory Bandwidth benchmark (default: 1024).
- `--tensor-core-matrix-size N`: Matrix size for Tensor Core benchmark (default: 4096).
- `--tensor-core-iterations N`: Iterations for Tensor Core benchmark (default: 1000).

### CPU Benchmarks:
- `--computational-task`: Run Computationally Intensive Task benchmark.
- `--cpu-single-thread`: Run CPU Single-threaded Performance benchmark.
- `--cpu-multi-thread`: Run CPU Multi-threaded Performance benchmark.
- `--cpu-to-disk-write`: Run CPU to Disk Write benchmark.
- `--memory-bandwidth`: Run Memory Bandwidth benchmark.
- `--cpu-num-threads N`: Number of threads to use for multi-threaded CPU benchmark (default: all logical cores).
- `--data-size-gb-cpu N`: Data size in GB for CPU to Disk Write benchmark (default: 5.0).
- `--memory-size-mb-cpu N`: Memory size in MB for CPU Memory Bandwidth benchmark (default: 1024).
- `--comp-epochs N`: Number of epochs for computational task (default: 200).
- `--comp-batch-size N`: Batch size for computational task (default: 2048).
- `--comp-input-size N`: Input size for computational task (default: 4096).
- `--comp-hidden-size N`: Hidden layer size for computational task (default: 4096).
- `--comp-output-size N`: Output size for computational task (default: 2000).

### Disk I/O Benchmarks:
- `--disk-io`: Run Disk I/O Performance benchmark.
- `--disk-data-size N`: Data size in GB for disk I/O benchmark (default: 4.0).
- `--disk-block-size N`: Block size in KB for disk I/O benchmark (default: 4).
- `--disk-io-depth N`: IO depth for disk I/O benchmark (default: 16).
- `--disk-num-jobs N`: Number of concurrent jobs for disk I/O benchmark (default: 8).

### Model-Specific Benchmarks:
- `--resnet-inference`: Run ResNet50 Inference benchmark.
- `--bert-inference`: Run BERT Inference benchmark.
- `--gpt-inference`: Run GPT-2 Inference benchmark.
- `--resnet-batch-size N`: Batch size for ResNet50 benchmark (default: 64).
- `--resnet-input-size N`: Input size for ResNet50 benchmark (default: 224).
- `--resnet-iterations N`: Iterations for ResNet50 benchmark (default: 100).
- `--bert-batch-size N`: Batch size for BERT benchmark (default: 16).
- `--bert-seq-length N`: Sequence length for BERT benchmark (default: 128).
- `--bert-iterations N`: Iterations for BERT benchmark (default: 100).
- `--gpt-batch-size N`: Batch size for GPT-2 benchmark (default: 8).
- `--gpt-seq-length N`: Sequence length for GPT-2 benchmark (default: 128).
- `--gpt-iterations N`: Iterations for GPT-2 benchmark (default: 100).

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
12 vCPUs, 128G RAM, 700 GB NVMe, 2x A16
```
Benchmark Results:
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Task                            | Input                          | Metrics                                            |   Exec Time (s) |   Score |
+=================================+================================+====================================================+=================+=========+
| GPU Data Generation             | Data Size: 5.0 GB              | Bandwidth: 101.19 GB/s                             |            0.24 |   202.4 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| GPU to CPU Transfer             | Data Size: 5.0 GB              | Bandwidth: 3.43 GB/s                               |            1.46 |    28.6 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| GPU to GPU Transfer             | Data Size: 5.0 GB              | Bandwidth: 6.24 GB/s                               |            8.01 |    12.5 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| CPU to Disk Write               | Data Size: 5.0 GB              | Bandwidth: 0.86 GB/s                               |            5.84 |   342.5 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Computationally Intensive Task  | Epochs: 200, Total Batch Size: | GFLOPS: 3058.29                                    |          103.84 |   305.8 |
|                                 | 2048, Input Size: 4096, Hidden |                                                    |                 |         |
|                                 | Size: 4096, Output Size: 2000, |                                                    |                 |         |
|                                 | GPUs: 2                        |                                                    |                 |         |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Disk I/O Performance            | Data Size: 4000000000 bytes,   | Seq Read: 2044.28 MB/s, Seq Write: 2256.65 MB/s,   |           245.5 |  3379.3 |
|                                 | Block Size: 4 KB, IO Depth:    | Rand Read IOPS: 221380, Rand Write IOPS: 204206    |                 |         |
|                                 | 16, Num Jobs: 8                |                                                    |                 |         |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| CPU Single-threaded Performance | Single-threaded CPU Benchmark  | Comp Perf: 240643.14 fib/sec, Crypto Perf: 379.44  |            5.96 |   151.7 |
|                                 |                                | MB/s, Data Proc Perf: 27.65 MB/s                   |                 |         |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| CPU Multi-threaded Performance  | Multi-threaded CPU Benchmark   | Comp Perf: 1724948.15 fib/sec, Crypto Perf: 321.57 |           15.36 |   154.1 |
|                                 | with 12 threads                | MB/s, Data Proc Perf: 147.26 MB/s                  |                 |         |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Memory Bandwidth                | Memory Size: 1024 MB           | Bandwidth: 3.41 GB/s                               |            0.31 |   170.7 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Tensor Core Performance         | Matrix Size: 4096, Iterations: | GFLOPS: 14167.44                                   |             9.7 |   283.3 |
|                                 | 1000                           |                                                    |                 |         |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| GPU Memory Bandwidth            | Data Size: 1024 MB             | Bandwidth: 79.33 GB/s                              |            0.01 |    15.9 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+
| Total Score / Exec. Time        |                                |                                                    |           428.6 |  5046.8 |
+---------------------------------+--------------------------------+----------------------------------------------------+-----------------+---------+

```
#### Full Benchmark Suite:
```bash
python3 gpubench.py --all
```

## License

This project is licensed under the GNU GPL v3 License. See the LICENSE file for more information.

```
Licensed under the GNU GPL v3.
Copyright (C) 2024 Liquid Web, LLC <deveng@liquidweb.com>
Ryan MacDonald <rmacdonald@liquidweb.com>
```
