# Related Works & Studies

## Common MLSys Framework

### DL

| Framework                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| Pytorch/MXNet/Tensorflow | ...                                                          |
| JAX                      | Still for research (JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla)) |
|                          |                                                              |

### Inference

| Framework                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [TensorRT](https://github.com/NVIDIA/TensorRT)               | Nvidia / Special Support for GPU                             |
| [AI Template](https://github.com/facebookincubator/AITemplate) | New from Meta (renders neural network into high performance CUDA/HIP C++ code. Specialized for FP16 TensorCore (NVIDIA GPU) and MatrixCore (AMD GPU) inference.) |
|                                                              |                                                              |

### Serving

| Framework                                                    | Description                                      |
| ------------------------------------------------------------ | ------------------------------------------------ |
| [triton-inference-server](https://github.com/triton-inference-server/server) | optimized cloud and edge inferencing from nvidia |
|                                                              |                                                  |

## Issues on Pytorch GH Repo

| Page                                                         | Description                                                  | Possible cause                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| [`torch.inverse` multi-threading RuntimeError: lazy wrapper should be called at most once ](https://github.com/pytorch/pytorch/issues/90613) | Multithreading error                                         | parallel unit testing **extra cuda synchronizations** |
| [`NotImplementedError` when using `torch.distributed.launch`for multiGPUs](https://github.com/pytorch/pytorch/issues/91408) | Data Parallel Error                                          | pytorch native `DistributedDataParallel` module       |
| [Option to let DistributedDataParallel know in advance unused parameters at each forward pass](https://github.com/pytorch/pytorch/issues/90171) | DistributedDataParallel Performance                          | find_unused_parameters=True                           |
| [PyTorch 2.1.0 Performance regression from PyTorch 2.0.1 · Issue #117081 · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/issues/117081) | Speed reduction for new pytorch version                      | ?                                                     |
| [RuntimeError: CUDA error: an illegal memory access was encountered using vmap and model ensembling call for cuda system](https://github.com/pytorch/pytorch/issues/116320) | Multiple models process multiple batches of data and models call for cuda to process data | ?                                                     |
| [Segmentation faults in DataLoader (in latest torch version)](https://github.com/pytorch/pytorch/issues/91245) | This happens with `num_workers=16` or `12`, `8`, `4`, `3`.   | ？                                                    |



## Concepts/Reference

| Page                                                         | form                      | Description |
| ------------------------------------------------------------ | ------------------------- | ----------- |
| [Solving real-world optimization tasks using physics-informed neural computing ](https://www.nature.com/articles/s41598-023-49977-3) | Scientific Reports        |             |
| [PyTorch distributed: experiences on accelerating data parallel training: Proceedings of the VLDB Endowment: Vol 13, No 12 (acm.org)](https://dl.acm.org/doi/10.14778/3415478.3415530) | VLDB                      |             |
| [BladeDISC: Optimizing Dynamic Shape Machine Learning Workloads via Compiler Approach ](https://dl.acm.org/doi/10.1145/3617327) | ACM on Management of Data |             |
| [EasyScale: Elastic Training with Consistent Accuracy and Improved Utilization on GPUs](https://dl.acm.org/doi/10.1145/3581784.3607054) | *SC '23*                  |             |

## Tools 

| Page                                                         | from      | Description                                                  |
| ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
|                                                              |           |                                                              |
| [PyTorch Model Performance Analysis and Optimization ](https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869) | Post      | Performance                                                  |
| [PyTorch Profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) |           | Tensorboard visualized pytorch profiler                      |
| [(beta) Pytorch Layer Profiler](https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html) |           | Detailed layer-to-layer profiler of pytorch                  |
| [microsoft/AI-System: System for AI Education Resource. (github.com)](https://github.com/microsoft/AI-System?tab=readme-ov-file) | Microsoft | An online AI System Course to help students learn the whole stack of systems that support AI |
|                                                              |           |                                                              |

## Multiple DDP Training with Torchrun

| Ways                                   |                                                              | Common Troubleshooting                                    |
| -------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| torchrun for multi-machine distributed | `-nproc_per_node=4\<br />nnodes=2<br />node_rank=0<br />rdzv_id=456<br />rdzv_backend=c10d<br />rdzv_endpoint=172.31.43.139:29603<br />multinode_torchrun.py 50 10` | * nodes communication<br />* network interface (firewall) |
| Slrum scheduler                        | `#SBATCH--job-name=multinode-example<br/>#SBATCH--nodes=4<br/>#SBATCH--ntasks=4<br/>#SBATCH--gpus-per-task=1#SBATCH --cpus-per-task=4<br/>nodes=( $( scontrol show hostnames $SLURM JOB NODELIST ) )nodes array=($nodes)head_node=$(nodes_array[0]}head node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address` | nodes bandwidth issues                                    |
|                                        |                                                              |                                                           |
|                                        |                                                              |                                                           |

