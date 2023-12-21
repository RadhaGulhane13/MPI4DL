# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os, sys
import math
import argparse


COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

import torch.distributed as dist


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default


def init_torch_distributed(backend):
    mpi_discovery()
    dist.init_process_group(backend)
    local_rank = int(os.environ['LOCAL_RANK'])
#     get_accelerator().set_device(local_rank)



def init_processes(local_rank, args):
   init_torch_distributed(args.backend)



def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def print_header(args, comm_op):
    if comm_op == 'pt2pt':
        world_size = 2
    else:
        world_size = dist.get_world_size()
    tput = f'Throughput ({args.bw_unit})'
    busbw = f'BusBW ({args.bw_unit})'
    header = f"\n---- Performance of {comm_op} on {world_size} devices ---------------------------------------------------------\n"
    duration_str = 'Duration'
    if args.raw:
        duration_str += ' (us)'
    header += f"{'Size (Bytes)':20s} {'Description':25s} {duration_str:20s} {tput:20s} {busbw:20s}\n"
    header += "----------------------------------------------------------------------------------------------------"
    print_rank_0(header)


def get_bw(comm_op, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")
        exit(0)

    if args.bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8

    return tput, busbw


def get_metric_strings(args, tput, busbw, duration):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    tput = f'{tput / 1e9:.3f}'
    busbw = f'{busbw /1e9:.3f}'

    if duration_us < 1e3 or args.raw:
        duration = f'{duration_us:.3f}'
        if not args.raw:
            duration += ' us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()

def total_memory(self, device_index=None):
        return torch.cuda.get_device_properties(device_index).total_memory

def max_numel(comm_op, dtype, mem_factor, local_rank, args):
    dtype_size = _element_size(dtype)
    max_memory_per_gpu = total_memory(local_rank) * mem_factor
    if comm_op == 'all_reduce' or comm_op == 'pt2pt' or comm_op == 'broadcast':
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
    elif comm_op == 'all_gather':
        # all_gather performance is lower for non-powers of two, and the output buffer size scales with world size
        # Therefore, divide by world size and round down to nearest power of 2
        elements_per_gpu = int(max_memory_per_gpu // dtype_size // dist.get_world_size())
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    elif comm_op == 'all_to_all':
        # Number of elements must be divisible by world_size
        # all_to_all performance is lower for non-powers of two. Round down like all_gather.
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
        elements_per_gpu = int(dist.get_world_size() * round(elements_per_gpu / dist.get_world_size()))
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    else:
        print(f"This communication operation: {comm_op} is not supported yet")
        exit(0)
    return elements_per_gpu


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# Copied from torch. Need to add the func here for old torch compatibility.
def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f'expected torch.dtype, but got {type(dtype)}')

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3

def mpi_discovery(distributed_port=29500, verbose=True):
    """
    Discovery MPI environment via mpi4py and map to relevant dist state
    """
    from mpi4py import MPI
    import subprocess

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    print("inside mpi_discovery....")

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode("utf-8").split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(distributed_port)
    # os.environ["NCCL_IB_PCI_RELAXED_ORDERING"] = "0"
    # os.environ['NCCL_IB_DISABLE']= str(1)
    # os.environ['NCCL_SOCKET_IFNAME'] = "eth0"
    print(f" RANK : {rank}  WORLD_SIZE : {world_size} LOCAL_RANK : {local_rank} MASTER_ADDR :{master_addr} MASTER_PORT : {distributed_port} NCCL_SOCKET_IFNAME : {os.environ.get('NCCL_SOCKET_IFNAME', -1)}")

def benchmark_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trials", type=int, default=50, help='Number of timed iterations')
    parser.add_argument("--warmups", type=int, default=5, help='Number of warmup (non-timed) iterations')
    parser.add_argument("--maxsize", type=int, default=24, help='Max message size as a power of 2')
    parser.add_argument("--async-op", action="store_true", help='Enables non-blocking communication')
    parser.add_argument("--bw-unit", type=str, default='Gbps', choices=['Gbps', 'GBps'])
    parser.add_argument("--backend",
                        type=str,
                        default='nccl',
                        choices=['nccl', 'mpi'],
                        help='Communication library to use')
    parser.add_argument("--dist",
                        type=str,
                        default='torch',
                        choices=['mcr_dl', 'torch'],
                        help='Distributed DL framework to use')
    parser.add_argument("--scan", action="store_true", help='Enables scanning all message sizes')
    parser.add_argument("--raw", action="store_true", help='Print the message size and latency without units')
    parser.add_argument("--all-reduce", action="store_true", help='Run all_reduce')
    parser.add_argument("--all-gather", action="store_true", help='Run all_gather')
    parser.add_argument("--all-to-all", action="store_true", help='Run all_to_all')
    parser.add_argument("--pt2pt", action="store_true", help='Run pt2pt')
    parser.add_argument("--broadcast", action="store_true", help='Run broadcast')
    parser.add_argument("--dtype", type=str, default='float', help='PyTorch tensor dtype')
    parser.add_argument("--mem-factor",
                        type=float,
                        default=.3,
                        help='Proportion of max available GPU memory to use for single-size evals')
    parser.add_argument("--debug", action="store_true", help='Enables all_to_all debug prints')
    return parser