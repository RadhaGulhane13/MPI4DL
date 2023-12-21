# Copyright 2023, The Ohio State University. All rights reserved.
# The MPI4DL software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os
import math
import argparse
from utils import *

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

def sync_all():
    torch.cuda.synchronize()
    dist.barrier()

def get_parser():
    parser = argparse.ArgumentParser(
        description="Halo exchange benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce",
    )

    parser.add_argument("--image-size", type=int, default=8, help="Full image size")
    parser.add_argument("--batch-size", type=int, default=1, help="input batch size")
    parser.add_argument("--halo-len", type=int, default=1, help="halo length")
    parser.add_argument(
        "--in-channels", type=int, default=1, help="Number of channels in the input"
    )
    parser.add_argument("--warmup", type=int, default=10, help="warmups")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument(
        "--out-channels", type=int, default=256, help="number of output channels"
    )
    parser.add_argument(
        "--num-spatial-parts",
        type=int,
        default="4",
        help="Number of partitions in spatial parallelism",
    )
    parser.add_argument(
        "--slice-method",
        type=str,
        default="square",
        help="Slice method (square, vertical, and horizontal) in Spatial parallelism",
    )

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


parser_obj = get_parser()
args = parser_obj.parse_args()
halo_len = args.halo_len
iterations = args.iterations
image_size = (args.image_size, args.image_size)
num_spatial_parts = args.num_spatial_parts
slice_method = args.slice_method


def validate_config(num_spatial_parts, comm_size):
    assert (
        num_spatial_parts > 1
    ), "num_spatial_parts should be greater than 1 for spatial parallelism."
    assert (
        comm_size >= num_spatial_parts
    ), "Spatial parts {num_spatial_parts} require {num_spatial_parts} GPUs."

    if slice_method == "square":
        parts = int(math.sqrt(num_spatial_parts))
        assert (
            parts * parts == num_spatial_parts
        ), "Invalid number of spatial parts for square spatial type"


class halo_bench_pt2pt:
    def __init__(
        self, local_rank, comm_size, num_spatial_parts, slice_method, halo_len
    ):
        self.local_rank = local_rank
        self.comm_size = comm_size
        # number of parts in one image
        self.num_spatial_parts = num_spatial_parts
        self.slice_method = slice_method
        self.halo_len = halo_len
        self.shapes_recv = None
        self.recv_tensors = []

        self.get_neighbours()
        self.get_neighbours_rank()
        self.set_tags()
        self.get_index_locations()

        self.identity = torch.nn.Identity()

    def set_tags(self):
        self.send_tag = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.recv_tag = [900, 800, 700, 600, 500, 400, 300, 200, 100]

    def get_neighbours_rank(self):
        self.rank_neighbours = []

        if self.slice_method == "square":
            # 0 1 2
            # 2 3 4
            # 5 6 7
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            # top_left will be (total_cols + 1) away from current rank
            top_left = -(total_cols + 1)
            top = -total_cols
            top_right = -(total_cols - 1)
            left = -1
            right = +1
            bottom_left = total_cols - 1
            bottom = total_cols
            bottom_right = total_cols + 1
            rank_offset = [
                top_left,
                top,
                top_right,
                left,
                0,
                right,
                bottom_left,
                bottom,
                bottom_right,
            ]

        elif self.slice_method == "vertical":
            rank_offset = [0, 0, 0, -1, 0, +1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            rank_offset = [0, -1, 0, 0, 0, 0, 0, +1, 0]

        for i in range(9):
            if self.neighbours[i] == 1:
                self.rank_neighbours.append(self.local_rank + rank_offset[i])
            else:
                self.rank_neighbours.append(-1)

    def get_neighbours(self):
        if self.local_rank < self.num_spatial_parts:
            self.ENABLE_SPATIAL = True
        else:
            self.ENABLE_SPATIAL = False
            self.neighbours = None
            return

        self.spatial_rank = self.local_rank % self.num_spatial_parts

        if self.slice_method == "square":
            self.neighbours = []
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            # current rank position in matrix of total_rows * total_cols
            row = self.local_rank / total_rows
            col = self.local_rank % total_cols
            dir = [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]

            for d in dir:
                neighbour_row = row + d[0]
                neighbour_col = col + d[1]
                if neighbour_row == row and neighbour_col == col:
                    self.neighbours.append(0)
                elif (
                    neighbour_row < 0
                    or neighbour_row >= total_rows
                    or neighbour_col < 0
                    or neighbour_col >= total_cols
                ):
                    self.neighbours.append(0)
                else:
                    self.neighbours.append(1)

        elif self.slice_method == "vertical":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 0, 0, 1, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 0, 0, 1, 0, 1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 1, 0]

    def get_index_locations(self):
        locations_recv = []
        locations_recv.append([[None, self.halo_len], [None, self.halo_len]])  # 1
        locations_recv.append(
            [[None, self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 2
        locations_recv.append([[None, self.halo_len], [-self.halo_len, None]])  # 3
        locations_recv.append(
            [[self.halo_len, -self.halo_len], [None, self.halo_len]]
        )  # 4
        locations_recv.append([[None, None], [None, None]])  # 5
        locations_recv.append(
            [[self.halo_len, -self.halo_len], [-self.halo_len, None]]
        )  # 6
        locations_recv.append([[-self.halo_len, None], [None, self.halo_len]])  # 7
        locations_recv.append(
            [[-self.halo_len, None], [self.halo_len, -self.halo_len]]
        )  # 8
        locations_recv.append([[-self.halo_len, None], [-self.halo_len, None]])  # 9

        self.locations_recv = locations_recv

        locations_send = []
        locations_send.append(
            [[self.halo_len, 2 * self.halo_len], [self.halo_len, 2 * self.halo_len]]
        )  # 1
        locations_send.append(
            [[self.halo_len, 2 * self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 2
        locations_send.append(
            [
                [self.halo_len, 2 * self.halo_len],
                [-2 * self.halo_len, -1 * self.halo_len],
            ]
        )  # 3
        locations_send.append(
            [[self.halo_len, -self.halo_len], [self.halo_len, 2 * self.halo_len]]
        )  # 4
        locations_send.append([[None, None], [None, None]])  # 5
        locations_send.append(
            [[self.halo_len, -self.halo_len], [-2 * self.halo_len, -1 * self.halo_len]]
        )  # 6
        locations_send.append(
            [
                [-2 * self.halo_len, -1 * self.halo_len],
                [self.halo_len, 2 * self.halo_len],
            ]
        )  # 7
        locations_send.append(
            [[-2 * self.halo_len, -1 * self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 8
        locations_send.append(
            [
                [-2 * self.halo_len, -1 * self.halo_len],
                [-2 * self.halo_len, -1 * self.halo_len],
            ]
        )  # 9
        self.locations_send = locations_send

    def get_shapes_recv(self, shapes):
        shapes_recv = []

        shapes_recv.append([self.halo_len, self.halo_len])  # 1
        shapes_recv.append([self.halo_len, shapes[3] - 2 * self.halo_len])  # 2
        shapes_recv.append([self.halo_len, self.halo_len])  # 3

        shapes_recv.append([shapes[2] - 2 * self.halo_len, self.halo_len])  # 4
        shapes_recv.append([None, None])  # 5
        shapes_recv.append([shapes[2] - 2 * self.halo_len, self.halo_len])  # 6

        shapes_recv.append([self.halo_len, self.halo_len])  # 7
        shapes_recv.append([self.halo_len, shapes[3] - 2 * self.halo_len])  # 8
        shapes_recv.append([self.halo_len, self.halo_len])  # 9

        return shapes_recv




    def start_halo_exchange(self, halo_input):
        req = []
        sync_all()
        for i in range(9):
            if self.neighbours[i] == 1:
                temp = (
                    halo_input[
                        :,
                        :,
                        self.locations_send[i][0][0] : self.locations_send[i][0][1],
                        self.locations_send[i][1][0] : self.locations_send[i][1][1],
                    ]
                    .clone()
                    .detach()
                )

                # torch.cuda.synchronize()

                # temp_req = dist.isend(
                #     temp, self.rank_neighbours[i], tag=self.send_tag[i]
                # )
                dist.isend(
                    temp, self.rank_neighbours[i], tag=self.send_tag[i]
                )
                print("*********done isend ")
                # req.append(temp_req)
                self.send_tag[i] += 1
        sync_all()

        self.recv_tensors = []

        shapes = halo_input.shape
        self.halo_input_shape = shapes
        if self.shapes_recv == None:
            self.shapes_recv = self.get_shapes_recv(shapes)
        sync_all()
        for i in range(9):
            if self.neighbours[i] == 1:
                temp_tensor = torch.zeros(
                    shapes[0],
                    shapes[1],
                    self.shapes_recv[i][0],
                    self.shapes_recv[i][1],
                    dtype=torch.float,
                    device="cuda",
                )

                """
				Synchronization is necessary at this point as all GPU operations
    			in PyTorch are asynchronous MPI copy operation is not under
       			PyTorch therefore it can start before pytorch finishes
          		initilization of tensor with zeros It will lead to data
            	corruption Spent 1 week on this issue (data validation)
				KEEP THIS IN MIND
				"""
                # torch.cuda.synchronize()
                # temp_req = dist.irecv(
                #     tensor=temp_tensor,
                #     src=self.rank_neighbours[i],
                #     tag=self.recv_tag[i],
                # )
                dist.irecv(
                    tensor=temp_tensor,
                    src=self.rank_neighbours[i],
                    tag=self.recv_tag[i],
                )
                # req.append(temp_req)
                self.recv_tag[i] += 1

                self.recv_tensors.append(temp_tensor)
            else:
                self.recv_tensors.append([])
        sync_all()

        return req



    def perform_halo_exchange(self, halo_input):
        shapes = halo_input.shape
        self.halo_input_shape = shapes
        if self.shapes_recv == None:
            self.shapes_recv = self.get_shapes_recv(shapes)

        self.recv_tensors = []
        for i in range(9):
            self.recv_tensors.append([])


        first_req_batch = []
        # sync_all()
        for i in range(9):
            if self.neighbours[i] == 1:
                if self.rank_neighbours[i] > self.local_rank:
                    temp = (
                        halo_input[
                            :,
                            :,
                            self.locations_send[i][0][0] : self.locations_send[i][0][1],
                            self.locations_send[i][1][0] : self.locations_send[i][1][1],
                        ]
                        .clone()
                        .detach()
                    )
                    req = dist.isend(
                        temp, self.rank_neighbours[i], tag=self.send_tag[i]
                    )
                    first_req_batch.append(req)
                    self.send_tag[i] += 1
                else:
                    temp_tensor = torch.zeros(
                        shapes[0],
                        shapes[1],
                        self.shapes_recv[i][0],
                        self.shapes_recv[i][1],
                        dtype=torch.float,
                        device="cuda",
                    )
                    req = dist.irecv(
                        tensor=temp_tensor,
                        src=self.rank_neighbours[i],
                        tag=self.recv_tag[i],
                    )
                    first_req_batch.append(req)
                    self.recv_tag[i] += 1
                    self.recv_tensors[i] = temp_tensor
        for req in first_req_batch:
            req.wait()

        # sync_all()

        second_req_batch = []
        for i in range(9):
            if self.neighbours[i] == 1:
                if self.rank_neighbours[i] < self.local_rank:
                    temp = (
                        halo_input[
                            :,
                            :,
                            self.locations_send[i][0][0] : self.locations_send[i][0][1],
                            self.locations_send[i][1][0] : self.locations_send[i][1][1],
                        ]
                        .clone()
                        .detach()
                    )
                    req = dist.isend(
                        temp, self.rank_neighbours[i], tag=self.send_tag[i]
                    )
                    second_req_batch.append(req)
                    self.send_tag[i] += 1
                else:
                    temp_tensor = torch.zeros(
                        shapes[0],
                        shapes[1],
                        self.shapes_recv[i][0],
                        self.shapes_recv[i][1],
                        dtype=torch.float,
                        device="cuda",
                    )
                    req = dist.irecv(
                        tensor=temp_tensor,
                        src=self.rank_neighbours[i],
                        tag=self.recv_tag[i],
                    )
                    second_req_batch.append(req)
                    self.recv_tag[i] += 1
                    self.recv_tensors[i] = temp_tensor

        for req in second_req_batch:
            req.wait()

    def end_halo_exchange(self, reqs):
        for req in reqs:
            req.wait()

    def copy_halo_exchange_values(self, halo_input):
        for i in range(9):
            if self.neighbours[i] == 1:
                halo_input[
                    :,
                    :,
                    self.locations_recv[i][0][0] : self.locations_recv[i][0][1],
                    self.locations_recv[i][1][0] : self.locations_recv[i][1][1],
                ] = self.recv_tensors[i]

    def run(self, tensor):
        s = torch.cuda.Stream(priority=0)

        rec = torch.cuda.Event(enable_timing=True, blocking=True)

        # reqs = self.start_halo_exchange(tensor)

        # self.end_halo_exchange(reqs)
        self.perform_halo_exchange(tensor)

        self.copy_halo_exchange_values(tensor)

        return tensor


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def initialize_cuda():
    my_local_rank = env2int(
        [
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "LOCAL_RANK",
        ],
        0,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(my_local_rank % num_spatial_parts)

    torch.cuda.init()


def init_comm(backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    return size, rank


def create_input_vertical(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0])
    image_width_local = int(image_size[1] / num_spatial_parts)  # use num_spatial_parts

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    start_left = rank * image_width_local
    end_right = (rank + 1) * image_width_local + 2 * halo_len

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        expected_output = expected_output[:, :, :, start_left:]
    else:
        expected_output = expected_output[:, :, :, start_left:end_right]

    start_left_i = rank * image_width_local
    end_right_i = (rank + 1) * image_width_local

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        input_local = np_x[:, :, :, start_left_i:]
    else:
        input_local = np_x[:, :, :, start_left_i:end_right_i]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input_horizontal(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0] / num_spatial_parts)  # use num_spatial_parts
    image_width_local = int(image_size[1])

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    start_top = rank * image_height_local
    end_bottom = (rank + 1) * image_height_local + 2 * halo_len

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        expected_output = expected_output[:, :, start_top:, :]
    else:
        expected_output = expected_output[:, :, start_top:end_bottom, :]

    start_top_i = rank * image_height_local
    end_bottom_i = (rank + 1) * image_height_local

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        input_local = np_x[:, :, start_top_i:, :]
    else:
        input_local = np_x[:, :, start_top_i:end_bottom_i, :]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input_square(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0] / math.sqrt(num_spatial_parts))
    image_width_local = int(image_size[1] / math.sqrt(num_spatial_parts))

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    total_rows = int(math.sqrt(num_spatial_parts))
    total_cols = int(math.sqrt(num_spatial_parts))
    # position of rank in matrix math.sqrt(num_spatial_parts) * math.sqrt(num_spatial_parts)
    row = int(rank / total_rows)
    col = int(rank % total_cols)

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    e_left_idx = col * image_width_local
    e_right_idx = (col + 1) * image_width_local + 2 * halo_len

    e_top_idx = row * image_height_local
    e_bottom_idx = (row + 1) * image_height_local + 2 * halo_len

    expected_output = expected_output[
        :, :, e_top_idx:e_bottom_idx, e_left_idx:e_right_idx
    ]

    left_idx = col * image_width_local
    right_idx = (col + 1) * image_width_local

    top_idx = row * image_height_local
    bottom_idx = (row + 1) * image_height_local

    input_local = np_x[:, :, top_idx:bottom_idx, left_idx:right_idx]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input(halo_len, image_size, num_spatial_parts, rank, slice_method):
    if slice_method == "vertical":
        return create_input_vertical(halo_len, image_size, num_spatial_parts, rank)
    elif slice_method == "horizontal":
        return create_input_horizontal(halo_len, image_size, num_spatial_parts, rank)
    elif slice_method == "square":
        return create_input_square(halo_len, image_size, num_spatial_parts, rank)


def test_output(output, expected_output, rank):
    np_out = output.to("cpu").numpy()

    if np.equal(np_out.astype("int"), expected_output.astype("int")).all():
        print(f"Validation passed for rank: {rank}")
    else:
        uneq = np.not_equal(np_out.astype("int"), expected_output.astype("int"))
        print(
            f"Rank : {rank} => Received : {np_out[uneq]} Expected : {expected_output[uneq]}"
        )
        print(f"Validation failed for rank: {rank}")


def run_benchmark(rank, size, hostname):
    input_tensor_local, expected_output = create_input(
        halo_len=halo_len,
        image_size=image_size,
        num_spatial_parts=num_spatial_parts,
        rank=rank,
        slice_method=slice_method,
    )

    b_pt2pt = halo_bench_pt2pt(
        local_rank=rank,
        comm_size=size,
        num_spatial_parts=num_spatial_parts,
        slice_method=slice_method,
        halo_len=halo_len,
    )

    for i in range(args.warmup):
        y = b_pt2pt.run(input_tensor_local)

    start_event = torch.cuda.Event(enable_timing=True, blocking=True)
    end_event = torch.cuda.Event(enable_timing=True, blocking=True)

    start_event.record()
    for i in range(iterations):
        y = b_pt2pt.run(input_tensor_local)

    end_event.record()
    torch.cuda.synchronize()

    t = start_event.elapsed_time(end_event)

    print(f"Rank: {rank} Time taken (ms): {(t / iterations)}")

    test_output(y, expected_output, rank)

def timed_pt2pt(input, start_event, end_event, args, local_rank, size):
    input_tensor_local, expected_output = create_input(
        halo_len=halo_len,
        image_size=image_size,
        num_spatial_parts=num_spatial_parts,
        rank=local_rank,
        slice_method=slice_method,
    )
    b_pt2pt = halo_bench_pt2pt(
        local_rank=local_rank,
        comm_size=size,
        num_spatial_parts=num_spatial_parts,
        slice_method=slice_method,
        halo_len=halo_len,
    )
    halo_input = input_tensor_local
    print("START HALO-EXCHANGE")
    sync_all()
    shape = (32, 3, 1024, 1024)
    shapes = halo_input.shape
    if b_pt2pt.shapes_recv == None:
        b_pt2pt.shapes_recv = b_pt2pt.get_shapes_recv(shapes)

    for i in range(args.warmups):
        sync_all()
        req_first = []
        for i in range(9):
            if b_pt2pt.neighbours[i] == 1:
                if  b_pt2pt.rank_neighbours[i] > local_rank:
                    temp = (
                        halo_input[
                            :,
                            :,
                            b_pt2pt.locations_send[i][0][0] : b_pt2pt.locations_send[i][0][1],
                            b_pt2pt.locations_send[i][1][0] : b_pt2pt.locations_send[i][1][1],
                        ]
                        .clone()
                        .detach()
                    )

                    print(f"local_rank : {local_rank} sending to Neighbour : {b_pt2pt.rank_neighbours[i]}")
                    req = dist.isend(
                        temp, b_pt2pt.rank_neighbours[i]
                    )
                    b_pt2pt.send_tag[i] += 1
                    req_first.append(req)
                else:
                    print(f"local_rank : {local_rank} receiving from Neighbour : {b_pt2pt.rank_neighbours[i]}")
                    temp_tensor = torch.zeros(
                        shapes[0],
                        shapes[1],
                        b_pt2pt.shapes_recv[i][0],
                        b_pt2pt.shapes_recv[i][1],
                        dtype=torch.float,
                        device="cuda",
                    )

                    req = dist.irecv(
                        tensor=temp_tensor,
                        src=b_pt2pt.rank_neighbours[i],

                    )
                    req_first.append(req)
                    b_pt2pt.recv_tag[i] += 1
        for req in req_first:
            req.wait()
        sync_all()

        for i in range(9):
            if b_pt2pt.neighbours[i] == 1:
                if  b_pt2pt.rank_neighbours[i] < local_rank:
                    temp = (
                        halo_input[
                            :,
                            :,
                            b_pt2pt.locations_send[i][0][0] : b_pt2pt.locations_send[i][0][1],
                            b_pt2pt.locations_send[i][1][0] : b_pt2pt.locations_send[i][1][1],
                        ]
                        .clone()
                        .detach()
                    )

                    print(f"local_rank : {local_rank} sending to Neighbour : {b_pt2pt.rank_neighbours[i]}")
                    req = dist.isend(
                        temp, b_pt2pt.rank_neighbours[i]
                    )
                    req_first.append(req)
                    b_pt2pt.send_tag[i] += 1
                else:
                    print(f"local_rank : {local_rank} receiving from  Neighbour : {b_pt2pt.rank_neighbours[i]}")
                    temp_tensor = torch.zeros(
                        shapes[0],
                        shapes[1],
                        b_pt2pt.shapes_recv[i][0],
                        b_pt2pt.shapes_recv[i][1],
                        dtype=torch.float,
                        device="cuda",
                    )

                    req = dist.irecv(
                        tensor=temp_tensor,
                        src=b_pt2pt.rank_neighbours[i],

                    )
                    req_first.append(req)
                    b_pt2pt.recv_tag[i] += 1
        for req in req_first:
            req.wait()
        sync_all()






    # sync_all()

    print("END HALO-EXCHANGE")
    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        if dist.get_rank() == 0:
            if args.async_op:
                dist.isend(input, 1)
            else:
                dist.send(input, 1)
        if dist.get_rank() == 1:
            if args.async_op:
                dist.irecv(input, src=0)
            else:
                dist.recv(input, src=0)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        if dist.get_rank() == 0:
            if args.async_op:
                dist.isend(input, 1)
            else:
                dist.send(input, 1)
        if dist.get_rank() == 1:
            if args.async_op:
                dist.irecv(input, src=0)
            else:
                dist.recv(input, src=0)

    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('pt2pt', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_pt2pt(local_rank, args, size):
    # Prepare benchmark header
    print_header(args, 'pt2pt')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.scan:
        # Create list of message sizes
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M,
                                 dtype=getattr(torch, args.dtype)).to("cuda")
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if dist.get_rank() == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_pt2pt(input, start_event, end_event, args, local_rank, size)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so double mem_factor
        elements_per_gpu = max_numel(comm_op='pt2pt',
                                     dtype=getattr(torch, args.dtype),
                                     mem_factor=args.mem_factor * 2,
                                     local_rank=local_rank,
                                     args=args)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch,
                                                             args.dtype)).to("cuda")
            input = ((mat.mul_(float(global_rank))).view(-1))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
        sync_all()
        timed_pt2pt(input, start_event, end_event, args, local_rank, size)

def init_processes(hostname, fn, backend="nccl"):
    """Initialize the distributed environment."""
    # initialize_cuda()
    if backend == "nccl":
        mpi_discovery()
    size, rank = init_comm()
    validate_config(num_spatial_parts, size)
    fn(rank, size, hostname)
    # run_pt2pt(local_rank=rank, args=args, size=size)


if __name__ == "__main__":
    init_processes("a", run_benchmark, backend="nccl")
