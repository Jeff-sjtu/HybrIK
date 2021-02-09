import os
import re
import torch
import torch.distributed as dist


def init_dist(opt):
    """Initialize distributed computing environment."""
    opt.ngpus_per_node = torch.cuda.device_count()

    if opt.launcher == 'pytorch':
        _init_dist_pytorch(opt)
    elif opt.launcher == 'mpi':
        _init_dist_mpi(opt)
    elif opt.launcher == 'slurm':
        _init_dist_slurm(opt)
    else:
        raise ValueError('Invalid launcher type: {}'.format(opt.launcher))


def _init_dist_pytorch(opt, **kwargs):
    """Set up environment."""
    # TODO: use local_rank instead of rank % num_gpus
    opt.rank = opt.rank * opt.ngpus_per_node + opt.gpu
    opt.world_size = opt.world_size
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    torch.cuda.set_device(opt.gpu)
    print(f"{opt.dist_url}, ws:{opt.world_size}, rank:{opt.rank}")

    if opt.rank % opt.ngpus_per_node == 0:
        opt.log = True
    else:
        opt.log = False


def _init_dist_slurm(opt, port=23333, **kwargs):
    """Set up slurm environment."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = str(os.environ['SLURM_NODELIST'])
    num_gpus = torch.cuda.device_count()

    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    print(f"{init_method}, rank: {rank}, local rank: {local_rank}")

    dist.init_process_group(backend=opt.dist_backend,
                            init_method=init_method,
                            world_size=world_size,
                            rank=rank)

    torch.cuda.set_device(local_rank)
    opt.rank = rank
    opt.world_size = world_size
    opt.ngpus_per_node = num_gpus
    opt.gpu = local_rank

    if opt.rank == 0:
        opt.log = True
    else:
        opt.log = False


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError
