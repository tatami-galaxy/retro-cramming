import os
import numpy as np

from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager

def is_true_env_flag(env_flag):
    return os.getenv(env_flag, 'false').lower() in ('true', '1', 't')

def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

@contextmanager
def memmap(*args, **kwargs):
    # flush the memmap instance to write the changes to the file
    # currently there is no API to close the underlying mmap
    # it is tricky to ensure the resource is actually closed,
    # since it may be shared between different memmap instances
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer
