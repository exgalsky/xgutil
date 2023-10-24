import numpy as np
import jax 
import os

import logging
log = logging.getLogger(__name__)

# Copied from TOAST
def jax_local_device():
    """Returns the device currenlty used by JAX."""
    # gets local device if it has been designated
    local_device = jax.config.jax_default_device
    # otherwise gets the first device available
    if local_device is None:
        devices = jax.local_devices()
        if len(devices) > 0:
            local_device = devices[0]
    return local_device

class jax_handler:

    def __init__(self, force_no_gpu=False,mpi_backend=None,max_GPU_mem_GB=40.0,
                 preallocate=False,allocator_platform=False):

        if preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if allocator_platform:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        self.GPU_available = False
        self.gpus = []
        self.mpi_backend = mpi_backend
        self.max_GPU_mem = max_GPU_mem_GB * 1024.**3 # GB --> bytes

        if force_no_gpu: 
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            
        else:
            try:
                import GPUtil
                self.GPU_available = True
                self.gpus = GPUtil.getGPUs()
                
            except:
                
                log.usky_warn(f"GPUtil not found. Assuming no GPUs are presentent and falling back to CPU. \n If GPUs are present then ensure GPUtil is installed to intialize JAX on the GPU.")
                
        self.ndevices = len(self.gpus)

        if self.ndevices > 1:
            log.usky_warn(f"Multiple GPU devices per processes is not supported at the moment. Using GPU device 0 only. \n To change this, divide the node to as many processes per node as there are GPU devices.")
            
        elif self.ndevices == 1:
            jax.distributed.initialize(local_device_ids=self.gpus[0].id)

        jax.config.update("jax_enable_x64", False)
        log.usky_info(f"JAX backend device set to: { jax_local_device() }")
            
        self.task_tag = "serial task"
        if self.mpi_backend is not None:
            self.task_tag = self.mpi_backend.rank_tag

    def jax_tasks(self, block_shape, peak_per_cell_memory, jax_overhead_factor, divide_axis=0):

        total_memory_required = block_shape[0] * block_shape[1] * block_shape[2] * peak_per_cell_memory * jax_overhead_factor
        log.usky_info(f"  {self.task_tag}: total_memory_required = {total_memory_required}")

        if self.GPU_available:
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            GPUmem = gpus[0].memoryTotal * 1024**2 # MB --> bytes
            GPUmem = min(GPUmem,self.max_GPU_mem)
            self.n_jaxcalls = int(np.ceil(total_memory_required / GPUmem))
            log.usky_info(f"  {self.task_tag}: GPUmem = {GPUmem}")
        else:
            import psutil
            mem = psutil.virtual_memory().total

            self.n_jaxcalls = int(np.ceil(total_memory_required / mem))

        log.usky_info(f"  {self.task_tag}: n_jaxcalls = {self.n_jaxcalls}")

        task_min = block_shape[divide_axis] // self.n_jaxcalls      # Q
        iter_remains = np.mod(block_shape[divide_axis], self.n_jaxcalls)    # R 

        #  SZ = R x (Q + 1) + (P-R) x Q 
        self.slices_per_jaxcall = np.zeros((self.n_jaxcalls,), dtype=np.int16)  # P = len(zslab_per_Proc)

        self.slices_per_jaxcall[0:int(iter_remains)] = task_min + 1     # R procs together get (Q+1)xR z slabs
        self.slices_per_jaxcall[int(iter_remains):]  = task_min         # P-R procs together get Qx(P-R) z slabs

        log.usky_info(f"  {self.task_tag}: slices_per_jaxcall[0:{int(iter_remains)}] = {task_min + 1}")
        log.usky_info(f"  {self.task_tag}: slices_per_jaxcall[{int(iter_remains)}:] = {task_min}")

        del task_min
    
    def jax_data_offset(self, chunk_shape, bytes_per_cell, mpi_offset=0, divide_axis=0, decom_type='slab'):
        undiv_axes_prod = 1.
        for i in range(len(chunk_shape)):
            if i != divide_axis: undiv_axes_prod *= chunk_shape[i] 

        if decom_type.lower() == 'slab':
            self.offsets_per_call = []
            for i in range(self.n_jaxcalls):
                self.offsets_per_call.append(mpi_offset + np.sum(self.slices_per_jaxcall[0:i]) * undiv_axes_prod * bytes_per_cell)
            
            self.offsets_per_call = np.array(self.offsets_per_call).astype(np.int64)
