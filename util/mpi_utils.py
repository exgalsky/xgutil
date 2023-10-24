import numpy as np 
import healpy as hp

import logging
log = logging.getLogger(__name__)

run_with_mpi = True
if run_with_mpi:
    try: 
        from mpi4py import MPI
    except:
        log.usky_warn("WARNING: mpi4py not found, fallback to serial implementation.")  # todo: Replace print warning messages with proper logging implementation
        run_with_mpi = False

# MPI communicator initialization

class mpi_handler():

    def __init__(self, force_no_mpi=False): 
        self.__run_with_mpi = run_with_mpi 
        self.root = 0
        if force_no_mpi: self.__run_with_mpi = False
        if self.__run_with_mpi:
            self.comm = MPI.COMM_WORLD
            self.id = self.comm.Get_rank()            #number of the process running the code
            self.numProc = self.comm.Get_size()       #total number of processes running
            self.rank_tag = f"MPI rank {self.id}"
        else:
            self.comm = None
            self.id = 0
            self.numProc = 1 
            self.rank_tag = f"serial task"
    def divide4mpi(self, data_shape, decom_type='slab', divide_axis=0):

        if decom_type.lower() == 'slab':
            
            # Run on root process
            if self.id == 0:
            # Slice the cube along x which is the slowest varying axis (C indexing)
            # Note: all binary files for numpy arrays follow C indexing

                slab_min = data_shape[divide_axis] // self.numProc      # Q
                iter_remains = np.mod(data_shape[divide_axis], self.numProc)    # R 

                #  SZ = R x (Q + 1) + (P-R) x Q 
                self.slab_per_Proc = np.zeros((self.numProc,), dtype=np.int16)  # P = len(zslab_per_Proc)

                self.slab_per_Proc[0:int(iter_remains)] = slab_min + 1     # R procs together get (Q+1)xR z slabs
                self.slab_per_Proc[int(iter_remains):]  = slab_min         # P-R procs together get Qx(P-R) z slabs

                # print(np.sum(zslab_per_Proc))

                del slab_min
            else: 
                self.slab_per_Proc = None 

            # Send copy of slab_per_Proc to each process.
            if self.__run_with_mpi:    self.slab_per_Proc = self.comm.bcast(self.slab_per_Proc, root=0)

            # Find the start and stop of the x-axis slab for each process
            self.slab_start_in_Proc = np.sum(self.slab_per_Proc[0:self.id])    
            self.slab_stop_in_Proc = self.slab_start_in_Proc + self.slab_per_Proc[self.id]

            log.usky_info(f"  {self.rank_tag}: slab_per_Proc      = {self.slab_per_Proc}")
            log.usky_info(f"  {self.rank_tag}: slab_start_in_Proc = {self.slab_start_in_Proc}")
            log.usky_info(f"  {self.rank_tag}: slab_stop_in_Proc  = {self.slab_stop_in_Proc}")

            return self.slab_start_in_Proc, self.slab_stop_in_Proc
        else:
            pass

    def data_offset(self, data_shape, bytes_per_cell, divide_axis=0, decom_type='slab'):
        undiv_axes_prod = 1.
        for i in range(len(data_shape)):
            if i != divide_axis: undiv_axes_prod *= data_shape[i] 

        if decom_type.lower() == 'slab':
            return np.int64(np.sum(self.slab_per_Proc[0:self.id]) * undiv_axes_prod * bytes_per_cell)
        
    def reduce2map(self, map_in_proc):
        if self.id == self.root:
            reduced_map = np.zeros(map_in_proc.shape)
        else:
            reduced_map = None 

        if self.__run_with_mpi:
            self.comm.Reduce([map_in_proc, MPI.DOUBLE], reduced_map, op=MPI.SUM, root=self.root)
        else:
            reduced_map = map_in_proc

        del map_in_proc 

        return reduced_map
            
    def writemap2file(self, map2write, filename, overwrite=True):

        if (self.id == self.root):
            hp.write_map(filename, map2write, dtype=map2write.dtype, overwrite=overwrite)


