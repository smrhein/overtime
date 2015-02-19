import mmap
import posix_ipc
import sys
import pickle
import numpy as np


class shmmap(mmap.mmap):
    @classmethod
    def fromlength(cls, length):
        return cls(None, None, length)
    
    def __new__(cls, semname, shmname, length, access=mmap.ACCESS_WRITE, offset=0):
        sem = posix_ipc.Semaphore(semname, 0 if semname else posix_ipc.O_CREX)
        try:
            shm = posix_ipc.SharedMemory(shmname, 0 if shmname else posix_ipc.O_CREX, size=length)
            try:
                self = super(shmmap, cls).__new__(cls, shm.fd, length, access=access, offset=offset)
                self.semname = sem.name
                self.shmname = shm.name
                self.length = length
                self.access = access
                self.offset = offset
                return self
            finally:
                shm.close_fd()
        except:
            exc = sys.exc_info()
            try:
                sem.acquire(0)                
            except posix_ipc.BusyError:
                posix_ipc.unlink_semaphore(sem.name)                
                posix_ipc.unlink_shared_memory(shm.name)
            finally:
                raise exc[0], exc[1], exc[2]
        finally:
            sem.close()
            
    def __reduce__(self):
        return self.__reduce_ex__(0)
    def __reduce_ex__(self, protocol):
        if self.access == mmap.ACCESS_COPY:
            raise pickle.PicklingError('copy on write memory cannot be pickled')        
        sem = posix_ipc.Semaphore(self.semname, 0)
        try:
            copy = self.__class__, (self.semname, self.shmname, self.length, self.access, self.offset)
            sem.release()            
            return copy
        finally:
            sem.close()
            
    def __del__(self):
        self.close()
        sem = posix_ipc.Semaphore(self.semname, 0)
        try:
            sem.acquire(0)            
        except posix_ipc.BusyError:
            posix_ipc.unlink_semaphore(self.semname)            
            posix_ipc.unlink_shared_memory(self.shmname)
        finally:
            sem.close()


class ndshmem(np.ndarray):
    @classmethod
    def empty(cls, shape, dtype=float, order='C'):
        return cls(shape=shape, dtype=dtype, order=order)
        
    def __new__(cls, shape, dtype=float, shm=None, offset=0, strides=None, order=None):
        if shm is None:
            shm = shmmap.fromlength(np.prod(shape) * np.dtype(dtype).itemsize)
        self = super(ndshmem, cls).__new__(cls, shape, dtype, shm, offset, strides, order)
        self._shm = shm
        self.offset = offset
        return self
    
    def __array_finalize__(self, obj):
        if np.may_share_memory(self, obj):
            try:
                self._shm = obj._shm
                self.offset = obj.offset
                if self._shm:
                    self.offset += np.byte_bounds(self)[0] - np.byte_bounds(obj)[0]
                return
            except AttributeError:
                pass
        self._shm = None
        self.offset = None
        
    def __reduce__(self):
        return self.__reduce_ex__(0)
    def __reduce_ex__(self, protocol):
        if not self._shm:
            raise pickle.PicklingError('array is not backed by shared memory')
        if self.flags['C_CONTIGUOUS']:
            order = 'C'
        elif self.flags['F_CONTIGUOUS']:
            order = 'F'
        else:
            order = None
        return self.__class__, (self.shape, self.dtype, self._shm, self.offset, self.strides, order)