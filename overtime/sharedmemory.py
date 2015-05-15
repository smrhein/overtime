from __future__ import print_function
import contextlib
import mmap
import posix_ipc
import sys
import pickle
import uuid

import numpy as np

from overtime.contextual import exiting


def _debug(*args, **kwargs):
    pass
    # k = multiprocessing.current_process().ident, threading.current_thread().ident
    # with _lck:
    # print(k, *args, **kwargs)


class shmmap(mmap.mmap):
    @classmethod
    def _ipc_name(cls, prefix='shmmap', suffix=''):
        return ''.join((prefix, str(uuid.uuid4()), suffix))

    @classmethod
    def fromlength(cls, length):
        return cls(None, None, length)

    def __new__(cls, semname, shmname, length, access=mmap.ACCESS_WRITE, offset=0):
        if semname and shmname:
            semkwargs = {'name': semname}
            shmkwargs = {'name': shmname}
            semkwargs['flags'] = shmkwargs['flags'] = 0
        elif not (semname or shmname):
            semkwargs = {'name': cls._ipc_name()}
            shmkwargs = {'name': cls._ipc_name()}
            semkwargs['flags'] = shmkwargs['flags'] = posix_ipc.O_CREX
        else:
            raise ValueError('semname is None if and only if shmname is None')
        shmkwargs['size'] = length

        with contextlib.closing(posix_ipc.Semaphore(**semkwargs)) as sem, \
                exiting(posix_ipc.SharedMemory(**shmkwargs), exiter=posix_ipc.SharedMemory.close_fd) as shm:
            try:
                self = super(shmmap, cls).__new__(cls, shm.fd, length, access=access, offset=offset)
                self.shmname = shm.name
                self.semname = sem.name
            except:
                exc = sys.exc_info()
                try:
                    sem.acquire(0)
                except posix_ipc.BusyError:
                    posix_ipc.unlink_semaphore(sem.name)
                    posix_ipc.unlink_shared_memory(shm.name)
                finally:
                    raise exc[0], exc[1], exc[2]
            else:
                self.length = length
                self.access = access
                self.offset = offset
                return self

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
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
            _debug('del', sem.name, sem.value)
        finally:
            sem.close()


class ndshm(np.ndarray):
    @classmethod
    def fromndarray(cls, a, dtype=None, order=None, subok=False):
        b = cls.empty_like(a, dtype, order, subok)
        b[...] = a
        return b

    @classmethod
    def zeros(cls, shape, dtype=None, order='C'):
        a = cls.empty(shape, dtype, order)
        a[...] = 0
        return a

    @classmethod
    def empty_like(cls, a, dtype=None, order='K', subok=True):
        b = (cls if subok else ndshm).empty(a.shape, dtype, order)
        return b

    @classmethod
    def empty(cls, shape, dtype=float, order='C'):
        return cls(shape=shape, dtype=dtype, order=order)

    def __new__(cls, shape, dtype=float, shm=None, offset=0, strides=None, order=None):
        if shm is None:
            shm = shmmap.fromlength(np.prod(shape) * np.dtype(dtype).itemsize)
        self = super(ndshm, cls).__new__(cls, shape, dtype, shm, offset, strides, order)
        self._shm = shm
        self.offset = offset
        return self

    def __array_finalize__(self, obj):
        if np.may_share_memory(self, obj):
            try:
                self._shm = obj._shm
                self.offset = obj.offset
                if self._shm is not None:
                    self.offset += np.byte_bounds(self)[0] - np.byte_bounds(obj)[0]
                return
            except AttributeError:
                pass
        self._shm = None
        self.offset = None

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if self._shm is None:
            raise pickle.PicklingError('array is not backed by shared memory')
        if self.flags['C_CONTIGUOUS']:
            order = 'C'
        elif self.flags['F_CONTIGUOUS']:
            order = 'F'
        else:
            order = None
        return self.__class__, (self.shape, self.dtype, self._shm, self.offset, self.strides, order)



