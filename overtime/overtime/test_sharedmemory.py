import unittest
from overtime.sharedmemory import shmmap, ndshmem


class Test_ndshmem(unittest.TestCase):
    def test_empty(self):
        import numpy as np
        
        shape = 0
        self.assertRaises(ValueError, ndshmem.empty, shape)
            
        shape = 1
        self.assertEqual(ndshmem.empty(shape).shape, (1,))
        
        shape = (2, 2)
        self.assertEqual(ndshmem.empty(shape).shape, (2, 2))
        
        shape = (1, 2, 3)
        self.assertEqual(ndshmem.empty(shape).shape, (1, 2, 3))

        self.assertEqual(ndshmem.empty((1, 2, 3), np.byte).dtype, np.byte)
        self.assertEqual(ndshmem.empty((1, 2, 3), int).dtype, np.dtype(int))
        self.assertEqual(ndshmem.empty((1, 2, 3), np.float32).dtype, np.float32)
        self.assertEqual(ndshmem.empty((1, 2, 3), float).dtype, np.dtype(float))
        
        a = ndshmem.empty((3, 6, 9), order='C')
        a[...] = np.arange(a.size).reshape(a.shape)
        b = ndshmem.empty((3, 6, 9), order='F')
        b[...] = np.arange(a.size).reshape(a.shape)
        
        self.assertEqual(a.shape, b.shape)
        self.assertNotEqual(a.strides, b.strides)
        self.assert_(np.allclose(a, b))
        
    def test_array_finalize(self):
        import numpy as np
        
        a = ndshmem.empty((3, 6, 9)) 
        self.assertEqual(a.offset, 0)
        
        b = a[1:-1, 1:-1, 1:-1]
        self.assertNotEqual(a.offset, b.offset)        
        self.assertEqual(a._shm.semname, b._shm.semname)
        
        c = np.array(a).view(ndshmem)
        self.assertEqual(c._shm, None)
        
        d = a.view(np.ndarray)
        self.assertFalse(hasattr(d, '_shm'))
        
    def test_pickle(self):
        import numpy as np
        import pickle
        
        a = ndshmem.empty((3, 6, 9), int, 'C')
        b = np.array(a).view(ndshmem)
        self.assertRaises(pickle.PicklingError, pickle.dumps, b)
        
        a = a[1:-1, 1:-1, 1:-1]
        c = pickle.loads(pickle.dumps(a))
        self.assertEqual(a._shm.semname, c._shm.semname)
        
        c[...] = np.random.random_integers(0, 2 ** 31 - 1, c.shape)
        self.assert_(np.allclose(a, c))
                

class Test_shmmap(unittest.TestCase):
    def test_fromlength(self):
        length = 0
        self.assertRaises(ValueError, shmmap.fromlength, length)
            
        length = 1
        shm = shmmap.fromlength(length)
        self.assertEqual(shm.size(), length)
        
    def test_pickle(self):
        import pickle
        import os

        length = 1024
        shm1 = shmmap.fromlength(length)
        shm2 = shm1
        shm2 = pickle.dumps(shm2)
        
        shm1[:] = os.urandom(length)
        shm2 = pickle.loads(shm2)
        self.assertEqual(shm1.semname, shm2.semname)
        self.assertEqual(shm1[:], shm2[:])        
        
    def test_del(self):
        import pickle
        import posix_ipc
        import contextlib
        
        length = 1
        shm1 = shmmap.fromlength(length)
        semname = shm1.semname
        
        shm2 = pickle.dumps(shm1)
        del shm1
        with self.assertRaises(posix_ipc.ExistentialError):
            with contextlib.closing(posix_ipc.Semaphore(name=semname, flags=posix_ipc.O_CREX)) as sem:
                posix_ipc.unlink_semaphore(sem.name)
                
        
        shm2 = pickle.loads(shm2)
        del shm2
        with self.assertRaises(posix_ipc.ExistentialError):
            with contextlib.closing(posix_ipc.Semaphore(name=semname)) as sem:
                posix_ipc.unlink_semaphore(sem.name)
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test_shmmap.test_fromlength']
    unittest.main()
