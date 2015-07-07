from collections import Sized
from multiprocessing.pool import ThreadPool
import os
import sys
import functools
import pickle
import threading
import warnings
import gc
import itertools
import time
import multiprocessing as mp

import joblib
from joblib.logger import short_format_time
from joblib.my_exceptions import TransportableException
from joblib.parallel import LockedIterator, JOBLIB_SPAWNED_PROCESS, WorkerInterrupt
from joblib.pool import MemmapingPool


class Parallel(joblib.Parallel):
    def __call__(self, iterable):
        if self._jobs:
            raise ValueError('This Parallel instance is already running')
        n_jobs = self.n_jobs
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        if n_jobs < 0 and mp is not None:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)

        # The list of exceptions that we will capture
        self.exceptions = [TransportableException]
        self._lock = threading.Lock()

        # Whether or not to set an environment flag to track
        # multiple process spawning
        set_environ_flag = False
        if (n_jobs is None or mp is None or n_jobs == 1):
            n_jobs = 1
            self._pool = None
        elif self.backend == 'threading':
            self._pool = ThreadPool(n_jobs)
        elif self.backend == 'multiprocessing':
            if mp.current_process().daemon:
                # Daemonic processes cannot have children
                n_jobs = 1
                self._pool = None
                warnings.warn(
                    'Multiprocessing-backed parallel loops cannot be nested,'
                    ' setting n_jobs=1',
                    stacklevel=2)
            elif threading.current_thread().name != 'MainThread':
                # Prevent posix fork inside in non-main posix threads
                n_jobs = 1
                self._pool = None
                warnings.warn(
                    'Multiprocessing backed parallel loops cannot be nested'
                    ' below threads, setting n_jobs=1',
                    stacklevel=2)
            else:
                already_forked = int(os.environ.get('__JOBLIB_SPAWNED_PARALLEL__', 0))
                if already_forked:
                    raise ImportError('[joblib] Attempting to do parallel computing '
                                      'without protecting your import on a system that does '
                                      'not support forking. To use parallel-computing in a '
                                      'script, you must protect you main loop using "if '
                                      "__name__ == '__main__'"
                                      '". Please see the joblib documentation on Parallel '
                                      'for more information'
                                      )

                # Make sure to free as much memory as possible before forking
                gc.collect()

                # Set an environment variable to avoid infinite loops
                set_environ_flag = True
                poolargs = dict(
                    max_nbytes=self._max_nbytes,
                    mmap_mode=self._mmap_mode,
                    temp_folder=self._temp_folder,
                    verbose=max(0, self.verbose - 50),
                    context_id=0,  # the pool is used only for one call
                )
                if self._mp_context is not None:
                    # Use Python 3.4+ multiprocessing context isolation
                    poolargs['context'] = self._mp_context
                self._pool = MemmapingPool(n_jobs, **poolargs)
                # We are using multiprocessing, we also want to capture
                # KeyboardInterrupts
                self.exceptions.extend([KeyboardInterrupt, WorkerInterrupt])
        else:
            raise ValueError("Unsupported backend: %s" % self.backend)

        pre_dispatch = self.pre_dispatch
        if isinstance(iterable, Sized):
            # We are given a sized (an object with len). No need to be lazy.
            pre_dispatch = 'all'

        if pre_dispatch == 'all' or n_jobs == 1:
            self._original_iterable = None
            self._pre_dispatch_amount = 0
        else:
            # The dispatch mechanism relies on multiprocessing helper threads
            # to dispatch tasks from the original iterable concurrently upon
            # job completions. As Python generators are not thread-safe we
            # need to wrap it with a lock
            iterable = LockedIterator(iterable)
            self._original_iterable = iterable
            self._dispatch_amount = 0
            if hasattr(pre_dispatch, 'endswith'):
                pre_dispatch = eval(pre_dispatch)
            self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)

            # The main thread will consume the first pre_dispatch items and
            # the remaining items will later be lazily dispatched by async
            # callbacks upon task completions
            iterable = itertools.islice(iterable, pre_dispatch)

        self._start_time = time.time()
        self.n_dispatched = 0
        try:
            if set_environ_flag:
                # Set an environment variable to avoid infinite loops
                os.environ[JOBLIB_SPAWNED_PROCESS] = '1'
            self._iterating = True
            for function, args, kwargs in iterable:
                self.dispatch(function, args, kwargs)

            if pre_dispatch == "all" or n_jobs == 1:
                # The iterable was consumed all at once by the above for loop.
                # No need to wait for async callbacks to trigger to
                # consumption.
                self._iterating = False
            self.retrieve()
            # Make sure that we get a last message telling us we are done
            elapsed_time = time.time() - self._start_time
            self._print('Done %3i out of %3i | elapsed: %s finished',
                        (len(self._output),
                         len(self._output),
                         short_format_time(elapsed_time)
                         ))

        finally:
            if n_jobs > 1:
                self._pool.close()
                self._pool.join()  # terminate does NOT do a join()
                if self.backend == 'multiprocessing':
                    os.environ.pop(JOBLIB_SPAWNED_PROCESS, 0)
            self._jobs = list()
        output = self._output
        self._output = None
        return output


class CtxObject(object):
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self._obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._obj


def ctx_delayed(function, check_pickle=True):
    if check_pickle:
        pickle.dumps(function)

    def delayed_function(*args, **kwargs):
        return ctx_worker, (function, args, kwargs), {}

    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        " functools.wraps fails on some callable objects "
    return delayed_function


def ctx_worker(function, args, kwargs):
    if not isinstance(function, CtxObject):
        function = CtxObject(function)
    args = [CtxObject(v) if not isinstance(v, CtxObject) else v for v in args]
    kwargs = {k: (CtxObject(v) if not isinstance(v, CtxObject) else v) for (k, v) in kwargs.iteritems()}
    with Iterated(function, *args, **kwargs) as (args, kwargs):
        function, args = args[0], args[1:]
        return function(*args, **kwargs)


class Exiting(object):
    def __init__(self, exiter, exiter_self, *exiter_args, **exiter_kwargs):
        self.exiter = exiter
        self.exiter_self = exiter_self
        self.exiter_args = exiter_args
        self.exiter_kwargs = exiter_kwargs

    def __enter__(self):
        return self.exiter_self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exiter(self.exiter_self, *self.exiter_args, **self.exiter_kwargs)


class GenericContextManager(object):
    def __init__(self, enter_call, exit_call):
        self._enter_call = enter_call
        self._exit_call = exit_call

    def __enter__(self):
        value = self._enter_call()
        self._exit_call = self._exit_call.with_object(value)
        return value

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_call()


def get_context_guard(mgr):
    try:
        return mgr.__enter__, mgr.__exit__
    except AttributeError:
        raise TypeError('argument is not a context manager')


class Iterated(object):
    def __init__(self, *args, **kwargs):
        self._args = list(args[::-1])
        self._kwargs = kwargs
        self._exits = []

    def __enter__(self):
        exc = (None,) * 3
        as_args = []
        as_kwargs = []

        try:
            while self._args:
                mgr = self._args.pop()
                try:
                    enter, exit_ = get_context_guard(mgr)
                    value = enter()
                except:
                    exc = sys.exc_info()
                else:
                    as_args.append(value)
                    self._exits.append(exit_)

            while self._kwargs:
                k, mgr = self._kwargs.popitem()
                try:
                    enter, exit_ = get_context_guard(mgr)
                    key_value = (k, enter())
                except:
                    exc = sys.exc_info()
                else:
                    as_kwargs.append(key_value)
                    self._exits.append(exit_)
        except:
            exc = sys.exc_info()
        finally:
            self._exits.reverse()
            if exc != (None,) * 3:
                self.__exit__(None, None, None)
                raise exc[0], exc[1], exc[2]
            else:
                return as_args, dict(as_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        exc = (None,) * 3
        suppress = False
        try:
            while self._exits:
                exit_ = self._exits.pop()
                try:
                    suppress |= True if exit_(exc_type, exc_value, traceback) else False
                except:
                    exc = sys.exc_info()
        except:
            exc = sys.exc_info()
        finally:
            if exc != (None,) * 3:
                raise exc[0], exc[1], exc[2]
            else:
                return suppress

# class _contextual_task(object):
# def __init__(self, ret_func, call_func):
# self._ret_func = ret_func
# self._call_func = call_func
#
# def __call__(self, *args, **kwargs):
# args = [self._ret_func, self._call_func] + list(args)
#
# mgr_indices = []
# for i, v in enumerate(args):
# try:
# get_context_guard(v)
# except TypeError:
# pass
# else:
# mgr_indices.append(i)
#
# mgr_keys = []
# for k, v in kwargs.iteritems():
# try:
# get_context_guard(v)
# except TypeError:
# pass
# else:
# mgr_keys.append(k)
#
# with IteratedContextManager(*(args[i] for i in mgr_indices), **{k: kwargs[k] for k in mgr_keys}) as (
# mgr_args, mgr_kwargs):
# for i, v in itertools.izip(mgr_indices, mgr_args):
# args[i] = v
# for k, v in itertools.izip(mgr_keys, mgr_kwargs):
# kwargs[k] = v
# ret_func, call_func, args = overtime.utils.unpack_nfirst(args, 2)
# return ret_func(call_func(*args, **kwargs))
#
#
# def contextual_task(ret_func, call_func):
# return joblib.delayed(_contextual_task(ret_func, call_func), check_pickle=False)
