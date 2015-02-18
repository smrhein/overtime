import overtime.utils
import sys
import itertools
import joblib


def get_context_guard(mgr):
    try:
        return mgr.__enter__, mgr.__exit__
    except AttributeError:
        raise TypeError('argument is not a context manager')
    

class IteratedContextManager(object): 
    def __init__(self, *args, **kwargs):
        self._args = list(args[::-1])
        self._kwargs = kwargs
        self._exits = []
        
    def __enter__(self):
        exc = None
        
        as_args = []
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
            
        as_kwargs = []
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

        self._exits.reverse()                            
        if exc and not self.__exit__(*exc):
            raise exc[0], exc[1], exc[2]   
        else:      
            return as_args, dict(as_kwargs)
    
    def __exit__(self, exc_type, exc_value, traceback):
        exc = (exc_type, exc_value, traceback)
        while self._exits:
            exit_ = self._exits.pop()
            try:
                if exit_(*exc):
                    exc = None
            except:
                exc = sys.exc_info()
        if exc and exc != (exc_type, exc_value, traceback):
            raise exc[0], exc[1], exc[2]
    

class _contextual_task(object):
    def __init__(self, ret_func, call_func):
        self._ret_func = ret_func
        self._call_func = call_func
        
    def __call__(self, *args, **kwargs):
        args = [self._ret_func, self._call_func] + list(args)
        
        mgr_indices = []
        for i, v in enumerate(args):
            try:
                get_context_guard(v)
            except TypeError:
                pass
            else:
                mgr_indices.append(i)

        mgr_keys = []
        for k, v in kwargs.iteritems():
            try:
                get_context_guard(v)
            except TypeError:
                pass
            else:
                mgr_keys.append(k)
            
        with IteratedContextManager(*(args[i] for i in mgr_indices), **{kwargs[k] for k in mgr_keys}) as (mgr_args, mgr_kwargs):            
            for i, v in itertools.izip(mgr_indices, mgr_args):
                args[i] = v
            for k, v in itertools.izip(mgr_keys, mgr_kwargs):
                kwargs[k] = v                
            ret_func, call_func, args = overtime.utils.unpack_nfirst(args, 2)
            return ret_func(call_func(*args, **kwargs))
        
        
def contextual_task(ret_func, call_func):
    return joblib.delayed(_contextual_task(ret_func, call_func), check_pickle=False)