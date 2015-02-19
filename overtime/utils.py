

def unpack_nfirst(seq, nfirst):
    it = iter(seq)
    for _ in xrange(nfirst):
        yield next(it)
    yield tuple(it)


class CallBuilder(object):
    def __init__(self, object_=None, name=None, *args, **kwargs):
        self._object = object_
        self._name = name
        self._args = args
        self._kwargs = kwargs
        
    def with_object(self, object_):
        return CallBuilder(object_, self._name, *self._args, **self._kwargs)
            
    def with_name(self, name):
        return CallBuilder(self._object, name, *self._args, **self._kwargs)
    
    def extend_args(self, iterable):
        new_args = list(self._args)
        new_args.extend(iterable)
        return CallBuilder(self._object, self._name, *new_args, **self._kwargs)
    
    def update_kwargs(self, other=(), **kwargs):
        new_kwargs = dict(self._kwargs)
        new_kwargs.update(other, **kwargs)
        return CallBuilder(self._object, self._name, *self._args, **new_kwargs)
    
    def __call__(self, *args, **kwargs):
        new_args = list(self._args)
        new_args.extend(args)
        new_kwargs = dict(self._kwargs)
        new_kwargs.update(**kwargs)
        return getattr(self._object, self._name).__call__(*new_args, **new_kwargs)
    
def __print_attr(obj):
    import traceback
    import sys
    
    for a in sorted(dir(obj)):
        if a[0] != '_':
            try:
                att = getattr(obj, a)
                if not hasattr(att, '__call__'):
                    print a
                    print att
                    print
            except:
                print a
                print
                traceback.print_exc(file=sys.stdout)
