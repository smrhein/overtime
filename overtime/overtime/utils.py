

def unpack_nfirst(seq, nfirst):
    it = iter(seq)
    for _ in xrange(nfirst):
        yield next(it, None)
    yield tuple(it)


class CallBuilder(object):
    def __init__(self, object_, name='__call__', *args, **kwargs):
        self._object = object_
        self._name = name
        self._args = list(args)
        self._kwargs = dict(kwargs)
    
    def extend_args(self, iterable):
        args = self._args + list(iterable)
        return CallBuilder(self._object, self._name, *args, **self._kwargs)
    
    def update_kwargs(self, other, **kwargs):
        kwargs = dict(self._kwargs.items() + dict(other).items() + kwargs.items())
        return CallBuilder(self._object, self._name, *self._args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        args = self._args + list(args)
        kwargs = dict(self._kwargs.items() + kwargs.items())
        return getattr(self._object, self._name).__call__(*args, **kwargs)
    
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