
class CommonReducer(dict):
    '''
    Object based on 'dict' that implements the binary addition (obj1 + obj1) and
    accumulation (obj += obj2). These operations pass through to the entries in
    the commonReducer.
    
    Instances of commonReducer are also callable, with the syntax:
        cr(key, value)
    this is equivalent to cr += {key: value}.
    ''' 

    DISALLOWED = ['__getinitargs__', '__getnewargs__', '__getstate__', '__setstate__']

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        result = CommonReducer(self)
        for key in other.keys():
            if key in result:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]

        return result

    def __iadd__(self, other):
        for key in other.keys():
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]

        return self

    def __mul__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] * other[key]

        return result

    def __sub__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] - other[key]

        return result

    def __div__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] / other[key]

        return result

    def __getattr__(self, attr):

        if not attr in self.DISALLOWED and all((getattr(self[key], attr, None) is not None for key in self)):

            if any((callable(getattr(self[key], attr)) for key in self)):

                def wrapperFunction(*args, **kwargs):

                    innerresult = CommonReducer({key: getattr(self[key], attr, None)(*args, **kwargs) for key in self})

                    if not all((innerresult[key] is None for key in innerresult)):
                        return innerresult

                result = wrapperFunction

            else:
                return CommonReducer({key: getattr(self[key], attr) for key in self})
        else:
            raise AttributeError('\'CommonReducer\' object has no attribute \'%s\', and it could not be satisfied through cascade lookup'%attr)

        return result

    def copy(self):

        return CommonReducer(self)

    def __call__(self, key, result):
        if key in self:
            self[key] += result
        else:
            self[key] = result