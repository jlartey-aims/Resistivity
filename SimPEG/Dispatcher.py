### PROTOTYPE INTERFACE FOR PARALLEL DISPATCHER ###

from functools import wraps

def synchronize(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        self = args[0]
        
        pr = isinstance(getattr(self, '_dispatcher', None), ParallelDispatcher)
        if pr:
            print('Parallel stuff: (start) %(prob)s.%(fn)s'%{'prob': self.__class__.__name__, 'fn': fn.__name__})

        result =  fn(*args, **kwargs)
        
        if pr:
            print('Parallel stuff: ( end ) %(prob)s.%(fn)s'%{'prob': self.__class__.__name__, 'fn': fn.__name__})
        
        return result

    return wrapper


class BaseDispatcher(object):
    
    def __init__(self, *args, **kwargs):
        print('INIT: Dispatcher!')

    def pair(self, problem):
        self._prob = problem
        print('PAIR: Dispatcher setup...')

class SerialDispatcher(BaseDispatcher):
    
    def __init__(self, *args, **kwargs):
        BaseDispatcher.__init__(self, *args, **kwargs)
        print('INIT: Serial dispatcher...')

class ParallelDispatcher(BaseDispatcher):

    remoteOnly = ['someotherattribute']
    
    def __init__(self, *args, **kwargs):
        BaseDispatcher.__init__(self, *args, **kwargs)
        print('INIT: Parallel dispatcher...')

    def pair(self, problem):
        BaseDispatcher.pair(self, problem)
        print('PAIR: Parallel dispatcher setup...')

    def interceptSetattr(self, prob, name, value):
        print('SET: Parallel dispatcher set %(prob)s.%(name)s = %(value)r'
            %{'prob': prob.__class__.__name__, 'name': name, 'value': value})

        if name in self.remoteOnly:
            print('Setting remote state...')
        else:
            raise AttributeError('Set local copy!')

    def interceptGetattr(self, prob, name):
        print('GET: Parallel dispatcher get %(prob)s.%(name)s'
            %{'prob': prob.__class__.__name__, 'name': name})

        if name in self.remoteOnly:
            return '***Value from remote state***'
        else:
            raise AttributeError('Attribute %s not in parallel namespace!'%(name,))

class StandinSurvey(object):
    
    def pair(self, problem):
        self._prob = problem
        
class StandinProblem(object):
    
    def __init__(self):
        print('INIT: Problem!')
        self._dispatcher = SerialDispatcher()

    def __setattr__(self, name, value):
        d = getattr(self, '_dispatcher', None)

        if isinstance(d, ParallelDispatcher):
            try:
                d.interceptSetattr(self, name, value)
            except AttributeError:
                super(self.__class__, self).__setattr__(name, value)
            finally:
                return

        else:
            super(self.__class__, self).__setattr__(name, value)

    def __getattr__(self, name):
        d = super(self.__class__, self).__getattribute__('_dispatcher')

        if isinstance(d, ParallelDispatcher):
            return d.interceptGetattr(self, name)
    
    def pair(self, survey, dispatcher=None):
        
        self._survey = survey
        self._survey.pair(self)
        
        if dispatcher is not None:
            self._dispatcher = dispatcher
            print('PAIR: Problem setup...')
            self._dispatcher.pair(self)
    
    @synchronize
    def dosomething(self):
        print('Doing something!')