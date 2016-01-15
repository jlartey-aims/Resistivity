

class SimPEGException(Exception):

    def __init__(self, reason=''):
        self.reason = reason

    def __str__(self):
        return '%s: %s' %(self.__class__.__name__,  self.reason)


class PairingException(SimPEGException):
    pass
