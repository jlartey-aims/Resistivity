from SimPEG.EM.Static.DC import BaseDCProblem


class BaseMMRProblem(BaseDCProblem):
    """
        Base DC Problem
    """
    surveyPair = Survey
    fieldsPair = FieldsDC
    Ainv = None

if __name__ == '__main__':
    pass
