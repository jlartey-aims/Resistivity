from SimPEG import Problem, Utils
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.ProblemDC import BaseDCProblem 
from SurveyMMR import Survey
from FieldsMMR import FieldsMMR, Fields_CC
from SimPEG.Utils import sdiag
import numpy as np
from SimPEG.Utils import Zero


class BaseMMRProblem(BaseDCProblem):
    """
        Base DC Problem
    """
    surveyPair = Survey
    fieldsPair = FieldsMMR
    Ainv = None

    def fields(self,m):
    	self.curModel = m

    	if self.Ainv is not None:
    		self.Ainv.clean()

    	f = self.fieldsPair(self.mesh,self.survey)
    	A = self.getA()

    	self.Ainv = self.Solver(A, **self.solverOpts)
    	RHS = self.getRHS()
    	a = self.Ainv*RHS
    	Srcs = self.survey.srcList
    	f[Srcs, self._solutionType] = a
    	return f

    def getSourceTerm(self):
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: tuple
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.srcList
        f = self.fieldsPair
        j = f[Srcs, 'j']

        #if self._formulation is 'EB':
         #   n = self.mesh.nN
            # return NotImplementedError

        #elif self._formulation is 'HJ':
         #   n = self.mesh.nC

        #q = np.zeros((n, len(Srcs)))

        #for i, src in enumerate(Srcs):
        #    q[:, i] = src.eval(self)
        return j

class Problem3D_CC(BaseMMRProblem):
    """
    3D cell centered MMR problem
    """

    _solutionType = 'BSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC

    def __init__(self, mesh, **kwargs):
        BaseMMRProblem.__init__(self, mesh, **kwargs)
        #self.setBC()

    def getA(self):
        """

        Make the A matrix for the cell centered MMR problem

        A = CURL*((1/mu_0)*Mue)*CURL.T + DIV.T*V*(1/mu_0)*DIV*Muf

        """

        CURL = self.mesh.edgeCurl
        DIV = self.mesh.faceDiv
        Mue = self.MeMuI
        Muf = self.MfMui
        V = Utils.sdiag(self.mesh.vol)

        #D = self.Div
        #G = self.Grad
        #MfRhoI = self.MfRhoI
        A = CURL*((1/mu_0)*Mue)*CURL.T + DIV.T*V*(1/mu_0)*DIV*Muf

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    #def getADeriv(self, u, v, adjoint=False):

        #D = self.Div
        #G = self.Grad
        #MfRhoIDeriv = self.MfRhoIDeriv

        #if adjoint:
        #    return(MfRhoIDeriv(G * u).T) * (D.T * v)

        #return D * (MfRhoIDeriv(G * u) * v)

    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()

        return RHS

    #def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        #return Zero()

