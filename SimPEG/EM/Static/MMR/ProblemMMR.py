from SimPEG import Problem, Utils
from SimPEG.EM.Static.DC.ProblemDC import BaseDCProblem
from SimPEG.EM.Static.DC.ProblemDC import Problem3D_CC as DCProblem3D_CC
from SurveyMMR import Survey
from FieldsMMR import FieldsMMR_CC
from SimPEG.Utils import sdiag
import numpy as np
from SimPEG.Utils import Zero
from scipy.constants import epsilon_0, mu_0


class MMRProblem3D_CC(DCProblem3D_CC):

    """
    3D cell centered MMR problem
    """

    _solutionDCType = 'phiSolution'
    _sourceType = 'j'
    _solutionType = 'aSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    surveyPair = Survey
    fieldsPair = FieldsMMR_CC
    Ainv_MMR = None

    def __init__(self, mesh, **kwargs):
        DCProblem3D_CC.__init__(self, mesh, **kwargs)
        #self.setBC()

    def fields(self, m):
        self.curModel = m

        if self.Ainv_MMR is not None:
            self.Ainv_MMR.clean()
        
        f = self.fieldsPair(self.mesh, self.survey)            

        #Solve DC problem
        A = self.getA()
        self.Ainv = self.Solver(A, **self.solverOpts)
        RHS = self.getRHS()
        u = self.Ainv * RHS
        Srcs = self.survey.srcList
        f[Srcs, self._solutionDCType] = u

        #Solve MMR problem        
        A_MMR = self.getA_MMR()
        self.Ainv_MMR = self.Solver(A_MMR, **self.solverOpts)
        RHS_MMR = self.getRHS_MMR(f)
        ua = self.Ainv_MMR * RHS_MMR
        f[Srcs, self._solutionType] = ua
        return f

    def getA_MMR(self):
        """

        Make the A matrix for the cell centered MMR problem

        A = CURL*((1/mu_0)*Mue)*CURL.T + DIV.T*V*(1/mu_0)*DIV*Muf

        """

        CURL = self.mesh.edgeCurl
        DIV = self.mesh.faceDiv
        # Mue = self.MeMui
        # Muf = self.MfMui
        Mue = self.mesh.getEdgeInnerProduct(1/mu_0)
        Muf = self.mesh.getFaceInnerProduct(1/mu_0)        
        V = Utils.sdiag(self.mesh.vol)

        #D = self.Div
        #G = self.Grad
        #MfRhoI = self.MfRhoI
        A = CURL*Mue*CURL.T + DIV.T*V*DIV*Muf

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getSourceTerm_MMR(self, f):
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: tuple
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.srcList        
        nF = self.mesh.nF        
        j = np.zeros((nF, len(Srcs)))
        for i, src in enumerate(Srcs):
           j[:, i] = f[src, 'j'].flatten()
        return j

    def getRHS_MMR(self, f):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm_MMR(f)
        return RHS

    #def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        #return Zero()

