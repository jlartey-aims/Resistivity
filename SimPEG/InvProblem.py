from __future__ import print_function
from . import Utils
from . import Props
from . import DataMisfit
from . import Regularization
from . import mkvc

import properties
import numpy as np
import scipy.sparse as sp
import gc
import logging

# Wrap the solver
SolverICG = Utils.SolverUtils.SolverWrapI(sp.linalg.cg, checkAccuracy=False)

class BaseInvProblem(Props.BaseSimPEG):
    """BaseInvProblem(dmisfit, reg, opt)"""

    #: Trade-off parameter
    beta = 1.0

    #: Print debugging information
    debug = False

    #: Set this to a SimPEG.Utils.Counter() if you want to count things
    counter = None

    #: DataMisfit
    dmisfit = None

    #: Regularization
    reg = None

    #: Optimization program
    opt = None

    #: List of strings, e.g. ['_MeSigma', '_MeSigmaI']
    deleteTheseOnModelUpdate = []

    model = Props.Model("Inversion model.")

    @properties.observer('model')
    def _on_model_update(self, value):
        """
            Sets the current model, and removes dependent properties
        """
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    def __init__(self, dmisfit, reg, opt, **kwargs):
        super(BaseInvProblem, self).__init__(**kwargs)
        assert isinstance(dmisfit, DataMisfit.BaseDataMisfit), 'dmisfit must be a DataMisfit class.'
        assert isinstance(reg, Regularization.BaseRegularization), 'reg must be a Regularization class.'
        self.dmisfit = dmisfit
        self.reg = reg
        self.opt = opt
        self.prob, self.survey = dmisfit.prob, dmisfit.survey
        # TODO: Remove: (and make iteration printers better!)
        self.opt.parent = self
        self.reg.parent = self
        self.dmisfit.parent = self

    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug:
            print('Calling InvProblem.startup')

        if self.reg.mref is None:
            print('SimPEG.InvProblem will set Regularization.mref to m0.')
            self.reg.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.model = m0

        print("""SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
                    ***Done using same Solver and solverOpts as the problem***""")
        self.opt.bfgsH0 = self.prob.Solver(self.reg.eval2Deriv(self.model), **self.prob.solverOpts)

    @property
    def warmstart(self):
        return getattr(self, '_warmstart', [])

    @warmstart.setter
    def warmstart(self, value):
        assert type(value) is list, 'warmstart must be a list.'
        for v in value:
            assert type(v) is tuple, 'warmstart must be a list of tuples (m, u).'
            assert len(v) == 2, 'warmstart must be a list of tuples (m, u). YOURS IS NOT LENGTH 2!'
            assert isinstance(v[0], np.ndarray), 'first warmstart value must be a model.'
        self._warmstart = value

    def getFields(self, m, store=False, deleteWarmstart=True):
        f = None

        for mtest, u_ofmtest in self.warmstart:
            if m is mtest:
                f = u_ofmtest
                if self.debug:
                    print('InvProb is Warm Starting!')
                break

        if f is None:
            f = self.prob.fields(m)

        if deleteWarmstart:
            self.warmstart = []
        if store:
            self.warmstart += [(m, f)]

        return f

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        # Log
        logger = logging.getLogger(
            'SimPEG.InvProblem.BaseInvProblem.evalFunction')
        logger.info('Starting calculations in invProb.evalFunction')
        # Initialize
        self.model = m
        gc.collect()


        f = self.getFields(m, store=(return_g is False and return_H is False))

        logger.debug('Solve the objective function')
        phi_d = self.dmisfit.eval(m, f=f)
        phi_m = self.reg.eval(m)

        # This is a cheap matrix vector calculation.
        self.dpred = self.survey.dpred(m, f=f)

        self.phi_d, self.phi_d_last = phi_d, self.phi_d
        self.phi_m, self.phi_m_last = phi_m, self.phi_m

        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            logger.debug('Solving the objective function gradient')
            phi_dDeriv = self.dmisfit.evalDeriv(m, f=f)
            phi_mDeriv = self.reg.evalDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            logger.debug('Solving the objective function Hessian')
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.eval2Deriv(m, v, f=f)
                phi_m2Deriv = self.reg.eval2Deriv(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (H,)
        return out if len(out) > 1 else out[0]

class storeFactors_InvProblem(BaseInvProblem):
    """
    Class aimed at taking advantage of the use of Pardiso Direct solver

    Inherits the BaseInvProblem but extends the evalFunction to store
    the factors within the problem for reuse.

    Assumes to be a NSEM problem

    """

    def __init__(self, dmisfit, reg, opt, **kwargs):
        super(storeFactors_InvProblem, self).__init__(dmisfit, reg, opt, **kwargs)


    def getFields(self, m, store=False, deleteWarmstart=True):
        """
        Function to get the fields
        """

        logger = logging.getLogger(
            'SimPEG.InvProblem.storeFactors_InvProblem.getFields')
        logger.info('Starting calculations ')

        # Set f to None
        f = None

        # Calcualte the factors if the aren't there
        if self.prob._factor_dict is None:
            self.prob.model = m
            self.prob._factor_dict = {}
            for freq in self.prob.survey.freqs:
                logger.debug('Starting factoring for {:.3e}'.format(freq))
                # Get the system
                A = self.prob.getA(freq)
                # Factor  the system
                Ainv = self.prob.Solver(A, **self.prob.solverOpts)
                self.prob._factor_dict[freq] = Ainv

        # Check if invProb is warmstarting
        for mtest, u_ofmtest in self.warmstart:
            if m is mtest:
                f = u_ofmtest
                if self.debug:
                    print('InvProb is Warm Starting!')
                break

        # If not warmstarting and fields are None, calculate
        if f is None:
            f = self.prob.fields(m)

        if deleteWarmstart:
            self.warmstart = []
        if store:
            self.warmstart += [(m, f)]

        return f

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        # Log
        logger = logging.getLogger(
            'SimPEG.InvProblem.storeFactors_InvProblem.evalFunction')
        logger.info('Starting calculations ')
        # Initialize
        self.model = m
        gc.collect()

        # Set the model to the problem
        self.prob.model = m

        # Get the fields
        f = self.getFields(m, store=(return_g is False and return_H is False))

        logger.debug('Calculate the objective function parts')
        phi_d = self.dmisfit.eval(m, f=f)
        phi_m = self.reg.eval(m)

        # This is a cheap matrix vector calculation.
        self.dpred = self.survey.dpred(m, f=f)

        self.phi_d, self.phi_d_last = phi_d, self.phi_d
        self.phi_m, self.phi_m_last = phi_m, self.phi_m

        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            logger.debug('Solving the objective function gradient')
            phi_dDeriv = self.dmisfit.evalDeriv(m, f=f)
            phi_mDeriv = self.reg.evalDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            logger.debug('Solving the objective function Hessian')
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.eval2Deriv(m, v, f=f)
                phi_m2Deriv = self.reg.eval2Deriv(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (H,)
        return out if len(out) > 1 else out[0]


class eachFreq_InvProblem(BaseInvProblem):
    """
    Class aimed at taking advantage of the use of Pardiso Direct solver

    Inherits the BaseInvProblem but extends the evalFunction to extract
    the looping over frequencies/sources for the base functions.

    Assumes to be a FDEM problem

    """

    def __init__(self, dmisfit, reg, opt, **kwargs):
        super(eachFreq_InvProblem, self).__init__(dmisfit, reg, opt, **kwargs)


    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)

        Sets up the evaluation of the objective function,
        gradient and Hessian (if needed).
        """
        # Log
        logger = logging.getLogger(
            'SimPEG.InvProblem.eachFreq_InvProblem.evalFunction')
        logger.info('Starting calculations')
        # Initilize
        # Set model
        self.model = m
        gc.collect()

        # Calculate the model objectives
        phi_m = self.reg.eval(m)
        phi_mDeriv = self.reg.evalDeriv(m)

        # Alias the problem
        problem = self.survey.prob
        problem.model = m
        # Set up the containers
        # Predicted data
        data_pred = problem.dataPair(problem.survey)
        # Observed data
        data_obs = problem.dataPair(problem.survey, self.survey.dobs)
        # Data uncertainty
        data_wd = problem.dataPair(self.survey, self.dmisfit.Wd)
        # Data objective (phi_d)
        data_phi_d = problem.dataPair(self.survey)
        # Gradient multiplecation vector
        data_vec_g = problem.dataPair(self.survey)

        # The Fields
        fields = problem.fieldsPair(problem.mesh, self.survey)
        phi_dDeriv = np.zeros(m.size)
        sD = np.zeros(m.size)
        for freq in self.survey.freqs:
            # Initialize at each loop


            # Factorize
            logger.debug('Working on frequency {:.3e} Hz'.format(freq))
            logger.debug('Factorization starting...')
            A = problem.getA(freq)
            Ainv = problem.Solver(A, *problem.solverOpts)
            logger.debug('Factorization completed')
            # Calculate fields
            logger.debug('Fields: Starting caluculation')
            fields = problem._solve_fields_atFreq(Ainv, freq, fields)

            # Calcualte the residual
            for src in self.survey.getSrcByFreq(freq):
                for rx in src.rxList:
                    data_pred[src, rx] = rx.eval(src, problem.mesh, fields)
                    data_phi_d[src, rx] = data_wd[src, rx] * (data_pred[src, rx] - data_obs[src, rx])
                    data_vec_g[src, rx] = data_wd[src, rx] * data_phi_d[src, rx]

            logger.debug('Fields: Finished caluculation')
            # Calculate the gradient
            if return_g or return_H:
                logger.debug('Gradient: Starting calculation')
                phi_f_dDeriv = problem._Jtvec_atFreq(Ainv, freq, data_vec_g, fields, None)
                phi_dDeriv += phi_f_dDeriv
                phi_f_deriv = phi_f_dDeriv + self.beta * phi_mDeriv
                logger.debug('Gradient: Finished calculation')

            # Need to set up the Hessian
            if return_H:
                logger.info('Search Direction: Starting caluculation')
                # Make it to a function
                def H_f_fun(v):
                    vec_t = problem.dataPair(self.survey)
                    vec_t[src, rx] = data_wd[src, rx] * (
                        data_wd[src, rx] * problem._Jvec_atFreq(
                            Ainv, freq, v, fields, vec_t)[src, rx])
                    phi_f_d2Deriv = problem._Jtvec_atFreq(
                        Ainv, freq, vec_t, fields, None)
                    phi_m2Deriv = self.reg.eval2Deriv(m, v=v)

                    return phi_f_d2Deriv + self.beta * phi_m2Deriv

                # Make the function into an linear operator
                H_f = sp.linalg.LinearOperator( (m.size, m.size), H_f_fun, dtype=m.dtype )

                # Iterativily solve the system.
                H_f_inv = SolverICG(H_f, M=self.opt.approxHinv, tol=self.opt.tolCG, maxiter=self.opt.maxIterCG)
                sD_f = H_f_inv * (-phi_f_deriv)
                sD += sD
                logger.info('Search Direction: Finished calcualtions')
                del H_f, H_f_inv #: Doing this saves memory, as it is not needed in the rest of the computations.
            Ainv.clean()


        # Calculate the parameters needed
        R = mkvc(data_phi_d)
        phi_d = 0.5*np.vdot(R, R)

        # This is a cheap matrix vector calculation.
        self.dpred = self.survey.dpred(m, f=fields) # Could change this.
        self.phi_d, self.phi_d_last = phi_d, self.phi_d
        self.phi_m, self.phi_m_last = phi_m, self.phi_m
        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            out += (sD,)
        return out if len(out) > 1 else out[0]
