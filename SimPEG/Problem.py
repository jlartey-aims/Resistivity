import Utils, Survey, Models, numpy as np, scipy.sparse as sp
Solver = Utils.SolverUtils.Solver
import Maps, Mesh, Exceptions
from Fields import Fields, TimeFields

class BaseProblem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.
    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None   #: A SimPEG.Utils.Counter object

    surveyPair = Survey.BaseSurvey   #: A SimPEG.Survey Class
    mapPair    = Maps.IdentityMap    #: A SimPEG.Map Class

    Solver = Solver   #: A SimPEG Solver class.
    solverOpts = {}   #: Sovler options as a kwarg dict

    PropMap = None    #: A SimPEG PropertyMap class.

    def __init__(self, mesh, mapping=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
        self.mesh = mesh
        self.mapping = mapping or Maps.IdentityMap(mesh)

    @property
    def mapping(self):
        "A SimPEG.Map instance or a property map is PropMap is not None"
        return getattr(self, '_mapping', None)
    @mapping.setter
    def mapping(self, val):
        if self.PropMap is None:
            val._assertMatchesPair(self.mapPair)
            self._mapping = val
        else:
            self._propMapMapping = val
            self._mapping = self.PropMap(val)

    @property
    def survey(self):
        """
        The survey object for this problem.
        """
        return getattr(self, '_survey', None)

    def pair(self, survey):
        """Bind a survey to this problem instance using pointers."""
        assert isinstance(survey, self.surveyPair), "Survey must be an instance of a %s class."%(self.surveyPair.__name__)
        if survey.ispaired:
            raise Exception("The survey object is already paired to a problem. Use survey.unpair()")
        try:
            self._survey = survey
            self._validatePairing()
        except Exceptions.PairingException, e:
            self._survey = None
            raise e
        survey._prob = self

    def _validatePairing(self):
        """Called when the pair is done, raise a SimPEG.Exceptions.PairingException if unsuccessful"""
        pass

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired: return
        self.survey._prob = None
        self._survey = None


    deleteTheseOnModelUpdate = [] # List of strings, e.g. ['_MeSigma', '_MeSigmaI']

    @property
    def curModel(self):
        """
            Sets the current model, and removes dependent mass matrices.
        """
        return getattr(self, '_curModel', None)
    @curModel.setter
    def curModel(self, value):
        if value is self.curModel:
            return # it is the same!
        if self.PropMap is not None:
            self._curModel = self.mapping(value)
        else:
            self._curModel = Models.Model(value, self.mapping)
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    @property
    def ispaired(self):
        """True if the problem is paired to a survey."""
        return self.survey is not None

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """Jvec(m, v, u=None)

            Effect of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv
        """
        raise NotImplementedError('J is not yet implemented.')

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """Jtvec(m, v, u=None)

            Effect of transpose of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv
        """
        raise NotImplementedError('Jt is not yet implemented.')


    @Utils.timeIt
    def Jvec_approx(self, m, v, u=None):
        """Jvec_approx(m, v, u=None)

            Approximate effect of J(m) on a vector v

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: approxJv
        """
        return self.Jvec(m, v, u)

    @Utils.timeIt
    def Jtvec_approx(self, m, v, u=None):
        """Jtvec_approx(m, v, u=None)

            Approximate effect of transpose of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv
        """
        return self.Jtvec(m, v, u)

    def fields(self, m):
        """
            The field given the model.

            :param numpy.array m: model
            :rtype: numpy.array
            :return: u, the fields

        """
        raise NotImplementedError('fields is not yet implemented.')


class BaseTimeProblem(BaseProblem):
    """Sets up that basic needs of a time domain problem."""

    @property
    def timeSteps(self):
        """Sets/gets the timeSteps for the time domain problem.

        You can set as an array of dt's or as a list of tuples/floats.
        Tuples must be length two with [..., (dt, repeat), ...]

        For example, the following setters are the same::

            prob.timeSteps = [(1e-6, 3), 1e-5, (1e-4, 2)]
            prob.timeSteps = np.r_[1e-6,1e-6,1e-6,1e-5,1e-4,1e-4]

        """
        return getattr(self, '_timeSteps', None)

    @timeSteps.setter
    def timeSteps(self, value):
        if isinstance(value, np.ndarray):
            self._timeSteps = value
            del self.timeMesh
            return

        self._timeSteps = Utils.meshTensor(value)
        del self.timeMesh

    @property
    def nT(self):
        "Number of time steps."
        return self.timeMesh.nC

    @property
    def t0(self):
        return getattr(self, '_t0', 0.0)
    @t0.setter
    def t0(self, value):
        assert Utils.isScalar(value), 't0 must be a scalar'
        del self.timeMesh
        self._t0 = float(value)

    @property
    def times(self):
        "Modeling times"
        return self.timeMesh.vectorNx

    @property
    def timeMesh(self):
        if getattr(self, '_timeMesh', None) is None:
            self._timeMesh = Mesh.TensorMesh([self.timeSteps], x0=[self.t0])
        return self._timeMesh
    @timeMesh.deleter
    def timeMesh(self):
        if hasattr(self, '_timeMesh'):
            del self._timeMesh


class GlobalProblem(BaseProblem):
    """

        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.

        This is handy for working with lots of sources,

    """

    surveyKwargs = {}
    probKwargs   = {}

    def __init__(self, SubProblem, globalMesh, mapping=None, **kwargs):

        # assert isclass??(SubProblem, BaseProblem), "SubProblem must be a SimPEG.Problem.BaseProblem object."
        self.surveyPair = SubProblem.surveyPair
        self.PropMap = SubProblem.PropMap
        self.mapPair = SubProblem.mapPair
        self.SubProblem = SubProblem

        Utils.setKwargs(self, **kwargs)
        assert isinstance(globalMesh, Mesh.BaseMesh), "globalMesh must be a SimPEG.Mesh object."
        self.globalMesh = globalMesh
        self.mapping = mapping or Maps.IdentityMap(mesh)

    @property
    def groups(self):
        """
            List of lists/integers to say how the sources are grouped.

            e.g.

            survey.srcList = [s0,s1,s2,s3,s4]
            groups = [ [0,4], [1,3], 2 ]
        """
        if getattr(self, '_groups', None) is None:
            if not self.ispaired: return None
            self._groups = range(self.survey.nSrc)
        return self._groups
    @groups.setter
    def groups(self, val):
        assert type(val) is list, 'This should be an list of groups'
        if self.ispaired:
            for g in val:
                assert type(g) in [int, list], 'Must be an integer or a list'
                if type(g) is int:
                    assert g >= 0 and g < self.survey.nSrc, '%d is outside the number of sources in the surveys list'%g
                if type(g) is list:
                    for sg in g:
                        assert type(g) is int, 'Must be an integer or a list'
                        assert g >= 0 and g < self.survey.nSrc, '%d is outside the number of sources in the surveys list'%g
            assert len(val) == len(self.survey.srcList), 'The groups must be the same length as the srcList in the survey'
        self._groups = val
        self._nGroups = None

    @property
    def meshes(self):
        if getattr(self, '_meshes', None) is None:
            if not self.ispaired: return None
            self._meshes = [self.globalMesh]*self.nGroups
        return self._meshes
    @meshes.setter
    def meshes(self, val):
        assert type(val) is list
        if self.ispaired:
            assert len(val) == self.nGroups
        self._meshes = val

    @property
    def nGroups(self):
        if getattr(self, '_groups', None) is None:
            return None
        return len(self.groups)

    def _validatePairing(self):
        try:
            self.groups = self.groups # check the assumptions for the grouping
        except Exception, e:
            raise Exceptions.PairingException(reason='The grouping does not match the survey')
        if self.nGroups is not len(self.meshes):
            raise Exceptions.PairingException(reason='The meshes are not the the same length as the number of groups')

    def getSubProblem(self, ind):

        assert self.ispaired, 'You must be paired to a survey'
        assert type(ind) in [int,long] and ind >= 0 and ind < self.nGroups, 'ind must be an index into the group list'

        subMesh = self.meshes[ind]
        subMap  = Maps.IdentityMap(subMesh) # this is probably a mesh2mesh mapping?

        if self.PropMap is None:
            prob = self.SubProblem(subMesh, mapping=subMap * self.mapping, **self.probKwargs)
        else:
            prob = self.SubProblem(subMesh, mapping=subMap * self._propMapMapping, **self.probKwargs)

        survey = self.survey.__class__(srcList=self.survey.srcList[self.groups[ind]], **self.surveyKwargs)
        prob.pair(survey)

        return prob


if __name__ == '__main__':


    from SimPEG import *
    from SimPEG import EM
    from scipy.constants import mu_0
    from pymatsolver import MumpsSolver

    cs = 10.
    ncx, ncy, ncz = 10, 10, 10
    npad = 4
    freq = 1e2

    hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
    hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.TensorMesh([hx,hy,hz], 'CCC')

    mapping = Maps.ExpMap(mesh)

    x = np.linspace(-10,10,5)
    XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])
    rxList = EM.FDEM.Rx(XYZ, 'exi')
    Src0 = EM.FDEM.Src.MagDipole([rxList],loc=np.r_[0.,0.,0.], freq=freq)
    Src1 = EM.FDEM.Src.MagDipole([rxList],loc=np.r_[0.,0.,0.], freq=freq)


    prb0 = EM.FDEM.Problem_b(mesh, mapping=mapping, Solver=MumpsSolver)
    survey = EM.FDEM.Survey([Src0])
    prb0.pair(survey)
    prb1 = EM.FDEM.Problem_b(mesh, mapping=mapping, Solver=MumpsSolver)
    survey = EM.FDEM.Survey([Src1])
    prb1.pair(survey)



    sig = 1e-1
    sigma = np.ones(mesh.nC)*sig
    sigma[mesh.gridCC[:,2] > 0] = 1e-8
    m = np.log(sigma)

    GP = GlobalProblem(EM.FDEM.Problem_b, mesh, mapping=mapping, meshes=[mesh,mesh])
    survey = EM.FDEM.Survey([Src0, Src1])
    GP.pair(survey)

    gp1 = GP.getSubProblem(0)
    gp1.Solver = MumpsSolver

    pu = prb0.fields(m)
    gpu = gp1.fields(m)

    bfz = mesh.r(pu[Src0, 'b'],'F','Fz','M')
    bfz = mesh.r(gpu[Src0, 'b'],'F','Fz','M')
    x = np.linspace(-55,55,12)
    XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])
    P = mesh.getInterpolationMat(XYZ, 'Fz')

    # an = EM.Analytics.FDEM.hzAnalyticDipoleF(x, Src0.freq, sig)

    # diff = np.log10(np.abs(P*np.imag(pu[Src0, 'b']) - mu_0*np.imag(an)))
    # diff = np.log10(np.abs(P*np.imag(gpu[Src0, 'b']) - mu_0*np.imag(an)))

    import matplotlib.pyplot as plt
    plt.plot(x,np.log10(np.abs(P*np.imag(pu[Src0, 'b']))), 'r-s')
    plt.plot(x,np.log10(np.abs(P*np.imag(gpu[Src0, 'b']))), 'b')
    # plt.plot(x,np.log10(np.abs(mu_0*np.imag(an))), 'r')
    # plt.plot(x,diff,'g')
    plt.show()




