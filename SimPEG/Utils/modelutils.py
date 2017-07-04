from .matutils import mkvc, ndgrid, kron3, speye
from .meshutils import closestPoints
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata, interp1d, NearestNDInterpolator


def surface2ind_topo(mesh, topo, gridLoc='CC', method='nearest', fill_value=np.nan):
    """
    Get active indices from topography
    
    Parameters
    ----------

    :param TensorMesh mesh: TensorMesh object on which to discretize the topography
    :param numpy.ndarray topo: [X,Y,Z] topographic data
    :param str gridLoc: 'CC' or 'N'. Default is 'CC'.
                        Discretize the topography
                        on cells-center 'CC' or nodes 'N'
    :param str method: 'nearest' or 'linear' or 'cubic'. Default is 'nearest'.
                       Interpolation method for the topographic data
    :param float fill_value: default is np.nan. Filling value for extrapolation

    Returns
    -------

    :param numpy.array actind: index vector for the active cells on the mesh
                               below the topography
    """

    if mesh.dim == 3:

        if gridLoc == 'CC':
            XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
            Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz), order='F')
            #gridTopo = Ftopo(XY)
            gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value)
            actind = [gridTopo >= Zcc[:, ixy] for ixy in range(np.prod(mesh.vnC[2]))]
            actind = np.hstack(actind)

        elif gridLoc == 'N':

            XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
            gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value)
            gridTopo = gridTopo.reshape(mesh.vnN[:2], order='F')

            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

            # TODO: this will only work for tensor meshes
            Nz = mesh.vectorNz[1:]
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                for jj in range(mesh.nCy):
                     actind[ii, jj, :] = [np.all(gridTopo[ii:ii+2, jj:jj+2] >= Nz[kk]) for kk in range(len(Nz))]

    elif mesh.dim == 2:

        Ftopo = interp1d(topo[:, 0], topo[:, 1], fill_value=fill_value, kind=method)

        if gridLoc == 'CC':
            gridTopo = Ftopo(mesh.gridCC[:, 0])
            actind = mesh.gridCC[:, 1] <= gridTopo

        elif gridLoc == 'N':

            gridTopo = Ftopo(mesh.vectorNx)
            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

            # TODO: this will only work for tensor meshes
            Ny = mesh.vectorNy[1:]
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                actind[ii, :] = [np.all(gridTopo[ii: ii+2] > Ny[kk]) for kk in range(len(Ny))]

    else:
        raise NotImplementedError('surface2ind_topo not implemented for 1D mesh')

    return mkvc(actind)


def MinCurvatureInterp(mesh, pts, vals, tol=1e-5, iterMax=None):
    """
        Interpolate properties with a minimum curvature interpolation
        :param mesh: SimPEG mesh object
        :param pts:  numpy.array of size n-by-3 of point locations
        :param vals: numpy.array of size n-by-m of values to be interpolated
        :return: numpy.array of size nC-by-m of interpolated values

    """

    assert pts.shape[0] == vals.shape[0], ("Number of interpolated pts " +
                                           "must match number of vals")

    # These two functions are almost identical to the ones in Discretize,
    # Only difference is the averaging operator for the boundary cells.
    # Can we add a switch in Discretize to have the options?

    def av_extrap(n):
        """Define 1D averaging operator from cell-centers to nodes."""
        Av = (
            sp.spdiags(
                (0.5 * np.ones((n, 1)) * [1, 1]).T,
                [-1, 0],
                n + 1, n,
                format="csr"
            ) #+
            #sp.csr_matrix(([0.5, 0.5], ([0, n], [0, n-1])), shape=(n+1, n))
        )
        Av[0, 1], Av[-1, -2] = 0.5, 0.5
        return Av

    def aveCC2F(mesh):
        "Construct the averaging operator on cell cell centers to faces."
        if mesh.dim == 1:
            aveCC2F = av_extrap(self.nCx)
        elif mesh.dim == 2:
            aveCC2F = sp.vstack((
                sp.kron(speye(mesh.nCy), av_extrap(mesh.nCx)),
                sp.kron(av_extrap(mesh.nCy), speye(mesh.nCx))
            ), format="csr")
        elif mesh.dim == 3:
            aveCC2F = sp.vstack((
                kron3(
                    speye(mesh.nCz), speye(mesh.nCy), av_extrap(mesh.nCx)
                ),
                kron3(
                    speye(mesh.nCz), av_extrap(mesh.nCy), speye(mesh.nCx)
                ),
                kron3(
                    av_extrap(mesh.nCz), speye(mesh.nCy), speye(mesh.nCx)
                )
            ), format="csr")
        return aveCC2F

    Ave = aveCC2F(mesh)

    # Get the grid location
    ijk = closestPoints(mesh, pts, gridLoc='CC')

    count = 0
    residual = 1.

    m = np.zeros((mesh.nC, vals.shape[1]))

    # Begin with neighrest primer
    for ii in range(m.shape[1]):
        F = NearestNDInterpolator(mesh.gridCC[ijk], vals[:, ii])
        m[:, ii] = F(mesh.gridCC)

    while np.all([count < iterMax, residual > tol]):
        m[ijk, :] = vals
        mtemp = m
        m = mesh.aveF2CC * (Ave * m)
        residual = np.linalg.norm(m-mtemp)/np.linalg.norm(mtemp)
        count += 1

    return m
