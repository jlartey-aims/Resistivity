from .matutils import mkvc, ndgrid, uniqueRows
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

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
    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
            ymin, ymax = mesh.vectorNy.min(), mesh.vectorNy.max()
            yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
            if (xminTopo > xmin) or (xmaxTopo < xmax) or (yminTopo > ymin) or (ymaxTopo < ymax):
                # If not, use nearest neihbor to extrapolate them
                Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                xinds =  np.logical_or(
                    xminTopo < mesh.vectorNx, xmaxTopo > mesh.vectorNx
                    )
                yinds =  np.logical_or(
                    yminTopo < mesh.vectorNy, ymaxTopo > mesh.vectorNy
                    )
                XYOut = ndgrid(mesh.vectorNx[xinds], mesh.vectorNy[yinds])
                topoOut = Ftopo(XYOut)
                topo = np.vstack((topo, np.c_[XYOut, topoOut]))

            if gridLoc == 'CC':
                XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
                Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz), order='F')
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
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
            if (xminTopo > xmin) or (xmaxTopo < xmax):
                fill_value = "extrapolate"

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

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            if gridLoc == "CC":
                # Compute unique XY location
                uniqXY = uniqueRows(mesh.gridCC[:, :2])

                if method == "nearest":
                    Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                elif method == "linear":
                    # Check if Topo points are inside of the mesh
                    xmin, xmax = mesh.x0[0], mesh.hx.sum()+mesh.x0[0]
                    xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
                    ymin, ymax = mesh.x0[1], mesh.hy.sum()+mesh.x0[1]
                    yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
                    if (xminTopo > xmin) or (xmaxTopo < xmax) or (yminTopo > ymin) or (ymaxTopo < ymax):
                        # If not, use nearest neihbor to extrapolate them
                        Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                        xinds =  np.logical_or(
                            xminTopo < uniqXY[0][:, 0], xmaxTopo > uniqXY[0][:, 0]
                            )
                        yinds =  np.logical_or(
                            yminTopo < uniqXY[0][:, 1], ymaxTopo > uniqXY[0][:, 1]
                            )
                        inds = np.logical_or(xinds, yinds)
                        XYOut = uniqXY[0][inds, :]
                        topoOut = Ftopo(XYOut)
                        topo = np.vstack((topo, np.c_[XYOut, topoOut]))
                    Ftopo = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                else:
                    raise NotImplementedError('Only nearest and linear method are available for TREE mesh')
                actind = np.zeros(mesh.nC, dtype='bool')
                npts = uniqXY[0].shape[0]
                for i in range(npts):
                    z = Ftopo(uniqXY[0][i, :])
                    inds = uniqXY[2] == i
                    actind[inds] = mesh.gridCC[inds, 2] < z[0]
            # Need to implement
            elif gridLoc == "N":
                raise NotImplementedError('gridLoc=N is not implemented for TREE mesh')
            else:
                raise Exception("gridLoc must be either CC or N")
        else:
            raise NotImplementedError('surface2ind_topo not implemented for Quadtree or 1D mesh')

    return mkvc(actind)

def meshBuilder(xyz, h, padDist,
                padCore=np.r_[1, 1, 1], meshGlobal=None,
                expFact=1.3,
                meshType='TENSOR'):
    """
        Function to quickly generate a Tensor mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.
        If a meshGlobal is provided, the core cells will be centered
        on the underlaying mesh to reduce interpolation errors.

        :param numpy.ndarray xyz: n x 3 array of locations [x, y, z]
        :param numpy.ndarray h: 1 x 3 cell size for the core mesh
        :param numpy.ndarray h: 2 x 3 padding distances [W,E,S,N,Down,Up]
        [OPTIONAL]
        :param numpy.ndarray padCore: Number of core cells around the xyz locs
        :object SimPEG.Mesh: Base mesh used to shift the new mesh for overlap
        :param float expFact: Expension factor for padding cells [1.3]
        :param string meshType: Specify output mesh type: "TensorMesh"

        RETURNS:
        :object SimPEG.Mesh: Mesh object

    """

    assert meshType in ['TENSOR', 'TREE'], ('Revise meshType. Only ' +
                                            ' TENSOR | TREE mesh ' +
                                            'are implemented')

    # Get extent of points
    limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
    limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]
    limz = np.r_[xyz[:, 2].max(), xyz[:, 2].min()]

    # Get center of the mesh
    midX = np.mean(limx)
    midY = np.mean(limy)
    midZ = np.mean(limz)

    nCx = int((xyz[:, 0].max() - xyz[:, 0].min()) / h[0]) + padCore[0]*2
    nCy = int((xyz[:, 1].max() - xyz[:, 1].min()) / h[1]) + padCore[1]*2
    nCz = int((xyz[:, 2].max() - xyz[:, 2].min()) / h[2]) + padCore[2]*2

    if meshType == 'TENSOR':
        # Make sure the core has odd number of cells for centereing
        # on global mesh
        if meshGlobal is not None:
            nCx += 1 - int(nCx % 2)
            nCy += 1 - int(nCy % 2)
            nCz += 1 - int(nCz % 2)

        # Figure out paddings
        def expand(dx, pad):
            L = 0
            nC = 0
            while L < pad:
                nC += 1
                L = np.sum(dx * expFact**(np.asarray(range(nC))+1))

            return nC

        # Figure number of padding cells required to fill the space
        npadEast = expand(h[0], padDist[0, 0])
        npadWest = expand(h[0], padDist[0, 1])
        npadSouth = expand(h[1], padDist[1, 0])
        npadNorth = expand(h[1], padDist[1, 1])
        npadDown = expand(h[2], padDist[2, 0])
        npadUp = expand(h[2], padDist[2, 1])

        # Create discretization
        hx = [(h[0], npadWest, -expFact),
              (h[0], nCx),
              (h[0], npadEast, expFact)]
        hy = [(h[1], npadSouth, -expFact),
              (h[1], nCy), (h[1],
              npadNorth, expFact)]
        hz = [(h[2], npadDown, -expFact),
              (h[2], nCz),
              (h[2], npadUp, expFact)]

        # Create mesh
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CC0')

        # Re-set the mesh at the center of input locations
        # z-shift is different for cases when no padding cells up
        mesh.x0 = np.r_[mesh.x0[0] + midX,
                         mesh.x0[1] + midY,
                         (mesh.x0[2] -
                          mesh.hz[:(npadDown + nCz)].sum() +  # down to top of core
                          midZ +                              # up to center locs
                          h[2]*nCz/2.)]                       # up half the core

    elif meshType == 'TREE':

        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                              nCy * h[1] + padDist[1, :].sum(),
                              nCz * h[2] + padDist[2, :].sum()])

        # Number of cells at the small octree level
        nCx = 2**int(np.log2(extent/h[0]))
        nCy = 2**int(np.log2(extent/h[1]))
        nCz = 2**int(np.log2(extent/h[2]))

        # Define the mesh and origin
        mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                              np.ones(nCy)*h[1],
                              np.ones(nCz)*h[2]])

        # Set origin
        mesh.x0 = np.r_[-nCx*h[0]/2+midX, -nCy*h[1]/2+midY, -nCz*h[2]/2+midZ]

        # Refine mesh around locations
        mesh.refine(2)

        maxLevel = int(np.log2(extent / 2**2 / h[0]))

        # Create iterative refinement
        def refineFun(level, locs):

            def refine(cell):
                bsw = np.kron(np.ones(locs.shape[0]),
                              (cell.center - np.r_[cell.h] * (padCore+1/2) +
                               mesh.x0)).reshape((locs.shape[0], 3))

                tne = np.kron(np.ones(locs.shape[0]),
                              (cell.center + np.r_[cell.h] * (padCore+1/2) +
                               mesh.x0)).reshape((locs.shape[0], 3))

#                xyz = cell.center + mesh.x0
                if np.any(np.all(np.c_[np.all(bsw < locs, axis=1),
                                 np.all(tne > locs, axis=1)], axis=1)):
                    return level
                return 0
            return refine

        # xlim = np.r_[topo[:,0].min(), topo[:,0].max()]
        # ylim = np.r_[topo[:,1].min(), topo[:,1].max()]
        # zlim = np.r_[topo[:,2].min()-80, topo[:,2].max()+80]
        level = 2
        while mesh.vol.min()**(1/3) > h.min():
            print('Smallest cell: ' + str(mesh.vol.min()**(1/3)))
            print('Refining Octree mesh to level: '+str(level))
            level += 1
            mesh.refine(refineFun(level, xyz))

    # Shift tile center to closest cell in base grid
    if meshGlobal is not None:


        # Select core cells
        core = mesh.vol == mesh.vol.min()
        center = np.percentile(mesh.gridCC[core,:], 50,
                               axis=0, interpolation='nearest')
        ind = closestPoints(meshGlobal, center, gridLoc='CC')
        shift = np.squeeze(meshGlobal.gridCC[ind, :]) - center
        mesh.x0 += shift

        if isinstance(mesh, Mesh.TreeMesh):
            mesh.number()

    return mesh
