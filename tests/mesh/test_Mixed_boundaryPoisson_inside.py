from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import unittest
import matplotlib.pyplot as plt
from SimPEG import Mesh, Tests, Utils, Solver
from pymatsolver import BicgJacobiSolver

MESHTYPES = ['uniformTensorMesh']


def getfaceBoundaryInd(mesh, activeCC):
    # Compute boundary indices
    out = mesh.faceDiv.T*activeCC.astype(float)
    indsn = out < 0.
    indsp = out > 0.

    if mesh.dim == 2:
        indsnx = indsn[:mesh.nFx]
        indsny = indsn[mesh.nFx:mesh.nFx+mesh.nFy]
        indspx = indsp[:mesh.nFx]
        indspy = indsp[mesh.nFx:mesh.nFx+mesh.nFy]
        return indsnx, indspx, indsny, indspy

    if mesh.dim == 3:
        indsnx = indsn[:mesh.nFx]
        indsny = indsn[mesh.nFx:mesh.nFx+mesh.nFy]
        indsnz = indsn[mesh.nFx+mesh.nFy:mesh.nFx+mesh.nFy+mesh.nFz]
        indspx = indsp[:mesh.nFx]
        indspy = indsp[mesh.nFx:mesh.nFx+mesh.nFy]
        indspz = indsp[mesh.nFx+mesh.nFy:mesh.nFx+mesh.nFy+mesh.nFz]
        return indsnx, indspx, indsny, indspy, indsnz, indspz


def getxBCyBC_CC(mesh, activeCC, alpha, beta, gamma):
    """
    This is a subfunction generating mixed-boundary condition:

    .. math::

        \nabla \cdot \vec{j} = -\nabla \cdot \vec{j}_s = q

        \rho \vec{j} = -\nabla \phi \phi

        \alpha \phi + \beta \frac{\partial \phi}{\partial r} = \gamma \ at \ r
        = \partial \Omega

        xBC = f_1(\alpha, \beta, \gamma)
        yBC = f(\alpha, \beta, \gamma)

    Computes xBC and yBC for cell-centered discretizations
    """

    # 2D
    if mesh.dim == 2:
        if (len(alpha) != 4 or len(beta) != 4 or len(gamma) != 4):
            raise Exception("Lenght of list, alpha should be 4")

        fxm, fxp, fym, fyp = getfaceBoundaryInd(mesh, activeCC)
        nBC = fxm.sum()+fxp.sum()+fym.sum()+fyp.sum()

        # Get alpha, beta, gamma for in and out faces
        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]

        # Compute size of boundary cells for in and out faces
        h_xm = mesh.gridFx[np.arange(mesh.nFx)[fxm]+1, 0] - mesh.gridFx[np.arange(mesh.nFx)[fxm], 0]
        h_xp = mesh.gridFx[np.arange(mesh.nFx)[fxp], 0] - mesh.gridFx[np.arange(mesh.nFx)[fxp]-1, 0]
        h_ym = mesh.gridFy[np.arange(mesh.nFy)[fym]+mesh.vnFx[0]-1, 1] - mesh.gridFy[np.arange(mesh.nFy)[fym], 1]
        h_yp = mesh.gridFy[np.arange(mesh.nFy)[fyp], 1] - mesh.gridFy[np.arange(mesh.nFy)[fyp]-mesh.vnFx[0]+1, 1]

        # Compute a and b then x and y
        a_xm = gamma_xm/(0.5*alpha_xm-beta_xm/h_xm)
        b_xm = (0.5*alpha_xm+beta_xm/h_xm)/(0.5*alpha_xm-beta_xm/h_xm)
        a_xp = gamma_xp/(0.5*alpha_xp-beta_xp/h_xp)
        b_xp = (0.5*alpha_xp+beta_xp/h_xp)/(0.5*alpha_xp-beta_xp/h_xp)

        a_ym = gamma_ym/(0.5*alpha_ym-beta_ym/h_ym)
        b_ym = (0.5*alpha_ym+beta_ym/h_ym)/(0.5*alpha_ym-beta_ym/h_ym)
        a_yp = gamma_yp/(0.5*alpha_yp-beta_yp/h_yp)
        b_yp = (0.5*alpha_yp+beta_yp/h_yp)/(0.5*alpha_yp-beta_yp/h_yp)

        xBC_xm = 0.5*a_xm
        xBC_xp = 0.5*a_xp/b_xp
        yBC_xm = 0.5*(1.-b_xm)
        yBC_xp = 0.5*(1.-1./b_xp)
        xBC_ym = 0.5*a_ym
        xBC_yp = 0.5*a_yp/b_yp
        yBC_ym = 0.5*(1.-b_ym)
        yBC_yp = 0.5*(1.-1./b_yp)

        xBC_x = np.zeros(mesh.nFx)
        xBC_y = np.zeros(mesh.nFy)
        yBC_x = np.zeros(mesh.nFx)
        yBC_y = np.zeros(mesh.nFy)

        xBC_x[fxm] = xBC_xm
        xBC_x[fxp] = xBC_xp
        yBC_x[fxm] = yBC_xm
        yBC_x[fxp] = yBC_xp
        xBC_y[fym] = xBC_ym
        xBC_y[fyp] = xBC_yp
        yBC_y[fym] = yBC_ym
        yBC_y[fyp] = yBC_yp

        fx = np.logical_or(fxm, fxp)
        fy = np.logical_or(fym, fyp)

        xBC_x = xBC_x[fx]
        yBC_x = yBC_x[fx]
        xBC_y = xBC_y[fy]
        yBC_y = yBC_y[fy]

        xBC = np.r_[xBC_x, xBC_y]
        yBC = np.r_[yBC_x, yBC_y]

    # 3D
    elif mesh.dim == 3:
        if (len(alpha) != 6 or len(beta) != 6 or len(gamma) != 6):
            raise Exception("Lenght of list, alpha should be 6")
        fxm, fxp, fym, fyp, fzm, fzp = getfaceBoundaryInd(mesh, activeCC)

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]
        alpha_zm, beta_zm, gamma_zm = alpha[4], beta[4], gamma[4]
        alpha_zp, beta_zp, gamma_zp = alpha[5], beta[5], gamma[5]

        # Compute size of boundary cells for in and out faces
        h_xm = mesh.gridFx[np.arange(mesh.nFx)[fxm]+1, 0] - mesh.gridFx[np.arange(mesh.nFx)[fxm], 0]
        h_xp = mesh.gridFx[np.arange(mesh.nFx)[fxp], 0] - mesh.gridFx[np.arange(mesh.nFx)[fxp]-1, 0]
        h_ym = mesh.gridFy[np.arange(mesh.nFy)[fym]+mesh.vnFx[0]-1, 1] - mesh.gridFy[np.arange(mesh.nFy)[fym], 1]
        h_yp = mesh.gridFy[np.arange(mesh.nFy)[fyp], 1] - mesh.gridFy[np.arange(mesh.nFy)[fyp]-mesh.vnFx[0]+1, 1]
        h_zm = mesh.gridFz[np.arange(mesh.nFz)[fzm]+mesh.vnC[0]*mesh.vnC[1], 2] - mesh.gridFz[np.arange(mesh.nFz)[fzm], 2]
        h_zp = mesh.gridFz[np.arange(mesh.nFz)[fzp], 2] - mesh.gridFz[np.arange(mesh.nFz)[fzp]-mesh.vnC[0]*mesh.vnC[1], 2]

        a_xm = gamma_xm/(0.5*alpha_xm-beta_xm/h_xm)
        b_xm = (0.5*alpha_xm+beta_xm/h_xm)/(0.5*alpha_xm-beta_xm/h_xm)
        a_xp = gamma_xp/(0.5*alpha_xp-beta_xp/h_xp)
        b_xp = (0.5*alpha_xp+beta_xp/h_xp)/(0.5*alpha_xp-beta_xp/h_xp)

        a_ym = gamma_ym/(0.5*alpha_ym-beta_ym/h_ym)
        b_ym = (0.5*alpha_ym+beta_ym/h_ym)/(0.5*alpha_ym-beta_ym/h_ym)
        a_yp = gamma_yp/(0.5*alpha_yp-beta_yp/h_yp)
        b_yp = (0.5*alpha_yp+beta_yp/h_yp)/(0.5*alpha_yp-beta_yp/h_yp)

        a_zm = gamma_zm/(0.5*alpha_zm-beta_zm/h_zm)
        b_zm = (0.5*alpha_zm+beta_zm/h_zm)/(0.5*alpha_zm-beta_zm/h_zm)
        a_zp = gamma_zp/(0.5*alpha_zp-beta_zp/h_zp)
        b_zp = (0.5*alpha_zp+beta_zp/h_zp)/(0.5*alpha_zp-beta_zp/h_zp)

        xBC_xm = 0.5*a_xm
        xBC_xp = 0.5*a_xp/b_xp
        yBC_xm = 0.5*(1.-b_xm)
        yBC_xp = 0.5*(1.-1./b_xp)
        xBC_ym = 0.5*a_ym
        xBC_yp = 0.5*a_yp/b_yp
        yBC_ym = 0.5*(1.-b_ym)
        yBC_yp = 0.5*(1.-1./b_yp)
        xBC_zm = 0.5*a_zm
        xBC_zp = 0.5*a_zp/b_zp
        yBC_zm = 0.5*(1.-b_zm)
        yBC_zp = 0.5*(1.-1./b_zp)

        xBC_x = np.zeros(mesh.nFx)
        xBC_y = np.zeros(mesh.nFy)
        xBC_z = np.zeros(mesh.nFz)

        yBC_x = np.zeros(mesh.nFx)
        yBC_y = np.zeros(mesh.nFy)
        yBC_z = np.zeros(mesh.nFy)

        xBC_x[fxm] = xBC_xm
        xBC_x[fxp] = xBC_xp
        yBC_x[fxm] = yBC_xm
        yBC_x[fxp] = yBC_xp
        xBC_y[fym] = xBC_ym
        xBC_y[fyp] = xBC_yp
        yBC_y[fym] = yBC_ym
        yBC_y[fyp] = yBC_yp
        xBC_z[fzm] = xBC_zm
        xBC_z[fzp] = xBC_zp
        yBC_z[fzm] = yBC_zm
        yBC_z[fzp] = yBC_zp

        fx = np.logical_or(fxm, fxp)
        fy = np.logical_or(fym, fyp)
        fz = np.logical_or(fzm, fzp)

        xBC_x = xBC_x[fx]
        yBC_x = yBC_x[fx]
        xBC_y = xBC_y[fy]
        yBC_y = yBC_y[fy]
        xBC_z = xBC_z[fz]
        yBC_z = yBC_z[fz]

        xBC = np.r_[xBC_x, xBC_y, xBC_z]
        yBC = np.r_[yBC_x, yBC_y, yBC_z]

    return xBC, yBC


def setupBC(mesh, activeCC):
    # Compute active face
    tempface = mesh.aveF2CC.T*activeCC
    activeface = np.logical_or(tempface == 1./mesh.dim, tempface == 1./mesh.dim*0.5)
    # Compute boundary indices
    out = mesh.faceDiv.T*activeCC.astype(float)
    indsn = out < 0.
    indsp = out > 0.
    inds = indsn + indsp
    if mesh.dim == 2:
        indsnx = indsn[:mesh.nFx]
        indsny = indsn[mesh.nFx:mesh.nFx+mesh.nFy]
        indspx = indsp[:mesh.nFx]
        indspy = indsp[mesh.nFx:mesh.nFx+mesh.nFy]
    if mesh.dim == 3:
        indsnx = indsn[:mesh.nFx]
        indsny = indsn[mesh.nFx:mesh.nFx+mesh.nFy]
        indsnz = indsn[mesh.nFx+mesh.nFy:mesh.nFx+mesh.nFy+mesh.nFz]
        indspx = indsp[:mesh.nFx]
        indspy = indsp[mesh.nFx:mesh.nFx+mesh.nFy]
        indspz = indsp[mesh.nFx+mesh.nFy:mesh.nFx+mesh.nFy+mesh.nFz]

    actCCinds = np.argwhere(activeCC).flatten()
    actfaceinds = np.argwhere(activeface).flatten()

    I = np.arange(actCCinds.size)
    J = actCCinds.copy()
    Pcc = sp.coo_matrix((np.ones_like(actCCinds), (I, J)), shape=(actCCinds.size, activeCC.size))
    Pcc = Pcc.tocsr()
    I = np.arange(actfaceinds.size)
    J = actfaceinds.copy()
    Pface = sp.coo_matrix((np.ones_like(actfaceinds), (I, J)), shape=(actfaceinds.size, activeface.size))
    Pface = Pface.tocsr()

    nBC = inds.sum()
    J = np.arange(mesh.nF)[inds]
    I = np.arange(J.size)
    B = sp.coo_matrix((np.ones_like(J), (I, J)), shape=(nBC, mesh.nF))
    B = B.tocsr() * Pface.T

    if mesh.dim == 2:
        vecx = np.zeros(mesh.nFx)
        vecy = np.zeros(mesh.nFy)
        vecx[indsnx] = -1.
        vecx[indspx] = 1.
        vecy[indsny] = -1.
        vecy[indspy] = 1.
        vec = np.r_[vecx, vecy][inds]

    elif mesh.dim == 3:
        vecx = np.zeros(mesh.nFx)
        vecy = np.zeros(mesh.nFy)
        vecz = np.zeros(mesh.nFz)
        vecx[indsnx] = -1.
        vecx[indspx] = 1.
        vecy[indsny] = -1.
        vecy[indspy] = 1.
        vecz[indsnz] = -1.
        vecz[indspz] = 1.
        vec = np.r_[vecx, vecy, vecz][inds]

    Pbc = sp.coo_matrix((vec, (J, I)), shape=(mesh.nF, nBC))
    Pbc = Pface * Pbc.tocsr() * Utils.sdiag(mesh.area[inds])

    if mesh.dim == 2:
        fact = 4
    elif mesh.dim == 3:
        fact = 6

    aveCC2Fact = Pface*mesh.aveF2CC.T*Pcc.T * fact
    M = B*aveCC2Fact
    Dact = Pcc*mesh.faceDiv*Pface.T
    return Dact, Pbc, M, Pface, Pcc


class Test2D_InhomogeneousMixed(Tests.OrderTest):
    name = "2D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        def phi_fun(x):
            return np.cos(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])

        def j_funX(x):
            return +np.pi*np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])

        def j_funY(x):
            return +np.pi*np.cos(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])

        def phideriv_funX(x):
            return -j_funX(x)

        def phideriv_funY(x):
            return -j_funY(x)

        def q_fun(x):
            return +2*(np.pi**2)*phi_fun(x)

        x = self.M.vectorNx
        y = -2.*abs(x-0.5) + 1.
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        activeCC = Utils.ModelBuilder.PolygonInd(self.M, np.c_[x, y])
        Dact, Pbc, M, Pface, Pcc = setupBC(self.M, activeCC)

        xc_ana = phi_fun(self.M.gridCC[activeCC, :])
        q_ana = q_fun(self.M.gridCC[activeCC, :])

        # Get boundary locations
        indsnx, indspx, indsny, indspy = getfaceBoundaryInd(self.M, activeCC)
        gBFxm = self.M.gridFx[indsnx, :]
        gBFxp = self.M.gridFx[indspx, :]
        gBFym = self.M.gridFy[indsny, :]
        gBFyp = self.M.gridFy[indspy, :]

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha_xm = np.ones_like(gBFxm[:, 0])
        alpha_xp = np.ones_like(gBFxp[:, 0])
        beta_xm = np.ones_like(gBFxm[:, 0])
        beta_xp = np.ones_like(gBFxp[:, 0])
        alpha_ym = np.ones_like(gBFym[:, 1])
        alpha_yp = np.ones_like(gBFyp[:, 1])
        beta_ym = np.ones_like(gBFym[:, 1])
        beta_yp = np.ones_like(gBFyp[:, 1])

        phi_bc_xm, phi_bc_xp = phi_fun(gBFxm), phi_fun(gBFxp)
        phi_bc_ym, phi_bc_yp = phi_fun(gBFym), phi_fun(gBFyp)

        phiderivX_bc_xm = phideriv_funX(gBFxm)
        phiderivX_bc_xp = phideriv_funX(gBFxp)
        phiderivY_bc_ym = phideriv_funY(gBFym)
        phiderivY_bc_yp = phideriv_funY(gBFyp)

        def gamma_fun(alpha, beta, phi, phi_deriv):
            return alpha*phi + beta*phi_deriv

        gamma_xm = gamma_fun(alpha_xm, beta_xm, phi_bc_xm, phiderivX_bc_xm)
        gamma_xp = gamma_fun(alpha_xp, beta_xp, phi_bc_xp, phiderivX_bc_xp)
        gamma_ym = gamma_fun(alpha_ym, beta_ym, phi_bc_ym, phiderivY_bc_ym)
        gamma_yp = gamma_fun(alpha_yp, beta_yp, phi_bc_yp, phiderivY_bc_yp)

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

        x_BC, y_BC = getxBCyBC_CC(self.M, activeCC, alpha, beta, gamma)

        K = np.ones(self.M.nC)
        Kact = K[activeCC]
        volact = self.M.vol[activeCC]
        aveF2CCFact = Pcc*self.M.aveF2CC*Pface.T
        MfKiIact = Utils.sdiag(1./(aveF2CCFact.T*(volact*Kact)*self.M.dim))
        V = Utils.sdiag(volact)
        Div = V*Dact
        q = q_fun(self.M.gridCC[activeCC, :])
        G = Div.T - Pbc*Utils.sdiag(y_BC)*M
        A = Div*MfKiIact*G
        rhs = V*q + Div*MfKiIact*Pbc*x_BC

        if self.myTest == 'xc':
            # Ainv = Solver(A)
            Ainv = BicgJacobiSolver(A)
            xc = Ainv*rhs
            err = np.linalg.norm((xc-xc_ana), np.inf)
        else:
            NotImplementedError
        return err

    def test_order(self):
        print("==== Testing Mixed boudary condition (insdide) for CC-problem ====")
        self.name = "2D"
        self.myTest = 'xc'
        self.orderTest()


class Test3D_InhomogeneousMixed(Tests.OrderTest):
    name = "3D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 3
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        def phi_fun(x):
            return (np.cos(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1]) *
                    np.cos(np.pi*x[:, 2]))

        def j_funX(x):
            return (np.pi*np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1]) *
                    np.cos(np.pi*x[:, 2]))

        def j_funY(x):
            return (np.pi*np.cos(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1]) *
                    np.cos(np.pi*x[:, 2]))

        def j_funZ(x):
            return (np.pi*np.cos(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1]) *
                    np.sin(np.pi*x[:, 2]))

        def phideriv_funX(x): return -j_funX(x)

        def phideriv_funY(x): return -j_funY(x)

        def phideriv_funZ(x): return -j_funZ(x)

        def q_fun(x): return 3*(np.pi**2)*phi_fun(x)

        x = np.r_[0, 1, 1, 0, 0, 0.5]
        y = np.r_[0, 0, 1, 1, 0, 0.5]
        z = np.r_[0, 0, 0, 0, 0, 1.]

        activeCC = Utils.ModelBuilder.PolygonInd(self.M, np.c_[x, y, z])
        Dact, Pbc, M, Pface, Pcc = setupBC(self.M, activeCC)

        xc_ana = phi_fun(self.M.gridCC[activeCC, :])
        q_ana = q_fun(self.M.gridCC[activeCC, :])

        # Get boundary locations
        fxm, fxp, fym, fyp, fzm, fzp = getfaceBoundaryInd(self.M, activeCC)
        gBFxm = self.M.gridFx[fxm, :]
        gBFxp = self.M.gridFx[fxp, :]
        gBFym = self.M.gridFy[fym, :]
        gBFyp = self.M.gridFy[fyp, :]
        gBFzm = self.M.gridFz[fzm, :]
        gBFzp = self.M.gridFz[fzp, :]

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha_xm = np.ones_like(gBFxm[:, 0])
        alpha_xp = np.ones_like(gBFxp[:, 0])
        beta_xm = np.ones_like(gBFxm[:, 0])
        beta_xp = np.ones_like(gBFxp[:, 0])
        alpha_ym = np.ones_like(gBFym[:, 1])
        alpha_yp = np.ones_like(gBFyp[:, 1])
        beta_ym = np.ones_like(gBFym[:, 1])
        beta_yp = np.ones_like(gBFyp[:, 1])
        alpha_zm = np.ones_like(gBFzm[:, 2])
        alpha_zp = np.ones_like(gBFzp[:, 2])
        beta_zm = np.ones_like(gBFzm[:, 2])
        beta_zp = np.ones_like(gBFzp[:, 2])

        phi_bc_xm, phi_bc_xp = phi_fun(gBFxm), phi_fun(gBFxp)
        phi_bc_ym, phi_bc_yp = phi_fun(gBFym), phi_fun(gBFyp)
        phi_bc_zm, phi_bc_zp = phi_fun(gBFzm), phi_fun(gBFzp)

        phiderivX_bc_xm = phideriv_funX(gBFxm)
        phiderivX_bc_xp = phideriv_funX(gBFxp)
        phiderivY_bc_ym = phideriv_funY(gBFym)
        phiderivY_bc_yp = phideriv_funY(gBFyp)
        phiderivY_bc_zm = phideriv_funZ(gBFzm)
        phiderivY_bc_zp = phideriv_funZ(gBFzp)

        def gamma_fun(alpha, beta, phi, phi_deriv):
            return alpha*phi + beta*phi_deriv

        gamma_xm = gamma_fun(alpha_xm, beta_xm, phi_bc_xm, phiderivX_bc_xm)
        gamma_xp = gamma_fun(alpha_xp, beta_xp, phi_bc_xp, phiderivX_bc_xp)
        gamma_ym = gamma_fun(alpha_ym, beta_ym, phi_bc_ym, phiderivY_bc_ym)
        gamma_yp = gamma_fun(alpha_yp, beta_yp, phi_bc_yp, phiderivY_bc_yp)
        gamma_zm = gamma_fun(alpha_zm, beta_zm, phi_bc_zm, phiderivY_bc_zm)
        gamma_zp = gamma_fun(alpha_zp, beta_zp, phi_bc_zp, phiderivY_bc_zp)

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp, alpha_zm, alpha_zp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm, gamma_zp]

        x_BC, y_BC = getxBCyBC_CC(self.M, activeCC, alpha, beta, gamma)

        K = np.ones(self.M.nC)
        Kact = K[activeCC]
        volact = self.M.vol[activeCC]
        aveF2CCFact = Pcc*self.M.aveF2CC*Pface.T
        MfKiIact = Utils.sdiag(1./(aveF2CCFact.T*(volact*Kact)*self.M.dim))
        V = Utils.sdiag(volact)
        Div = V*Dact
        q = q_fun(self.M.gridCC[activeCC, :])
        G = Div.T - Pbc*Utils.sdiag(y_BC)*M
        A = Div*MfKiIact*G
        rhs = V*q + Div*MfKiIact*Pbc*x_BC

        if self.myTest == 'xc':
            # Ainv = Solver(A)
            Ainv = BicgJacobiSolver(A)
            xc = Ainv*rhs
            err = np.linalg.norm((xc-xc_ana), np.inf)
        else:
            NotImplementedError
        return err

    def test_order(self):
        print("==== Testing Mixed boudary condition (inside) for CC-problem ====")
        self.name = "3D"
        self.myTest = 'xc'
        self.orderTest()

if __name__ == '__main__':
    unittest.main()
