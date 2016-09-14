from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
from SimPEG import Utils

# TODO:
# r = lambda dx, dy, dz: np.sqrt( dx**2. + dy**2. + dz**2.)


def ElectricDipoleWholeSpace_E(XYZ, srcLoc, sig, t, current=1., length=1., orientation='X', kappa=0., epsr=1.):

    """
        Computing the analytic electric fields (E) from an electrical dipole in a wholespace
        - You have the option of computing E for multiple times at a single reciever location
          or a single time at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate E
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array t: array of times at which to measure
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :rtype: numpy.array
        :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and times.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & t.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    theta = np.sqrt((mu*sig)/(4*t))

    front = current * length / (4.* pi * sig * r**3)
    mid   = 3 * erf(theta*r) - (4/np.srqt(pi) * (theta)**3 * r**3 + 6/np.srqt(pi) * theta * r) * np.exp(-(theta)**2 * (r)**2)
    extra = (erf(theta*r) - (4/np.srqt(pi) * (theta)**3 * r**3 + 2/np.srqt(pi) * theta * r) * np.exp(-(theta)**2 * (r)**2))

    if orientation.upper() == 'X':
        Ex = front*(dx**2 / r**2)*mid - extra
        Ey = front*(dx*dy  / r**2)*mid
        Ez = front*(dx*dz  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey = front*(dy**2 / r**2)*mid - extra
        Ez = front*(dy*dz  / r**2)*mid
        Ex = front*(dy*dx  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez = front*(dz**2 / r**2)*mid - extra
        Ex = front*(dz*dx  / r**2)*mid
        Ey = front*(dz*dy  / r**2)*mid
        return Ex, Ey, Ez



def ElectricDipoleWholeSpace_J(XYZ, srcLoc, sig, t, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing the analytic current density (J) from an electrical dipole in a wholespace
        - You have the option of computing J for multiple times at a single reciever location
          or a single time at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate J
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array t: array of times at which to measure
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :rtype: numpy.array
        :return: Jx, Jy, Jz: arrays containing all 3 components of J evaluated at the specified locations and times.
    """

    Ex, Ey, Ez = ElectricDipoleWholeSpace_E(XYZ, srcLoc, sig, t, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx = sig*Ex
    Jy = sig*Ey
    Jz = sig*Ez
    return Jx, Jy, Jz


def ElectricDipoleWholeSpace_H(XYZ, srcLoc, sig, t, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing the analytic magnetic fields (H) from an electrical dipole in a wholespace
        - You have the option of computing H for multiple times at a single reciever location
          or a single time at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate H
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array t: array of times at which to measure
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :rtype: numpy.array
        :return: Hx, Hy, Hz: arrays containing all 3 components of H evaluated at the specified locations and times.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & t.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

    dx = XYZ[:, 0]-srcLoc[0]
    dy = XYZ[:, 1]-srcLoc[1]
    dz = XYZ[:, 2]-srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    theta = np.sqrt((mu*sig)/(4*t))

    front = (current * length) / (4.*pi*(r)**3)
    mid = erf(theta*r) - (2/np.sqrt(pi)) * theta * r * np.exp(-(theta)**2 * (r)**2)
    if orientation.upper() == 'X':
        Hy = front * mid * -dz
        Hz = front * mid * dy
        Hx = np.zeros_like(Hy)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Y':
        Hx = front * mid * dz
        Hz = front * mid * -dx
        Hy = np.zeros_like(Hx)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Z':
        Hx = front * mid * -dy
        Hy = front * mid * dx
        Hz = np.zeros_like(Hx)
        return Hx, Hy, Hz


def ElectricDipoleWholeSpace_B(XYZ, srcLoc, sig, t, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing the analytic magnetic flux density (B) from an electrical dipole in a wholespace
        - You have the option of computing B for multiple times at a single reciever location
          or a single time at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate B
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array t: array of times at which to measure
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :rtype: numpy.array
        :return: Bx, By, Bz: arrays containing all 3 components of B evaluated at the specified locations and times.
    """

    mu = mu_0*(1+kappa)

    Hx, Hy, Hz = ElectricDipoleWholeSpace_H(XYZ, srcLoc, sig, t, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Bx = mu*Hx
    By = mu*Hy
    Bz = mu*Hz
    return Bx, By, Bz


# def ElectricDipoleWholeSpace_A(XYZ, srcLoc, sig, t, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

#     """
#         Computing the analytic electric vector potential (A) from an electrical dipole in a wholespace
#         - You have the option of computing A for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate A
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Ax, Ay, Az: arrays containing all 3 components of A evaluated at the specified locations and times.
#     """
#     mu = mu_0*(1+kappa)
#     epsilon = epsilon_0*epsr
#     XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
#     # Check
#     if XYZ.shape[0] > 1 & t.shape[0] > 1:
#         raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

#     dx = XYZ[:,0]-srcLoc[0]
#     dy = XYZ[:,1]-srcLoc[1]
#     dz = XYZ[:,2]-srcLoc[2]

#     r  = np.sqrt( dx**2. + dy**2. + dz**2.)
#     k  = np.sqrt( omega(t)**2. *mu*epsilon -1j*omega(t)*mu*sig )

#     front = current * length / (4.*pi*r)

#     if orientation.upper() == 'X':
#         Ax = front*np.exp(-1j*k*r)
#         Ay = np.zeros_like(Ax)
#         Az = np.zeros_like(Ax)
#         return Ax, Ay, Az

#     elif orientation.upper() == 'Y':
#         Ay = front*np.exp(-1j*k*r)
#         Ax = np.zeros_like(Ay)
#         Az = np.zeros_like(Ay)
#         return Ax, Ay, Az

#     elif orientation.upper() == 'Z':
#         Az = front*np.exp(-1j*k*r)
#         Ax = np.zeros_like(Ay)
#         Ay = np.zeros_like(Ay)
#         return Ax, Ay, Az


# def MagneticDipoleWholeSpace_E(XYZ, srcLoc, sig, t, current=1., loopArea=1., orientation='X', kappa=0., epsr=1., t=0.):

#     """
#         Computing the analytic electric fields (E) from a magnetic dipole in a wholespace
#         - You have the option of computing E for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate E
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and times.
#     """

#     mu = mu_0 * (1+kappa)
#     epsilon = epsilon_0 * epsr
#     m = current * loopArea

#     XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
#     # Check
#     if XYZ.shape[0] > 1 & t.shape[0] > 1:
#         raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

#     dx = XYZ[:,0]-srcLoc[0]
#     dy = XYZ[:,1]-srcLoc[1]
#     dz = XYZ[:,2]-srcLoc[2]

#     r  = np.sqrt( dx**2. + dy**2. + dz**2.)
#     # k  = np.sqrt( -1j*2.*pi*t*mu*sig )
#     k  = np.sqrt( omega(t)**2. *mu*epsilon -1j*omega(t)*mu*sig )

#     front = ((1j * omega(t) * mu * m) / (4.* pi * r**2)) * (1j * k * r + 1) * np.exp(-1j*k*r)

#     if orientation.upper() == 'X':
#         Ey = front * (dz / r)
#         Ez = front * (-dy / r)
#         Ex = np.zeros_like(Ey)
#         return Ex, Ey, Ez

#     elif orientation.upper() == 'Y':
#         Ex = front * (-dz / r)
#         Ez = front * (dx / r)
#         Ey = np.zeros_like(Ex)
#         return Ex, Ey, Ez

#     elif orientation.upper() == 'Z':
#         Ex = front * (dy / r)
#         Ey = front * (-dx / r)
#         Ez = np.zeros_like(Ex)
#         return Ex, Ey, Ez


# def MagneticDipoleWholeSpace_J(XYZ, srcLoc, sig, t, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

#     """
#         Computing the analytic current density (J) from a magnetic dipole in a wholespace
#         - You have the option of computing J for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate J
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Jx, Jy, Jz: arrays containing all 3 components of J evaluated at the specified locations and times.
#     """

#     Ex, Ey, Ez = MagneticDipoleWholeSpace_E(XYZ, srcLoc, sig, t, current=current, loopArea=loopArea, orientation=orientation, kappa=kappa, epsr=epsr)
#     Jx = sig * Ex
#     Jy = sig * Ey
#     Jz = sig * Ez
#     return Jx, Jy, Jz


# def MagneticDipoleWholeSpace_H(XYZ, srcLoc, sig, t, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

#     """
#         Computing the analytic magnetic fields (H) from a magnetic dipole in a wholespace
#         - You have the option of computing H for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate H
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Hx, Hy, Hz: arrays containing all 3 components of H evaluated at the specified locations and times.
#     """

#     mu = mu_0 * (1+kappa)
#     epsilon = epsilon_0 * epsr
#     m = current * loopArea

#     XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
#     # Check
#     if XYZ.shape[0] > 1 & t.shape[0] > 1:
#         raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

#     dx = XYZ[:, 0]-srcLoc[0]
#     dy = XYZ[:, 1]-srcLoc[1]
#     dz = XYZ[:, 2]-srcLoc[2]

#     r = np.sqrt(dx**2. + dy**2. + dz**2.)
#     # k  = np.sqrt( -1j*2.*pi*t*mu*sig )
#     k = np.sqrt(omega(t)**2. * mu*epsilon - 1j*omega(t)*mu*sig)

#     front = m / (4.*pi*(r)**3) * np.exp(-1j*k*r)
#     mid   = -k**2 * r**2 + 3*1j*k*r + 3

#     if orientation.upper() == 'X':
#         Hx = front*((dx**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
#         Hy = front*(dx*dy  / r**2)*mid
#         Hz = front*(dx*dz  / r**2)*mid
#         return Hx, Hy, Hz

#     elif orientation.upper() == 'Y':
#         #  x--> y, y--> z, z-->x
#         Hy = front * ((dy**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
#         Hz = front * (dy*dz  / r**2)*mid
#         Hx = front * (dy*dx  / r**2)*mid
#         return Hx, Hy, Hz

#     elif orientation.upper() == 'Z':
#         # x --> z, y --> x, z --> y
#         Hz = front*((dz**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
#         Hx = front*(dz*dx  / r**2)*mid
#         Hy = front*(dz*dy  / r**2)*mid
#         return Hx, Hy, Hz


# def MagneticDipoleWholeSpace_B(XYZ, srcLoc, sig, t, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

#     """
#         Computing the analytic magnetic flux density (B) from a magnetic dipole in a wholespace
#         - You have the option of computing B for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate B
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Bx, By, Bz: arrays containing all 3 components of B evaluated at the specified locations and times.
#     """

#     mu = mu_0 * (1+kappa)

#     Hx, Hy, Hz = MagneticDipoleWholeSpace_H(XYZ, srcLoc, sig, t, current=current, loopArea=loopArea, orientation=orientation, kappa=kappa, epsr=epsr)
#     Bx = mu * Hx
#     By = mu * Hy
#     Bz = mu * Hz
#     return Bx, By, Bz


# def MagneticDipoleWholeSpace_F(XYZ, srcLoc, sig, t, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

#     """
#         Computing the analytic magnetic vector potential (t) from a magnetic dipole in a wholespace
#         - You have the option of computing t for multiple times at a single reciever location
#           or a single time at multiple locations

#         :param numpy.array XYZ: reciever locations at which to evaluate t
#         :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
#         :param float sig: value specifying the conductivity (S/m) of the wholespace
#         :param numpy.array t: array of times at which to measure
#         :param float current: size of the injected current (A), default is 1.0 A
#         :param float length: length of the dipole (m), default is 1.0 m
#         :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
#         :param float kappa: magnetic susceptiblity value (unitless), default is 0.
#         :param float epsr: relative permitivitty value (unitless),  default is 1.0
#         :rtype: numpy.array
#         :return: Fx, Fy, Fz: arrays containing all 3 components of t evaluated at the specified locations and times.
#     """

#     mu = mu_0 * (1+kappa)
#     epsilon = epsilon_0*epsr
#     m = current * loopArea

#     XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
#     # Check
#     if XYZ.shape[0] > 1 & t.shape[0] > 1:
#         raise Exception("I/O type error: For multiple field locations only a single time can be specified.")

#     dx = XYZ[:,0]-srcLoc[0]
#     dy = XYZ[:,1]-srcLoc[1]
#     dz = XYZ[:,2]-srcLoc[2]

#     r  = np.sqrt( dx**2. + dy**2. + dz**2.)
#     k  = np.sqrt( omega(t)**2. *mu*epsilon -1j*omega(t)*mu*sig )

#     front = (1j * omega(t) * mu * m) / (4.* pi * r)

#     if orientation.upper() == 'X':
#         Fx = front*np.exp(-1j*k*r)
#         Fy = np.zeros_like(Fx)
#         Fz = np.zeros_like(Fx)
#         return Fx, Fy, Fz

#     elif orientation.upper() == 'Y':
#         Fy = front*np.exp(-1j*k*r)
#         Fx = np.zeros_like(Fy)
#         Fz = np.zeros_like(Fy)
#         return Fx, Fy, Fz

#     elif orientation.upper() == 'Z':
#         Fz = front*np.exp(-1j*k*r)
#         Fx = np.zeros_like(Fy)
#         Fy = np.zeros_like(Fy)
#         return Fx, Fy, Fz



