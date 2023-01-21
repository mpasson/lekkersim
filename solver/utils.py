#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:48:24 2021

@author: marco
"""
import functools

import numpy as np
import types


class ProtectedPartial(functools.partial):
    """Like partial, but keywords provided at creation cannot be overwritten al call time"""

    def __call__(self, /, *args, **keywords):
        keywords = {**keywords, **self.keywords}
        return self.func(*self.args, *args, **keywords)


def map_args(func, varsmap={}):
    """This function maps the argument names from a funciton to a new one

    Example: usnign this gunction on f(y, x=2) with mapping {'x' : 'a', 'y' : 'b'} will return f(a, b=2)
    Order of arguments and default values will be preserved

    Args:
        func (function):    Function which arguments need to be mapped
        varsmap (dict):     Dictionary of the mapping {'oldname' : 'newname'}

    Returns:
        function:   New function with the updated argument names
    """
    y = func
    oldargs = y.__code__.co_varnames[: y.__code__.co_argcount]
    new_args = []
    for x in oldargs:
        if x in varsmap:
            new_args.append(varsmap[x])
        else:
            new_args.append(x)
    new_args = tuple(new_args) + y.__code__.co_varnames[y.__code__.co_argcount :]
    print(new_args)
    y_code = types.CodeType(
        y.__code__.co_argcount,
        y.__code__.co_kwonlyargcount,
        y.__code__.co_nlocals,
        y.__code__.co_stacksize,
        y.__code__.co_flags,
        y.__code__.co_code,
        y.__code__.co_consts,
        y.__code__.co_names,  # y.__code__.co_varnames, \
        new_args,
        y.__code__.co_filename,
        y.__code__.co_name,
        y.__code__.co_firstlineno,
        y.__code__.co_lnotab,
    )
    return types.FunctionType(y_code, y.__globals__, y.__name__, y.__defaults__)


def line(N):
    """It returns N-values spaced by 1 and centered around 0"""
    return np.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)


class GaussianBeam:
    def __init__(
        self,
        w0,
        n,
        wl,
        z0=0.0,
        x0=0.0,
        theta=0.0,
    ):
        """Gaussian beam for propagation in 2D slabs.

        The beam is launched at (z0, x0) at an angle theta with respect to the z-axis and is characterized by
        a beam waist w0. The medium is assumed to be uniform in x and z, so that the power is constant among two
        xy planes.

        Args:
            w0 (float): Beam waist in [um].
            n (float): Refractive index of the medium.
            wl (float): Wavelength in [um].
            z0 (float): Initial z position of the beam.
            x0 (float): Initial x position of the beam.
            theta (float): Launch angle in [deg] with respect to the x-axis.

        """
        self.wl = wl
        self.w0 = w0
        self.n = n
        self.z_r = np.pi * w0 * w0 * n / wl  # Rayleigh range
        self.z0 = z0
        self.x0 = x0
        self.theta = np.deg2rad(-theta)  # Convert to radians here

    def rotate(self, z, x):
        """Creates a 2D rotation matrix based on the angle theta.

        Args:
            z (list | np.array): The z position vector (horizontal).
            x (list | np.array): The x position vector (vertical).

        Returns:
            np.array, np.array: The rotated vectors
        """
        z = z - self.z0
        x = x - self.x0
        _z = np.cos(self.theta) * z - np.sin(self.theta) * x
        _x = np.sin(self.theta) * z + np.cos(self.theta) * x
        return _z, _x

    def waist(self, z):
        """Beam waist.

        Args:
            z (float | list | numpy.array): Position in [um].

        Returns:
            float | list | numpy.array: The beam waist at the position z.
        """
        return self.w0 * np.sqrt(1 + (z / self.z_r) ** 2)

    def curvature(self, z):
        """Curvature radius.

        Args:
            z (float | list | numpy.array): Position in [um].

        Returns:
            float | list | numpy.array: The curvature at the position z.
        """
        return z * (1 + (self.z_r / z) ** 2)

    def gouy(self, z):
        """Gouy phase.

        Args:
            z (float | list | numpy.array): Position in [um].

        Returns:
            float | list | numpy.array: The Gouy phase at the position z.
        """
        return np.arctan(z / self.z_r)

    def field(self, z, x):
        """Complex electric field.

        Args:
            z (float | list | numpy.array): z position in [um].
            x (float): x position in [um].

        Returns:
            np.ndarray: x, z, field
        """
        _z, _x = self.rotate(z, x)

        k_z = 2 * np.pi * self.n / self.wl
        k_x = 2 * np.pi * self.n / self.wl
        _waist = self.waist(_z)

        _amplitude = np.sqrt(np.sqrt(2 / np.pi) / _waist) * np.exp(
            -_x * _x / (_waist * _waist)
        )
        _phase = np.exp(
            -1j * (k_z * _z + k_x * _x * _x / (2 * self.curvature(_z)) - self.gouy(_z))
        )

        _field = _amplitude * _phase

        return _field
