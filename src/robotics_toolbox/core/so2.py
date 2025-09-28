#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 2D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO2:
    """This class represents an SO2 rotations internally represented by rotation
    matrix."""

    def __init__(self, angle: float = 0.0) -> None:
        """Creates a rotation transformation that rotates vector by a given angle, that
        is expressed in radians. Rotation matrix .rot is used internally, no other
        variables can be stored inside the class."""
        super().__init__()

        # todo HW01: implement computation of rotation matrix from the given
        # angle
        self.rot: np.ndarray = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def __mul__(self, other: SO2) -> SO2:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotations.
        result = SO2()
        result.rot = self.rot @ other.rot
        return result

    @property
    def angle(self) -> float:
        """Return angle [rad] from the internal rotation matrix representation."""
        # todo: HW01: implement computation of the angle from rotation matrix
        # self.rot.
        angle = np.arctan2(self.rot[1, 0], self.rot[0, 0])
        return angle

    def inverse(self) -> SO2:
        """Return inverse of the transformation. Do not change internal property of the
        object."""
        # todo: HW01: implement inverse, do not use np.linalg.inverse()
        result = SO2()
        # for SO2 inverse is always a transpose
        result.rot = self.rot.T
        return result

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (2,)
        return self.rot @ v

    def __eq__(self, other: SO2) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    def __hash__(self):
        return id(self)
