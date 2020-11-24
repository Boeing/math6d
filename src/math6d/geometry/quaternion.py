import copy
import math

import enum
import genpy
import numpy
import typing
from geometry_msgs.msg import Quaternion as RosQuaternion

from math6d.geometry.msg import MsgConversion
from math6d.geometry.vector3 import Vector3


class Axis(enum.Enum):
    AXIS_X = Vector3(1, 0, 0)  # type: Axis
    AXIS_Y = Vector3(0, 1, 0)  # type: Axis
    AXIS_Z = Vector3(0, 0, 1)  # type: Axis


class Quaternion(MsgConversion):
    X_KW = 'x'
    Y_KW = 'y'
    Z_KW = 'z'
    W_KW = 'w'

    def __init__(self, *args, **kwargs):

        # Coefficients are stored (w, x, y, z,)
        self.__coeffs = numpy.array([1, 0, 0, 0], dtype=numpy.float64)

        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 1:
            _arg = args[0]
            if isinstance(_arg, genpy.Message):
                qt = Quaternion.from_msg(_arg)
                self.__coeffs[:] = qt.coeffs()
            elif isinstance(_arg, numpy.ndarray) and _arg.size == 4:
                self.__coeffs[:] = _arg
            elif isinstance(_arg, numpy.matrix) and _arg.shape == (3, 3):
                qt = Quaternion.from_matrix(_arg)
                self.__coeffs[:] = qt.coeffs()
            elif isinstance(_arg, Vector3):
                # rotation vector
                qt = Quaternion.from_axis_angle(axis=_arg.normalized(), angle=_arg.norm())
                self.__coeffs[:] = qt.coeffs()
            else:
                raise NotImplementedError()
        elif len(args) == 4:
            self.__coeffs[0] = args[0]
            self.__coeffs[1] = args[1]
            self.__coeffs[2] = args[2]
            self.__coeffs[3] = args[3]
        elif len(args) + len(kwargs) == 4:
            self.__coeffs[0] = kwargs[self.W_KW] if self.W_KW in kwargs else 0.0
            self.__coeffs[1] = kwargs[self.X_KW] if self.X_KW in kwargs else 0.0
            self.__coeffs[2] = kwargs[self.Y_KW] if self.Y_KW in kwargs else 0.0
            self.__coeffs[3] = kwargs[self.Z_KW] if self.Z_KW in kwargs else 0.0
        else:
            raise NotImplementedError()

    #
    # Properties
    #

    @property
    def x(self):
        # type: () -> float
        return self.__coeffs[1]

    @property
    def y(self):
        # type: () -> float
        return self.__coeffs[2]

    @property
    def z(self):
        # type: () -> float
        return self.__coeffs[3]

    @property
    def w(self):
        # type: () -> float
        return self.__coeffs[0]

    @x.setter
    def x(self, x):
        self.__coeffs[1] = x

    @y.setter
    def y(self, y):
        self.__coeffs[2] = y

    @z.setter
    def z(self, z):
        self.__coeffs[3] = z

    @w.setter
    def w(self, w):
        self.__coeffs[0] = w

    def vec(self):
        # type: () -> numpy.ndarray
        return self.__coeffs[numpy.s_[1:4]]

    def coeffs(self):
        # type: () -> numpy.ndarray
        return self.__coeffs

    #
    # Quaternion Methods
    #

    def squared_norm(self):
        # type: () -> float
        return self.__coeffs[0] ** 2 + numpy.dot(self.vec(), self.vec())

    def norm(self):
        # type: () -> float
        return numpy.sqrt(self.squared_norm())

    def normalize(self):
        # type: () -> None
        n = self.norm()
        if n > 0:
            self.__coeffs /= n

    def normalized(self):
        # type: () -> Quaternion
        n = copy.deepcopy(self)
        n.normalize()
        return n

    def dot(self, quaternion):
        return numpy.dot(self.__coeffs, quaternion.coeffs())

    def angular_distance(self, quaternion):
        # type: (Quaternion) -> float
        d = self * quaternion.conjugate()
        return 2.0 * math.atan2(numpy.linalg.norm(d.vec()), abs(d.w))

    def invert(self):
        n = self.squared_norm()
        if n > 0:
            self.__coeffs = numpy.hstack((self.__coeffs[0], -self.__coeffs[1:4])) / n
        else:
            raise ZeroDivisionError()

    def inverse(self):
        # type: () -> Quaternion
        n = copy.copy(self)
        n.invert()
        return n

    def conjugate(self):
        # type: () -> Quaternion
        return Quaternion(numpy.hstack((self.__coeffs[0], -self.__coeffs[1:4])))

    def slerp(self):
        raise NotImplementedError()

    def matrix(self):
        # type: () -> numpy.matrix
        a = self.normalized()
        s = a.w
        v = a.vec()
        x = v[0]
        y = v[1]
        z = v[2]
        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2
        return numpy.matrix([
            [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
            [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
            [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)]
        ]
        )

    def to_axis_angle(self):
        # type: () -> typing.Tuple[Vector3, float]
        assert abs(self.w) <= 1.0
        denom = numpy.sqrt(1 - self.w*self.w)
        angle = 2.0 * numpy.arccos(self.w)
        if denom > numpy.finfo(float).eps:
            return Vector3(self.vec() / denom), angle
        else:
            return Vector3(1.0, 0.0, 0.0), angle  # Singularity

    def to_rotation_vector(self):
        # type: () -> Vector3
        axis, angle = self.to_axis_angle()
        return Vector3(axis * angle)

    def to_euler(self, axis_1=Axis.AXIS_X, axis_2=Axis.AXIS_Y, axis_3=Axis.AXIS_Z, intrinsic=False):
        # type: (Axis, Axis, Axis, bool) -> typing.Tuple[float, float, float]
        repetition = axis_1 == axis_3

        if intrinsic:
            parity = (axis_2, axis_3) not in {
                (Axis.AXIS_Y, Axis.AXIS_X),
                (Axis.AXIS_Z, Axis.AXIS_Y),
                (Axis.AXIS_X, Axis.AXIS_Z)
            }
        else:
            parity = (axis_1, axis_2) not in {
                (Axis.AXIS_X, Axis.AXIS_Y),
                (Axis.AXIS_Y, Axis.AXIS_Z),
                (Axis.AXIS_Z, Axis.AXIS_X)
            }

        inner = axis_3 if intrinsic else axis_1

        i = (Axis.AXIS_X, Axis.AXIS_Y, Axis.AXIS_Z).index(inner)
        j = (i + 1 + parity) % 3
        k = (i + 2 - parity) % 3

        # h = k if repetition else i
        m = self.matrix()
        if repetition:
            sy = numpy.sqrt(m[i, j] ** 2 + m[i, k] ** 2)
            if sy > numpy.finfo(numpy.float64).eps:
                ax = numpy.arctan2(m[i, j], m[i, k])
                ay = numpy.arctan2(sy, m[i, i])
                az = numpy.arctan2(m[j, i], -m[k, i])
            else:
                ax = numpy.arctan2(-m[j, k], m[j, j])
                ay = numpy.arctan2(sy, m[i, i])
                az = 0.0
        else:  # not repetition
            cy = numpy.sqrt(m[i, i] ** 2 + m[j, i] ** 2)
            if cy > numpy.finfo(numpy.float64).eps:
                ax = numpy.arctan2(m[k, j], m[k, k])
                ay = numpy.arctan2(-m[k, i], cy)
                az = numpy.arctan2(m[j, i], m[i, i])
            else:
                ax = numpy.arctan2(-m[j, k], m[j, j])
                ay = numpy.arctan2(-m[k, i], cy)
                az = 0.0
        if parity:
            ax, ay, az = -ax, -ay, -az
        if intrinsic:
            ax, az = az, ax
        return ax, ay, az

    #
    # Static Methods
    #

    @classmethod
    def identity(cls):
        # type: () -> Quaternion
        return Quaternion(1, 0, 0, 0)

    @classmethod
    def from_two_vectors(cls, vector_a, vector_b):
        # type: (Vector3, Vector3) -> Quaternion
        a = Vector3(vector_a)
        b = Vector3(vector_b)
        a.normalize()
        b.normalize()
        c = a.dot(b)
        axis = a.cross(b)
        s = math.sqrt((1.0 + c) * 2.0)
        vec = axis * (1.0 / s)
        return Quaternion(w=s * 0.5, x=vec.x, y=vec.y, z=vec.z)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        # type: (Vector3, float) -> Quaternion
        a = axis.normalized()
        sa = math.sin(angle / 2.0)
        ca = math.cos(angle / 2.0)
        return Quaternion(w=ca, x=a.x * sa, y=a.y * sa, z=a.z * sa)

    @classmethod
    def from_euler_extrinsic(cls, rx, ry, rz):
        # type: (float, float ,float) -> Quaternion
        return Quaternion.from_axis_angle(Axis.AXIS_Z.value, rz) \
               * Quaternion.from_axis_angle(Axis.AXIS_Y.value, ry) \
               * Quaternion.from_axis_angle(Axis.AXIS_X.value, rx)

    @classmethod
    def from_euler_intrinsic(cls, angles, ordering):
        # type: () -> Quaternion
        qt = Quaternion()
        for angle, axis in zip(angles, ordering):
            qt *= Quaternion.from_axis_angle(axis.value, angle)
        return qt

    @classmethod
    def from_matrix(cls, matrix):
        # type: (numpy.matrix) -> Quaternion

        # Check matrix properties
        if matrix.shape != (3, 3):
            raise ValueError('Invalid matrix shape: Input must be a 3x3 matrix')
        if not numpy.allclose(numpy.dot(matrix, matrix.conj().transpose()), numpy.eye(3)):
            raise ValueError('Matrix must be orthogonal (transpose == inverse)')
        if not numpy.isclose(numpy.linalg.det(matrix), 1.0):
            raise ValueError('Matrix determinant must be +1.0')

        m = matrix.conj().transpose()  # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

        q = numpy.array(q)
        q *= 0.5 / numpy.sqrt(t)

        return Quaternion(q)

    #
    # Class Operators
    #

    def __eq__(self, quaternion):
        # type: (Quaternion) -> bool
        return numpy.isclose(self.__coeffs, quaternion.coeffs()).all()

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3((self * Quaternion(0, other.x, other.y, other.z) * self.conjugate()).vec())
        elif isinstance(other, Quaternion):
            w = self.w * other.w - numpy.dot(self.vec(), other.vec())
            vec = numpy.cross(self.vec(), other.vec()) + self.w * other.vec() + other.w * self.vec()
            return Quaternion(w=w, x=vec[0], y=vec[1], z=vec[2])
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Vector3):
            return Vector3((self * Quaternion(0, other.x, other.y, other.z) * self.conjugate()).vec())
        else:
            return NotImplemented

    def __imul__(self, other):
        return self * other

    #
    # Printing
    #

    def __repr__(self):
        return '[w={:.9f}, x={:.9f}, y={:.9f}, z={:.9f}]'.format(self.w, self.x, self.y, self.z)

    def __str__(self):
        return self.__repr__()

    #
    # ROS compatibility
    #

    @classmethod
    def from_msg(cls, ros_msg):
        # type: (genpy.Message) -> Quaternion
        if isinstance(ros_msg, RosQuaternion):
            return Quaternion(w=ros_msg.w, x=ros_msg.x, y=ros_msg.y, z=ros_msg.z)
        else:
            raise NotImplementedError()

    def to_msg(self, message_type=RosQuaternion):
        # type: (type) -> RosQuaternion
        if message_type == RosQuaternion:
            return RosQuaternion(w=self.w, x=self.x, y=self.y, z=self.z)
        else:
            raise NotImplementedError()
