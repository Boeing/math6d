import copy
import typing
import genpy

import numpy
from geometry_msgs.msg import Vector3 as RosVector3
from geometry_msgs.msg import Point as RosPoint
from math6d.geometry.msg import MsgConversion


class Vector3(MsgConversion):
    X_KW = 'x'
    Y_KW = 'y'
    Z_KW = 'z'

    def __init__(self, *args, **kwargs):
        # Coefficients are stored (x, y, z)
        self.__data = numpy.array([0, 0, 0], dtype=numpy.float64)

        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 1:
            _arg = args[0]
            if isinstance(_arg, genpy.Message):
                v = Vector3.from_msg(_arg)
                self.__data[:] = v.data()
            elif isinstance(_arg, Vector3):
                self.__data[:] = _arg.data()
            elif isinstance(_arg, numpy.ndarray) and _arg.size == 3:
                self.__data[:] = _arg
            else:
                raise NotImplementedError()
        elif len(args) == 3:
            self.__data[0] = args[0]
            self.__data[1] = args[1]
            self.__data[2] = args[2]
        elif len(args) + len(kwargs) == 3:
            self.__data[0] = kwargs[self.X_KW] if self.X_KW in kwargs else 0.0
            self.__data[1] = kwargs[self.Y_KW] if self.Y_KW in kwargs else 0.0
            self.__data[2] = kwargs[self.Z_KW] if self.Z_KW in kwargs else 0.0
        else:
            raise NotImplementedError()

    #
    # Properties
    #

    @property
    def x(self):
        # type: () -> float
        return self.__data[0]

    @property
    def y(self):
        # type: () -> float
        return self.__data[1]

    @property
    def z(self):
        # type: () -> float
        return self.__data[2]

    @x.setter
    def x(self, x):
        self.__data[0] = x

    @y.setter
    def y(self, y):
        self.__data[1] = y

    @z.setter
    def z(self, z):
        self.__data[2] = z

    def data(self):
        # type: () -> numpy.ndarray
        return self.__data

    #
    # Vector Methods
    #

    def norm(self):
        # type: () -> float
        return numpy.linalg.norm(self.__data, ord=2)

    def normalize(self):
        # type: () -> None
        self.__data /= self.norm()

    def length(self):
        # type: () -> float
        return self.norm()

    def normalized(self):
        # type: () -> Vector3
        n = copy.deepcopy(self)
        n.normalize()
        return n

    def dot(self, vector):
        # type: (Vector3) -> float
        return numpy.dot(self.__data, vector.data())

    def cross(self, vector):
        # type: (Vector3) -> Vector3
        return Vector3(numpy.cross(self.__data, vector.data()))

    def angle(self, vector):
        # type: (Vector3) -> float
        return numpy.arccos(numpy.dot(self.__data, vector.data()) / (self.norm() * vector.norm()))

    def dist(self, other):
        """Compute euclidean distance between points given by self and 'other'."""
        return numpy.sqrt(self.dist_squared(other))

    def dist_squared(self, other):
        """Compute euclidean distance between points given by self and 'other'."""
        if type(other) == Vector3:
            sub = numpy.subtract(self.data(), other.data())
            return numpy.dot(sub, sub)
        else:
            return NotImplemented

    #
    # Operators
    #

    def __eq__(self, vector):
        # type: (Vector3) -> bool
        return numpy.isclose(self.__data, vector.data()).all()

    def __sub__(self, other):
        if type(other) == Vector3:
            return Vector3(numpy.subtract(self.__data, other.data()))
        else:
            return NotImplemented

    def __isub__(self, other):
        if type(other) == Vector3:
            self.__data -= other.data()
        else:
            return NotImplemented
        return self

    def __add__(self, other):
        if type(other) == Vector3:
            return Vector3(self.__data + other.data())
        else:
            return NotImplemented

    def __iadd__(self, other):
        if type(other) == Vector3:
            self.__data += other.data()
        else:
            return NotImplemented
        return self

    def __neg__(self):
        return Vector3(-self.__data)

    def __mul__(self, other):
        if type(other) == Vector3:
            return self.dot(other)
        elif isinstance(other, (numpy.number, float, int)):
            return Vector3(numpy.dot(self.__data, other))
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, (numpy.number, float, int)):
            self.__data *= other
        else:
            return NotImplemented
        return self

    def __rmul__(self, other):
        if isinstance(other, (numpy.number, float, int)):
            return Vector3(other * self.__data)
        else:
            return NotImplemented

    #
    # Printing
    #

    def __repr__(self):
        return '[{:.6f}, {:.6f}, {:.6f}]'.format(*self.__data)

    def __str__(self):
        return self.__repr__()

    #
    # ROS compatibility
    #

    @classmethod
    def from_msg(cls, ros_msg):
        # type: (genpy.Message) -> Vector3
        if isinstance(ros_msg, RosVector3):
            return Vector3(x=ros_msg.x, y=ros_msg.y, z=ros_msg.z)
        elif isinstance(ros_msg, RosPoint):
            return Vector3(x=ros_msg.x, y=ros_msg.y, z=ros_msg.z)
        else:
            raise NotImplementedError()

    def to_msg(self, message_type=RosVector3):
        # type: (type) -> typing.Union[RosVector3, RosPoint]
        if message_type == RosVector3:
            return RosVector3(x=self.x, y=self.y, z=self.z)
        elif message_type == RosPoint:
            return RosPoint(x=self.x, y=self.y, z=self.z)
        else:
            raise NotImplementedError()
