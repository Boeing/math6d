import numpy
import copy
import typing
import genpy

from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import Transform as RosTransform
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Vector3 as RosVector3
from math6d.geometry.msg import MsgConversion

from math6d.geometry.quaternion import Quaternion
from math6d.geometry.vector3 import Vector3


class Transform(MsgConversion):
    """
    Isometric Transform
    """

    TRANSLATION_KW = 'translation'
    ROTATION_KW = 'rotation'

    def __init__(self, *args, **kwargs):

        self.__rotation = None
        self.__translation = None

        if len(args) == 0 and len(kwargs) == 0:
            self.__rotation = Quaternion()
            self.__translation = Vector3()
        elif len(args) == 1:
            _arg = args[0]
            if isinstance(_arg, genpy.Message):
                t = Transform.from_msg(_arg)
                self.__rotation = t.rotation
                self.__translation = t.translation
            else:
                raise NotImplementedError('Expected Message type, got: {}'.format(type(_arg)))
        elif len(args) + len(kwargs) == 2:
            if len(args) == 2:
                _tr = args[0]
                _rot = args[1]
            else:
                _tr = kwargs[self.TRANSLATION_KW] if self.TRANSLATION_KW in kwargs else Vector3()
                _rot = kwargs[self.ROTATION_KW] if self.ROTATION_KW in kwargs else Quaternion()

            if isinstance(_tr, RosVector3):
                self.__translation = Vector3().from_msg(_tr)
            elif isinstance(_tr, Vector3):
                self.__translation = _tr
            else:
                self.__translation = Vector3(_tr)

            if isinstance(_rot, RosQuaternion):
                self.__rotation = Quaternion().from_msg(_rot)
            elif isinstance(_rot, Quaternion):
                self.__rotation = _rot
            else:
                self.__rotation = Quaternion(_rot)
        else:
            raise NotImplementedError('Unexpected input args of len: {}'.format(len(args) + len(kwargs)))

    #
    # Properties
    #

    @property
    def translation(self):
        # type: () -> Vector3
        return self.__translation

    @property
    def rotation(self):
        # type: () -> Quaternion
        return self.__rotation

    @translation.setter
    def translation(self, translation):
        # type: (Vector3) -> None
        self.__translation = translation

    @rotation.setter
    def rotation(self, rotation):
        # type: (Quaternion) -> None
        self.__rotation = rotation

    #
    # Transform Methods
    #

    def invert(self):
        self.__rotation.invert()
        self.__translation = -(self.__rotation * self.__translation)

    def inverse(self):
        # type: () -> Transform
        t = copy.deepcopy(self)
        t.invert()
        return t

    def matrix(self):
        # type: () -> numpy.matrix
        m = numpy.identity(4, dtype=numpy.float64)
        m[:3, :3] = self.rotation.matrix()
        m[:3, 3] = self.translation.data()
        return numpy.asmatrix(m)

    def dist_squared(self, other):
        """Return the square of the metric distance, as the unweighted sum of
        linear and angular distance, to the 'other' transform. Note
        that the units and scale among linear and angular
        representations matters heavily.
        """
        return self.translation.dist_squared(other.translation) + self.rotation.angular_distance(other.rotation) ** 2

    def dist(self, other):
        """Return the metric distance, as unweighted combined linear and
        angular distance, to the 'other' transform. Note that the
        units and scale among linear and angular representations
        matters heavily.
        """
        return numpy.sqrt(self.dist_squared(other))

    #
    # Static Methods
    #

    @classmethod
    def identity(cls):
        # type: () -> Transform
        return Transform()

    @classmethod
    def from_matrix(cls, matrix):
        # type: (numpy.matrix) -> Transform

        # Check matrix properties
        if matrix.shape != (4, 4):
            raise ValueError('Invalid matrix shape: Input must be a 4x4 matrix')
        if not numpy.isclose(numpy.linalg.det(matrix), 1.0):
            raise ValueError('Matrix determinant must be +1.0')

        q = Quaternion.from_matrix(matrix[0:3, 0:3])
        t = Vector3(matrix[0:3, 3].flatten())

        return Transform(t, q)

    #
    # Operators
    #

    def __mul__(self, other):
        if isinstance(other, Transform):
            t = self.translation + (self.rotation * other.translation)
            q = self.rotation * other.rotation
            return Transform(t, q)
        elif isinstance(other, Vector3):
            v = numpy.ones(4)
            v[:3] = other.data()
            return Vector3(numpy.dot(self.matrix(), v)[:3])
        else:
            raise NotImplementedError('Cannot multiply type of: {}'.format(type(other)))

    #
    # Printing
    #

    def __repr__(self):
        return 'T[tran={}, rot={}]'.format(self.translation, self.rotation)

    def __str__(self):
        return self.__repr__()

    #
    # ROS compatibility
    #

    @classmethod
    def from_msg(cls, ros_msg):
        # type: (genpy.Message) -> Transform
        if isinstance(ros_msg, RosTransform):
            return Transform(
                translation=Vector3.from_msg(ros_msg.translation),
                rotation=Quaternion.from_msg(ros_msg.rotation)
            )
        elif isinstance(ros_msg, RosPose):
            return Transform(
                translation=Vector3.from_msg(ros_msg.position),
                rotation=Quaternion.from_msg(ros_msg.orientation)
            )
        else:
            raise NotImplementedError('Unknown message type to convert to math6D.Transform: {}'.format(type(ros_msg)))

    def to_msg(self, message_type=RosTransform):
        # type: (type) -> typing.Union[RosTransform, RosPose]
        if message_type == RosTransform:
            return RosTransform(
                translation=self.__translation.to_msg(RosVector3),
                rotation=self.__rotation.to_msg(RosQuaternion)
            )
        elif message_type == RosPose:
            return RosPose(
                position=self.__translation.to_msg(RosPoint),
                orientation=self.__rotation.to_msg(RosQuaternion)
            )
        else:
            raise NotImplementedError('Unknown target msg type: {}'.format(message_type))
