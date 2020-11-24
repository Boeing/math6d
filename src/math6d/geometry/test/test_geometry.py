import math
import unittest

import numpy
from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import Transform as RosTransform
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Vector3 as RosVector3

from math6d.geometry.quaternion import Quaternion, Axis
from math6d.geometry.transform import Transform
from math6d.geometry.vector3 import Vector3


class TestGeometry(unittest.TestCase):

    def test_transform_multiplication(self):
        t1 = Transform(translation=numpy.array([1, 2, 3]), rotation=numpy.array([0, 1, 0, 0]))
        t2 = Transform(translation=numpy.array([2, 1, 0]), rotation=numpy.array([0, 0, 1, 0]))
        t3 = t1 * t2
        numpy_mult = t1.matrix() * t2.matrix()
        self.assertTrue(numpy.isclose(t3.matrix(), numpy_mult).all())

    def test_inverse(self):
        t = Transform(translation=numpy.array([1, 2, 3]), rotation=numpy.array([0, 0, 1, 0]))
        numpy_inv = numpy.linalg.inv(t.matrix())
        t_inv = t.inverse().matrix()
        self.assertTrue(numpy.isclose(t_inv, numpy_inv).all())

    def test_quaternion_inverse(self):
        qt = Quaternion(1, 0, 0, 0)
        qt_inv = qt.inverse()
        self.assertTrue(numpy.isclose(qt.coeffs(), qt_inv.coeffs()).all())

    def test_angle_between(self):
        v1 = Vector3(100, 0, 0)
        v2 = Vector3(0, 1, 0)
        angle = v1.angle(v2)
        self.assertAlmostEqual(angle, math.pi / 2)

        v1 = Vector3(1000, 0, 0)
        v2 = Vector3(1, 1, 0)
        angle = v1.angle(v2)
        self.assertAlmostEqual(angle, math.pi / 4)

    def test_ros_conversion(self):
        ros_t = RosTransform(
            translation=RosVector3(x=1, y=3, z=2),
            rotation=RosQuaternion(x=1)
        )

        t1 = Transform(ros_t)

        t1.to_msg()

        self.assertEqual(ros_t, t1.to_msg())
        self.assertTrue(numpy.isclose(Transform.from_msg(ros_t).matrix(), t1.matrix()).all())

        ros_pose = t1.to_msg(message_type=RosPose)
        ros_pose_2 = Transform(ros_pose).to_msg(message_type=RosPose)
        self.assertEqual(ros_pose_2, ros_pose_2)

    def test_from_two_vecs(self):
        v1 = Vector3(100, 0, 0)
        v2 = Vector3(1, 1, 0)
        qt = Quaternion.from_two_vectors(v1, v2)
        angle = qt.angular_distance(quaternion=Quaternion())
        self.assertAlmostEqual(angle * 180.0 / math.pi, 45.0)
        v3 = v1 * qt
        v3.normalize()
        v2.normalize()
        self.assertTrue(numpy.isclose(v2.data(), v3.data()).all())

    def test_euler(self):
        angles = (0.1, 0.2, 0.3)
        qt_ex = Quaternion.from_euler_extrinsic(*angles)
        qt_in = Quaternion.from_euler_intrinsic(
            angles=reversed(angles),
            ordering=[Axis.AXIS_Z, Axis.AXIS_Y, Axis.AXIS_X]
        )
        self.assertTrue(numpy.isclose(qt_ex.coeffs(), qt_in.coeffs()).all())

        euler = qt_ex.to_euler()
        for i in range(3):
            self.assertAlmostEqual(angles[i], euler[i])

    def test_from_xy(self):
        random_qt = Quaternion.from_euler_extrinsic(0.2, 0.3, 0.9)

        x_vec = Vector3(1, 0, 0) * random_qt
        y_vec = Vector3(0, 1, 0) * random_qt
        z_vec = Vector3(0, 0, 1) * random_qt

        m = numpy.identity(3)
        m[:, 0] = x_vec.data()
        m[:, 1] = y_vec.data()
        m[:, 2] = z_vec.data()
        m = numpy.matrix(m)

        new_q = Quaternion.from_matrix(m)

        self.assertTrue(numpy.isclose(new_q.coeffs(), random_qt.coeffs()).all())

    def test_transform_matrix(self):
        t = Transform(
            Vector3(1, 2, 3),
            Quaternion.from_euler_extrinsic(0.1, 0.2, 0.3)
        )
        m = t.matrix()
        t2 = Transform.from_matrix(m)
        m2 = t2.matrix()
        self.assertTrue(numpy.isclose(m, m2).all())


if __name__ == '__main__':
    unittest.main()
