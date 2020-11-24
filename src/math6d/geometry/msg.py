import abc


class MsgConversion(object):

    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def from_msg(cls, ros_quaternion):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_msg(self, message_type):
        raise NotImplementedError()
