# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class Reject(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Reject()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReject(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # Reject
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Reject
    def RetCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Reject
    def Message(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None


def RejectStart(builder):
    builder.StartObject(2)


def Start(builder):
    return RejectStart(builder)


def RejectAddRetCode(builder, retCode):
    builder.PrependInt32Slot(0, retCode, 0)


def AddRetCode(builder, retCode):
    return RejectAddRetCode(builder, retCode)


def RejectAddMessage(builder, message):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(message), 0)


def AddMessage(builder, message):
    return RejectAddMessage(builder, message)


def RejectEnd(builder):
    return builder.EndObject()


def End(builder):
    return RejectEnd(builder)