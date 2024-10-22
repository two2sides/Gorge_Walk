# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class EventReq(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EventReq()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsEventReq(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # EventReq
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EventReq
    def ClientId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # EventReq
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # EventReq
    def DataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    # EventReq
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # EventReq
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0


def EventReqStart(builder):
    builder.StartObject(2)


def Start(builder):
    return EventReqStart(builder)


def EventReqAddClientId(builder, clientId):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(clientId), 0)


def AddClientId(builder, clientId):
    return EventReqAddClientId(builder, clientId)


def EventReqAddData(builder, data):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)


def AddData(builder, data):
    return EventReqAddData(builder, data)


def EventReqStartDataVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def StartDataVector(builder, numElems):
    return EventReqStartDataVector(builder, numElems)


def EventReqEnd(builder):
    return builder.EndObject()


def End(builder):
    return EventReqEnd(builder)
