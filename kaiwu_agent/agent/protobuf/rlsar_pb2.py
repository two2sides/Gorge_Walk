# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rlsar.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0brlsar.proto\"3\n\rPBObservation\x12\x0f\n\x07\x66\x65\x61ture\x18\x01 \x03(\x02\x12\x11\n\tlegal_act\x18\x02 \x03(\x05\"9\n\x11PBObservationList\x12$\n\x0cobservations\x18\x01 \x03(\x0b\x32\x0e.PBObservation\"(\n\x08PBAction\x12\x0b\n\x03\x61\x63t\x18\x01 \x01(\x05\x12\x0f\n\x07sub_act\x18\x02 \x01(\x05\"*\n\x0cPBActionList\x12\x1a\n\x07\x61\x63tions\x18\x01 \x03(\x0b\x32\t.PBActionb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rlsar_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_PBOBSERVATION']._serialized_start=15
  _globals['_PBOBSERVATION']._serialized_end=66
  _globals['_PBOBSERVATIONLIST']._serialized_start=68
  _globals['_PBOBSERVATIONLIST']._serialized_end=125
  _globals['_PBACTION']._serialized_start=127
  _globals['_PBACTION']._serialized_end=167
  _globals['_PBACTIONLIST']._serialized_start=169
  _globals['_PBACTIONLIST']._serialized_end=211
# @@protoc_insertion_point(module_scope)
