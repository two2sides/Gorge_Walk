// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: aisrv_actor_req_resp.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_aisrv_5factor_5freq_5fresp_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_aisrv_5factor_5freq_5fresp_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3009000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3009002 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/descriptor.pb.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_aisrv_5factor_5freq_5fresp_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_aisrv_5factor_5freq_5fresp_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
namespace kaiwu_aisvr {
class AisrvActorRequest;
class AisrvActorRequestDefaultTypeInternal;
extern AisrvActorRequestDefaultTypeInternal _AisrvActorRequest_default_instance_;
class AisrvActorResponse;
class AisrvActorResponseDefaultTypeInternal;
extern AisrvActorResponseDefaultTypeInternal _AisrvActorResponse_default_instance_;
}  // namespace kaiwu_aisvr
PROTOBUF_NAMESPACE_OPEN
template<> ::kaiwu_aisvr::AisrvActorRequest* Arena::CreateMaybeMessage<::kaiwu_aisvr::AisrvActorRequest>(Arena*);
template<> ::kaiwu_aisvr::AisrvActorResponse* Arena::CreateMaybeMessage<::kaiwu_aisvr::AisrvActorResponse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace kaiwu_aisvr {

// ===================================================================

class AisrvActorRequest :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:kaiwu_aisvr.AisrvActorRequest) */ {
 public:
  AisrvActorRequest();
  virtual ~AisrvActorRequest();

  AisrvActorRequest(const AisrvActorRequest& from);
  AisrvActorRequest(AisrvActorRequest&& from) noexcept
    : AisrvActorRequest() {
    *this = ::std::move(from);
  }

  inline AisrvActorRequest& operator=(const AisrvActorRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline AisrvActorRequest& operator=(AisrvActorRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const std::string& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline std::string* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const AisrvActorRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const AisrvActorRequest* internal_default_instance() {
    return reinterpret_cast<const AisrvActorRequest*>(
               &_AisrvActorRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(AisrvActorRequest& a, AisrvActorRequest& b) {
    a.Swap(&b);
  }
  inline void Swap(AisrvActorRequest* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline AisrvActorRequest* New() const final {
    return CreateMaybeMessage<AisrvActorRequest>(nullptr);
  }

  AisrvActorRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<AisrvActorRequest>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)
    final;
  void CopyFrom(const AisrvActorRequest& from);
  void MergeFrom(const AisrvActorRequest& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  void DiscardUnknownFields();
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(AisrvActorRequest* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "kaiwu_aisvr.AisrvActorRequest";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kComposeIdFieldNumber = 2,
    kFeatureFieldNumber = 3,
    kClientIdFieldNumber = 1,
    kSampleSizeFieldNumber = 4,
  };
  // repeated int32 compose_id = 2;
  int compose_id_size() const;
  void clear_compose_id();
  ::PROTOBUF_NAMESPACE_ID::int32 compose_id(int index) const;
  void set_compose_id(int index, ::PROTOBUF_NAMESPACE_ID::int32 value);
  void add_compose_id(::PROTOBUF_NAMESPACE_ID::int32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      compose_id() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_compose_id();

  // repeated float feature = 3;
  int feature_size() const;
  void clear_feature();
  float feature(int index) const;
  void set_feature(int index, float value);
  void add_feature(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      feature() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_feature();

  // required int32 client_id = 1;
  bool has_client_id() const;
  void clear_client_id();
  ::PROTOBUF_NAMESPACE_ID::int32 client_id() const;
  void set_client_id(::PROTOBUF_NAMESPACE_ID::int32 value);

  // required int32 sample_size = 4;
  bool has_sample_size() const;
  void clear_sample_size();
  ::PROTOBUF_NAMESPACE_ID::int32 sample_size() const;
  void set_sample_size(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:kaiwu_aisvr.AisrvActorRequest)
 private:
  class _Internal;

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArenaLite _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 > compose_id_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > feature_;
  ::PROTOBUF_NAMESPACE_ID::int32 client_id_;
  ::PROTOBUF_NAMESPACE_ID::int32 sample_size_;
  friend struct ::TableStruct_aisrv_5factor_5freq_5fresp_2eproto;
};
// -------------------------------------------------------------------

class AisrvActorResponse :
    public ::PROTOBUF_NAMESPACE_ID::MessageLite /* @@protoc_insertion_point(class_definition:kaiwu_aisvr.AisrvActorResponse) */ {
 public:
  AisrvActorResponse();
  virtual ~AisrvActorResponse();

  AisrvActorResponse(const AisrvActorResponse& from);
  AisrvActorResponse(AisrvActorResponse&& from) noexcept
    : AisrvActorResponse() {
    *this = ::std::move(from);
  }

  inline AisrvActorResponse& operator=(const AisrvActorResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline AisrvActorResponse& operator=(AisrvActorResponse&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const std::string& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline std::string* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const AisrvActorResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const AisrvActorResponse* internal_default_instance() {
    return reinterpret_cast<const AisrvActorResponse*>(
               &_AisrvActorResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(AisrvActorResponse& a, AisrvActorResponse& b) {
    a.Swap(&b);
  }
  inline void Swap(AisrvActorResponse* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline AisrvActorResponse* New() const final {
    return CreateMaybeMessage<AisrvActorResponse>(nullptr);
  }

  AisrvActorResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<AisrvActorResponse>(arena);
  }
  void CheckTypeAndMergeFrom(const ::PROTOBUF_NAMESPACE_ID::MessageLite& from)
    final;
  void CopyFrom(const AisrvActorResponse& from);
  void MergeFrom(const AisrvActorResponse& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  void DiscardUnknownFields();
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(AisrvActorResponse* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "kaiwu_aisvr.AisrvActorResponse";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  std::string GetTypeName() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kComposeIdFieldNumber = 1,
    kOutputFieldNumber = 2,
  };
  // repeated int32 compose_id = 1;
  int compose_id_size() const;
  void clear_compose_id();
  ::PROTOBUF_NAMESPACE_ID::int32 compose_id(int index) const;
  void set_compose_id(int index, ::PROTOBUF_NAMESPACE_ID::int32 value);
  void add_compose_id(::PROTOBUF_NAMESPACE_ID::int32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      compose_id() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_compose_id();

  // repeated float output = 2;
  int output_size() const;
  void clear_output();
  float output(int index) const;
  void set_output(int index, float value);
  void add_output(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      output() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_output();

  // @@protoc_insertion_point(class_scope:kaiwu_aisvr.AisrvActorResponse)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArenaLite _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 > compose_id_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > output_;
  friend struct ::TableStruct_aisrv_5factor_5freq_5fresp_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// AisrvActorRequest

// required int32 client_id = 1;
inline bool AisrvActorRequest::has_client_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void AisrvActorRequest::clear_client_id() {
  client_id_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AisrvActorRequest::client_id() const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorRequest.client_id)
  return client_id_;
}
inline void AisrvActorRequest::set_client_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  client_id_ = value;
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorRequest.client_id)
}

// repeated int32 compose_id = 2;
inline int AisrvActorRequest::compose_id_size() const {
  return compose_id_.size();
}
inline void AisrvActorRequest::clear_compose_id() {
  compose_id_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AisrvActorRequest::compose_id(int index) const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorRequest.compose_id)
  return compose_id_.Get(index);
}
inline void AisrvActorRequest::set_compose_id(int index, ::PROTOBUF_NAMESPACE_ID::int32 value) {
  compose_id_.Set(index, value);
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorRequest.compose_id)
}
inline void AisrvActorRequest::add_compose_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  compose_id_.Add(value);
  // @@protoc_insertion_point(field_add:kaiwu_aisvr.AisrvActorRequest.compose_id)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
AisrvActorRequest::compose_id() const {
  // @@protoc_insertion_point(field_list:kaiwu_aisvr.AisrvActorRequest.compose_id)
  return compose_id_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
AisrvActorRequest::mutable_compose_id() {
  // @@protoc_insertion_point(field_mutable_list:kaiwu_aisvr.AisrvActorRequest.compose_id)
  return &compose_id_;
}

// repeated float feature = 3;
inline int AisrvActorRequest::feature_size() const {
  return feature_.size();
}
inline void AisrvActorRequest::clear_feature() {
  feature_.Clear();
}
inline float AisrvActorRequest::feature(int index) const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorRequest.feature)
  return feature_.Get(index);
}
inline void AisrvActorRequest::set_feature(int index, float value) {
  feature_.Set(index, value);
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorRequest.feature)
}
inline void AisrvActorRequest::add_feature(float value) {
  feature_.Add(value);
  // @@protoc_insertion_point(field_add:kaiwu_aisvr.AisrvActorRequest.feature)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AisrvActorRequest::feature() const {
  // @@protoc_insertion_point(field_list:kaiwu_aisvr.AisrvActorRequest.feature)
  return feature_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AisrvActorRequest::mutable_feature() {
  // @@protoc_insertion_point(field_mutable_list:kaiwu_aisvr.AisrvActorRequest.feature)
  return &feature_;
}

// required int32 sample_size = 4;
inline bool AisrvActorRequest::has_sample_size() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void AisrvActorRequest::clear_sample_size() {
  sample_size_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AisrvActorRequest::sample_size() const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorRequest.sample_size)
  return sample_size_;
}
inline void AisrvActorRequest::set_sample_size(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  sample_size_ = value;
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorRequest.sample_size)
}

// -------------------------------------------------------------------

// AisrvActorResponse

// repeated int32 compose_id = 1;
inline int AisrvActorResponse::compose_id_size() const {
  return compose_id_.size();
}
inline void AisrvActorResponse::clear_compose_id() {
  compose_id_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AisrvActorResponse::compose_id(int index) const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorResponse.compose_id)
  return compose_id_.Get(index);
}
inline void AisrvActorResponse::set_compose_id(int index, ::PROTOBUF_NAMESPACE_ID::int32 value) {
  compose_id_.Set(index, value);
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorResponse.compose_id)
}
inline void AisrvActorResponse::add_compose_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  compose_id_.Add(value);
  // @@protoc_insertion_point(field_add:kaiwu_aisvr.AisrvActorResponse.compose_id)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
AisrvActorResponse::compose_id() const {
  // @@protoc_insertion_point(field_list:kaiwu_aisvr.AisrvActorResponse.compose_id)
  return compose_id_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
AisrvActorResponse::mutable_compose_id() {
  // @@protoc_insertion_point(field_mutable_list:kaiwu_aisvr.AisrvActorResponse.compose_id)
  return &compose_id_;
}

// repeated float output = 2;
inline int AisrvActorResponse::output_size() const {
  return output_.size();
}
inline void AisrvActorResponse::clear_output() {
  output_.Clear();
}
inline float AisrvActorResponse::output(int index) const {
  // @@protoc_insertion_point(field_get:kaiwu_aisvr.AisrvActorResponse.output)
  return output_.Get(index);
}
inline void AisrvActorResponse::set_output(int index, float value) {
  output_.Set(index, value);
  // @@protoc_insertion_point(field_set:kaiwu_aisvr.AisrvActorResponse.output)
}
inline void AisrvActorResponse::add_output(float value) {
  output_.Add(value);
  // @@protoc_insertion_point(field_add:kaiwu_aisvr.AisrvActorResponse.output)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AisrvActorResponse::output() const {
  // @@protoc_insertion_point(field_list:kaiwu_aisvr.AisrvActorResponse.output)
  return output_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AisrvActorResponse::mutable_output() {
  // @@protoc_insertion_point(field_mutable_list:kaiwu_aisvr.AisrvActorResponse.output)
  return &output_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace kaiwu_aisvr

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_aisrv_5factor_5freq_5fresp_2eproto
