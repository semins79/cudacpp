#include <cudacpp/ChannelFormatDescriptor.h>
#include <cudacpp/Vector2.h>
#include <cudacpp/Vector4.h>
#include <driver_types.h>
#include <channel_descriptor.h>

namespace cudacpp
{
  inline static cudaChannelFormatDesc * ccfdPtr(void * const t) { return  reinterpret_cast<cudaChannelFormatDesc * >(t); }
  inline static cudaChannelFormatDesc & ccfdRef(void * const t) { return *reinterpret_cast<cudaChannelFormatDesc * >(t); }

  ChannelFormatDescriptor::ChannelFormatDescriptor()
  {
    handle = new cudaChannelFormatDesc;
  }

  ChannelFormatDescriptor::~ChannelFormatDescriptor()
  {
    delete ccfdPtr(handle);
  }

  int ChannelFormatDescriptor::getX() const { return ccfdPtr(handle)->x; }
  int ChannelFormatDescriptor::getY() const { return ccfdPtr(handle)->y; }
  int ChannelFormatDescriptor::getZ() const { return ccfdPtr(handle)->z; }
  int ChannelFormatDescriptor::getFormat() const
  {
    switch (ccfdPtr(handle)->f)
    {
    case cudaChannelFormatKindSigned:   return CHANNEL_FORMAT_SIGNED;
    case cudaChannelFormatKindUnsigned: return CHANNEL_FORMAT_UNSIGNED;
    case cudaChannelFormatKindFloat:    return CHANNEL_FORMAT_FLOAT;
    case cudaChannelFormatKindNone:
    default:                            return CHANNEL_FORMAT_NONE;
    }
  }
  void * ChannelFormatDescriptor::getHandle()
  {
    return handle;
  }
  const void * ChannelFormatDescriptor::getHandle() const
  {
    return handle;
  }

  #define CCFD_CREATE(type)                                         \
  template <>                                                       \
  ChannelFormatDescriptor * ChannelFormatDescriptor::create<type>() \
  {                                                                 \
    ChannelFormatDescriptor * ret = new ChannelFormatDescriptor;    \
    ccfdRef(ret->handle) = cudaCreateChannelDesc<type>();           \
    return ret;                                                     \
  }                                                                 \

  #define CCFD_CREATE_WRAPPER(vector_type, type)                            \
  template <>                                                               \
  ChannelFormatDescriptor * ChannelFormatDescriptor::create<vector_type>()  \
  {                                                                         \
    ChannelFormatDescriptor * ret = new ChannelFormatDescriptor;            \
    ccfdRef(ret->handle) = cudaCreateChannelDesc<type>();                   \
    return ret;                                                             \
  }                                                                         \

  CCFD_CREATE(char)
  CCFD_CREATE(signed char)
  CCFD_CREATE(unsigned char)
  CCFD_CREATE(char1)
  CCFD_CREATE(uchar1)
  CCFD_CREATE(char2)
  CCFD_CREATE(uchar2)
  CCFD_CREATE(char4)
  CCFD_CREATE(uchar4)
  CCFD_CREATE(short)
  CCFD_CREATE(unsigned short)
  CCFD_CREATE(short1)
  CCFD_CREATE(ushort1)
  CCFD_CREATE(short2)
  CCFD_CREATE(ushort2)
  CCFD_CREATE(short4)
  CCFD_CREATE(ushort4)
  CCFD_CREATE(int)
  CCFD_CREATE(unsigned int)
  CCFD_CREATE(int1)
  CCFD_CREATE(uint1)
  CCFD_CREATE(int2)
  CCFD_CREATE(uint2)
  CCFD_CREATE(int4)
  CCFD_CREATE(uint4)
  CCFD_CREATE(long)
  CCFD_CREATE(unsigned long)
  CCFD_CREATE(long1)
  CCFD_CREATE(ulong1)
  CCFD_CREATE(long2)
  CCFD_CREATE(ulong2)
  CCFD_CREATE(long4)
  CCFD_CREATE(ulong4)
  CCFD_CREATE(float)
  CCFD_CREATE(float1)
  CCFD_CREATE(float2)
  CCFD_CREATE(float4)
  CCFD_CREATE_WRAPPER(Vector2<char>          , char2)
  CCFD_CREATE_WRAPPER(Vector2<unsigned char> , uchar2)
  CCFD_CREATE_WRAPPER(Vector4<char>          , char4)
  CCFD_CREATE_WRAPPER(Vector4<unsigned char> , uchar4)
  CCFD_CREATE_WRAPPER(Vector2<short>         , short2)
  CCFD_CREATE_WRAPPER(Vector2<unsigned short>, ushort2)
  CCFD_CREATE_WRAPPER(Vector4<short>         , short4)
  CCFD_CREATE_WRAPPER(Vector4<unsigned short>, ushort4)
  CCFD_CREATE_WRAPPER(Vector2<int>           , int2)
  CCFD_CREATE_WRAPPER(Vector2<unsigned int>  , uint2)
  CCFD_CREATE_WRAPPER(Vector4<int>           , int4)
  CCFD_CREATE_WRAPPER(Vector4<unsigned int>  , uint4)
  CCFD_CREATE_WRAPPER(Vector2<long>          , long2)
  CCFD_CREATE_WRAPPER(Vector2<unsigned long> , ulong2)
  CCFD_CREATE_WRAPPER(Vector4<long>          , long4)
  CCFD_CREATE_WRAPPER(Vector4<unsigned long> , ulong4)
  CCFD_CREATE_WRAPPER(Vector2<float>         , float2)
  CCFD_CREATE_WRAPPER(Vector4<float>         , float4)

  #undef CCFD_CREATE
  #undef CCFD_CREATE_WRAPPER

}
