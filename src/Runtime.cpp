#include <cudacpp/Runtime.h>
#include <cudacpp/Array.h>
#include <cudacpp/ChannelFormatDescriptor.h>
#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Error.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>
#include <oscpp/Timer.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <map>
#include <string>
#include <vector>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <execinfo.h>
#include <unistd.h>

namespace cudacpp
{
  typedef std::map<void *, std::pair<size_t, String> > AllocMap;
  inline AllocMap * allocMapPtr(void * const t) { return  reinterpret_cast<AllocMap * >(t); }
  bool Runtime::quitOnError = false;
  bool Runtime::errorsToStderr = false;
  bool Runtime::logCalls = false;
  void * Runtime::cudaCallLog   = new std::vector<std::string>();
  void * Runtime::hostAllocs    = new AllocMap();
  void * Runtime::deviceAllocs  = new AllocMap();
  oscpp::Timer Runtime::timer;

  String Runtime::getStackTrace()
  {
    String backTrace = "";
    const int MAX_FUNCS = 1024;
    void * buffers[MAX_FUNCS];
    char ** strings;
    int numFuncs = backtrace(buffers, MAX_FUNCS);
    strings = backtrace_symbols(buffers, numFuncs);
    if (strings != NULL)
    {
      for (int i = 0; i < numFuncs; ++i)
      {
        backTrace += "               ";
        backTrace += strings[i];
        backTrace += "\n";
      }
      ::free(strings);
    }
    return backTrace;
  }
  Runtime::Runtime()
  {
  }
  Runtime::~Runtime()
  {
  }
  void   Runtime::init              ()
  {
    timer.start();
  }
  void   Runtime::finalize          ()
  {
    timer.stop();
  }
  int    Runtime::getLastError      ()
  {
    logCudaCall("cudaGetLastError()");
    return static_cast<int>(cudaGetLastError());
  }
  String Runtime::getErrorString    (const int errorID)
  {
    logCudaCall("cudaGetErrorString(%d)", errorID);
    return cudaGetErrorString(static_cast<cudaError_t>(errorID));
  }
  int    Runtime::getDeviceCount    ()
  {
    logCudaCall("cudaGetDeviceCount()");
    int t;
    checkCudaError(cudaGetDeviceCount(&t));
    return t;
  }
  void   Runtime::setDevice         (const int deviceID)
  {
    logCudaCall("cudaSetDevice(%d)", deviceID);
    checkCudaError(cudaSetDevice(deviceID));
  }
  int    Runtime::getDevice         ()
  {
    int ret;
    cudaError_t err = cudaGetDevice(&ret);
    logCudaCall("cudaGetDevice(%p, ret=%d)", &ret, ret);
    checkCudaError(err);
    if (err != cudaSuccess) ret = -1;
    return ret;
  }
  int    Runtime::chooseDevice      (const DeviceProperties * const desiredProps)
  {
    int ret;
    cudaDeviceProp props;
    strncpy(props.name, desiredProps->getName().c_str(), sizeof(props.name) / sizeof(props.name[0]) - 1);
    props.maxThreadsPerBlock  = desiredProps->getMaxThreadsPerBlock();
    props.maxThreadsDim[0]    = desiredProps->getMaxBlockSize().x;
    props.maxThreadsDim[1]    = desiredProps->getMaxBlockSize().y;
    props.maxThreadsDim[2]    = desiredProps->getMaxBlockSize().z;
    props.maxGridSize[0]      = desiredProps->getMaxGridSize().x;
    props.maxGridSize[1]      = desiredProps->getMaxGridSize().y;
    props.maxGridSize[2]      = desiredProps->getMaxGridSize().z;
    props.sharedMemPerBlock   = desiredProps->getSharedMemoryPerBlock();
    props.totalConstMem       = desiredProps->getTotalConstantMemory();
    props.warpSize            = desiredProps->getWarpSize();
    props.memPitch            = desiredProps->getMemoryPitch();
    props.regsPerBlock        = desiredProps->getRegistersPerBlock();
    props.clockRate           = desiredProps->getClockRate();
    props.textureAlignment    = desiredProps->getTextureAlignment();
    props.totalGlobalMem      = desiredProps->getTotalMemory();
    props.major               = desiredProps->getMajor();
    props.minor               = desiredProps->getMinor();
    props.multiProcessorCount = desiredProps->getMultiProcessorCount();

    logCudaCall("cudaChooseDevice(%p[\n"
                "    maxThreadsPerBlock  = %d\n"
                "    maxThreadsDim       = { %d %d %d }\n"
                "    maxGridSize         = { %d %d %d }\n"
                "    sharedMemPerBlock   = %d\n"
                "    totalConstMem       = %d\n"
                "    warpSize            = %d\n"
                "    memPitch            = %d\n"
                "    regsPerBlock        = %d\n"
                "    clockRate           = %d kHz\n"
                "    textureAlignment    = %d\n"
                "    totalGlobalMem      = %d\n"
                "    major               = %d\n"
                "    minor               = %d\n"
                "    multiProcessorCount = %d])",
                &props,
                props.name,
                props.maxThreadsPerBlock ,
                props.maxThreadsDim[0]   ,
                props.maxThreadsDim[1]   ,
                props.maxThreadsDim[2]   ,
                props.maxGridSize[0]     ,
                props.maxGridSize[1]     ,
                props.maxGridSize[2]     ,
                props.sharedMemPerBlock  ,
                props.totalConstMem      ,
                props.warpSize           ,
                props.memPitch           ,
                props.regsPerBlock       ,
                props.clockRate          ,
                props.textureAlignment   ,
                props.totalGlobalMem     ,
                props.major              ,
                props.minor              ,
                props.multiProcessorCount);
    checkCudaError(cudaChooseDevice(&ret, &props));
    return ret;
  }
  void * Runtime::malloc            (const size_t size)
  {
    void * t;
    if (size == 0) checkCudaError(cudaErrorMemoryAllocation);
    cudaError_t err = cudaMalloc(&t, size);
    logCudaCall("cudaMalloc(ptr, %u, ret=%p)", static_cast<unsigned int>(size), t);
    checkCudaError(err);
    if (err != cudaSuccess) return NULL;
    else allocMapPtr(deviceAllocs)->insert(std::make_pair(t, std::make_pair(size, getStackTrace())));
    return t;
  }
  void * Runtime::mallocPitch       (size_t * const pitch, const size_t widthInBytes, const size_t height)
  {
    void * t;
    cudaError_t err = cudaMallocPitch(&t, pitch, widthInBytes, height);
    logCudaCall("cudaMallocPitch(%p, %p, %d, %d, ret=%p)", &t, pitch, static_cast<int>(widthInBytes), static_cast<int>(height), t);
    checkCudaError(err);
    if (err != cudaSuccess) return NULL;
    return t;
  }
  void * Runtime::mallocHost        (const size_t size)
  {
    void * t;
    cudaError_t err = cudaMallocHost(&t, size);
    logCudaCall("cudaMallocHost(ptr, %u, ret=%p)", static_cast<unsigned int>(size), t);
    if (size == 0) checkCudaError(cudaErrorMemoryAllocation);
    checkCudaError(err);
    if (err != cudaSuccess) return NULL;
    else allocMapPtr(hostAllocs)->insert(std::make_pair(t, std::make_pair(size, getStackTrace())));
    return t;
  }
  Array * Runtime::mallocArray      (const ChannelFormatDescriptor & channel, const size_t width, const size_t height)
  {
    const struct cudaChannelFormatDesc * desc = reinterpret_cast<const struct cudaChannelFormatDesc * >(channel.getHandle());
    struct cudaArray * arr;
    logCudaCall("cudaMallocArray(ptr, %p, %d, %d)", desc, width, height);
    cudaError_t err = cudaMallocArray(&arr, desc, width, height);
    checkCudaError(err);
    if (err != cudaSuccess) return NULL;

    return new Array(arr);
  }
  void   Runtime::free              (void * const ptr)
  {
    logCudaCall("cudaFree(%p)", ptr);
    checkCudaError(cudaFree(ptr));
    AllocMap * map = allocMapPtr(deviceAllocs);
    if (map->find(ptr) == map->end()) checkCudaError(cudaErrorInvalidDevicePointer);
    else                              map->erase(map->find(ptr));
  }
  void   Runtime::freeArray         (Array * const arr)
  {
    logCudaCall("cudaFreeArray(%p)", arr->getHandle());
    checkCudaError(cudaFreeArray(reinterpret_cast<struct cudaArray * >(arr->getHandle())));
    delete arr;
  }
  void   Runtime::freeHost          (void * const ptr)
  {
    logCudaCall("cudaFreeHost(%p)", ptr);
    checkCudaError(cudaFreeHost(ptr));
    AllocMap * map = allocMapPtr(hostAllocs);
    if (map->find(ptr) == map->end()) checkCudaError(cudaErrorInvalidDevicePointer);
    else                              map->erase(map->find(ptr));
  }
  void   Runtime::memcpyHtoD        (void * const dst, const void * const src, const size_t size)
  {
    logCudaCall("cudaMemcpy(%p, %p, %d, cudaMemcpyHostToDevice)", dst, src, static_cast<int>(size));
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }
  void   Runtime::memcpyDtoH        (void * const dst, const void * const src, const size_t size)
  {
    logCudaCall("cudaMemcpy(%p, %p, %d, cudaMemcpyDeviceToHost)", dst, src, static_cast<int>(size));
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }
  void   Runtime::memcpyDtoD        (void * const dst, const void * const src, const size_t size)
  {
    logCudaCall("cudaMemcpy(%p, %p, %d, cudaMemcpyDeviceToDevice)", dst, src, static_cast<int>(size));
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  }
  void   Runtime::memcpyToSymbolHtoD(const char * const dst, const void * const src, const size_t size, const size_t offset)
  {
    logCudaCall("cudaMemcpyToSymbol(%s, %p, %d, %d, cudaMemcpyHostToDevice)", dst, src, static_cast<int>(size), static_cast<int>(offset));
    checkCudaError(cudaMemcpyToSymbol(dst, src, size, offset, cudaMemcpyHostToDevice));
  }
  void   Runtime::memcpyToSymbolDtoH(const char * const dst, const void * const src, const size_t size, const size_t offset)
  {
    logCudaCall("cudaMemcpyToSymbol(%s, %p, %d, %d, cudaMemcpyDeviceToHost)", dst, src, static_cast<int>(size), static_cast<int>(offset));
    checkCudaError(cudaMemcpyToSymbol(dst, src, size, offset, cudaMemcpyDeviceToHost));
  }
  void   Runtime::memcpyToSymbolDtoD(const char * const dst, const void * const src, const size_t size, const size_t offset)
  {
    logCudaCall("cudaMemcpyToSymbol(%s, %p, %d, %d, cudaMemcpyDeviceToDevice)", dst, src, static_cast<int>(size), static_cast<int>(offset));
    checkCudaError(cudaMemcpyToSymbol(dst, src, size, offset, cudaMemcpyDeviceToDevice));
  }
  void   Runtime::memcpyHtoDAsync   (void * const dst, const void * const src, const size_t size, Stream * stream)
  {
    logCudaCall("cudaMemcpyAsync(%p, %p, %d, cudaMemcpyHostToDevice, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyDtoHAsync   (void * const dst, const void * const src, const size_t size, Stream * stream)
  {
    logCudaCall("cudaMemcpyAsync(%p, %p, %d, cudaMemcpyDeviceToHost, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream->getHandle()));
  }
  void   Runtime::memcpyDtoDAsync   (void * const dst, const void * const src, const size_t size, Stream * stream)
  {
    logCudaCall("cudaMemcpyAsync(%p, %p, %d, cudaMemcpyDeviceToDevice, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyToSymbolHtoDAsync(const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream)
  {
    logCudaCall("cudaMemcpyToSymbolAsync(%s, %p, %d, cudaMemcpyHostToDevice, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyToSymbolAsync(dst, src, size, offset, cudaMemcpyHostToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyToSymbolDtoHAsync(const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream)
  {
    logCudaCall("cudaMemcpyToSymbolAsync(%s, %p, %d, cudaMemcpyDeviceToHost, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyToSymbolAsync(dst, src, size, offset, cudaMemcpyDeviceToHost, stream->getHandle()));
  }
  void   Runtime::memcpyToSymbolDtoDAsync(const char * const dst, const void * const src, const size_t size, const size_t offset, Stream * stream)
  {
    logCudaCall("cudaMemcpyToSymbolAsync(%s, %p, %d, cudaMemcpyDeviceToDevice, %p)", dst, src, static_cast<int>(size), stream->getHandle());
    checkCudaError(cudaMemcpyToSymbolAsync(dst, src, size, offset, cudaMemcpyDeviceToDevice, stream->getHandle()));
  }
  void   Runtime::memcpy2DHtoH      (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height)
  {
    logCudaCall("cudaMemcpy2D(%p, %d, %p, %d, %d, %d, cudaMemcpyHostToHost)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height));
    checkCudaError(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToHost));
  }
  void   Runtime::memcpy2DHtoD      (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height)
  {
    logCudaCall("cudaMemcpy2D(%p, %d, %p, %d, %d, %d, cudaMemcpyHostToDevice)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height));
    checkCudaError(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
  }
  void   Runtime::memcpy2DDtoD      (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height)
  {
    logCudaCall("cudaMemcpy2D(%p, %d, %p, %d, %d, %d, cudaMemcpyDeviceToDevice)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height));
    checkCudaError(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice));
  }
  void   Runtime::memcpy2DDtoH      (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height)
  {
    logCudaCall("cudaMemcpy2D(%p, %d, %p, %d, %d, %d, cudaMemcpyDeviceToHost)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height));
    checkCudaError(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost));
  }

  void   Runtime::memcpy2DHtoHAsync (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream)
  {
    logCudaCall("cudaMemcpy2DAsync(%p, %d, %p, %d, %d, %d, cudaMemcpyHostToHost, %p)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height), stream->getHandle());
    checkCudaError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToHost, stream->getHandle()));
  }
  void   Runtime::memcpy2DHtoDAsync (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream)
  {
    logCudaCall("cudaMemcpy2DAsync(%p, %d, %p, %d, %d, %d, cudaMemcpyHostToDevice, %p)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height), stream->getHandle());
    checkCudaError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice, stream->getHandle()));
  }
  void   Runtime::memcpy2DDtoDAsync (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream)
  {
    logCudaCall("cudaMemcpy2DAsync(%p, %d, %p, %d, %d, %d, cudaMemcpyDeviceToDevice, %p)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height), stream->getHandle());
    checkCudaError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream->getHandle()));
  }
  void   Runtime::memcpy2DDtoHAsync (void * const dst, const size_t dpitch, const void * const src, const size_t spitch, const size_t width, const size_t height, Stream * const stream)
  {
    logCudaCall("cudaMemcpy2DAsync(%p, %d, %p, %d, %d, %d, cudaMemcpyDeviceToHost, %p)", dst, static_cast<int>(dpitch), src, static_cast<int>(spitch), static_cast<int>(width), static_cast<int>(height), stream->getHandle());
    checkCudaError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost, stream->getHandle()));
  }

  void   Runtime::memcpyToArrayHtoH (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count)
  {
    logCudaCall("cudaMemcpyToArray(%p, %d, %d, %p, %d, cudaMemcpyHostToHost)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count));
    checkCudaError(cudaMemcpyToArray(array->getHandle(), dstX, dstY, src, count, cudaMemcpyHostToHost));
  }
  void   Runtime::memcpyToArrayHtoD (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count)
  {
    logCudaCall("cudaMemcpyToArray(%p, %d, %d, %p, %d, cudaMemcpyHostToDevice)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count));
    checkCudaError(cudaMemcpyToArray(array->getHandle(), dstX, dstY, src, count, cudaMemcpyHostToDevice));
  }
  void   Runtime::memcpyToArrayDtoD (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count)
  {
    logCudaCall("cudaMemcpyToArray(%p, %d, %d, %p, %d, cudaMemcpyDeviceToDevice)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count));
    checkCudaError(cudaMemcpyToArray(array->getHandle(), dstX, dstY, src, count, cudaMemcpyDeviceToDevice));
  }
  void   Runtime::memcpyToArrayDtoH (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count)
  {
    logCudaCall("cudaMemcpyToArray(%p, %d, %d, %p, %d, cudaMemcpyDeviceToHost)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count));
    checkCudaError(cudaMemcpyToArray(array->getHandle(), dstX, dstY, src, count, cudaMemcpyDeviceToHost));
  }

  void   Runtime::memcpyToArrayHtoHAsync (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyToArrayAsync(%p, %d, %d, %p, %d, cudaMemcpyHostToHost, %p)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyToArrayAsync(array->getHandle(), dstX, dstY, src, count, cudaMemcpyHostToHost, stream->getHandle()));
  }
  void   Runtime::memcpyToArrayHtoDAsync (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyToArrayAsync(%p, %d, %d, %p, %d, cudaMemcpyHostToDevice, %p)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyToArrayAsync(array->getHandle(), dstX, dstY, src, count, cudaMemcpyHostToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyToArrayDtoDAsync (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyToArrayAsync(%p, %d, %d, %p, %d, cudaMemcpyDeviceToDevice, %p)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyToArrayAsync(array->getHandle(), dstX, dstY, src, count, cudaMemcpyDeviceToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyToArrayDtoHAsync (Array * const array, const size_t dstX, const size_t dstY, const void * const src, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyToArrayAsync(%p, %d, %d, %p, %d, cudaMemcpyDeviceToHost, %p)", array->getHandle(), static_cast<int>(dstX), static_cast<int>(dstY), src, static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyToArrayAsync(array->getHandle(), dstX, dstY, src, count, cudaMemcpyDeviceToHost, stream->getHandle()));
  }

  void   Runtime::memcpyFromArrayHtoH (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count)
  {
    logCudaCall("cudaMemcpyFromArray(%p, p, %d, %d, %d, cudaMemcpyHostToHost)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count));
    checkCudaError(cudaMemcpyFromArray(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyHostToHost));
  }
  void   Runtime::memcpyFromArrayHtoD (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count)
  {
    logCudaCall("cudaMemcpyFromArray(%p, %p, %d, %d, %d, cudaMemcpyHostToDevice)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count));
    checkCudaError(cudaMemcpyFromArray(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyHostToDevice));
  }
  void   Runtime::memcpyFromArrayDtoD (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count)
  {
    logCudaCall("cudaMemcpyFromArray(%p, %p, %d, %d, %d, cudaMemcpyDeviceToDevice)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count));
    checkCudaError(cudaMemcpyFromArray(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyDeviceToDevice));
  }
  void   Runtime::memcpyFromArrayDtoH (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count)
  {
    logCudaCall("cudaMemcpyFromArray(%p, %p, %d, %d, %d, cudaMemcpyDeviceToHost)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count));
    checkCudaError(cudaMemcpyFromArray(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyDeviceToHost));
  }

  void   Runtime::memcpyFromArrayHtoHAsync (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyFromArrayAsync(%p, p, %d, %d, %d, cudaMemcpyHostToHost, %p)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyFromArrayAsync(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyHostToHost, stream->getHandle()));
  }
  void   Runtime::memcpyFromArrayHtoDAsync (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyFromArrayAsync(%p, %p, %d, %d, %d, cudaMemcpyHostToDevice, %p)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyFromArrayAsync(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyHostToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyFromArrayDtoDAsync (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyFromArrayAsync(%p, %p, %d, %d, %d, cudaMemcpyDeviceToDevice, %p)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyFromArrayAsync(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyDeviceToDevice, stream->getHandle()));
  }
  void   Runtime::memcpyFromArrayDtoHAsync (void * const dst, const Array * const array, const size_t srcX, const size_t srcY, const size_t count, Stream * const stream)
  {
    logCudaCall("cudaMemcpyFromArrayAsync(%p, %p, %d, %d, %d, cudaMemcpyDeviceToHost, %p)", dst, array->getHandle(), static_cast<int>(srcX), static_cast<int>(srcY), static_cast<int>(count), stream->getHandle());
    checkCudaError(cudaMemcpyFromArrayAsync(dst, reinterpret_cast<const struct cudaArray * >(array->getHandle()), srcX, srcY, count, cudaMemcpyDeviceToHost, stream->getHandle()));
  }

  void   Runtime::memset            (void * const devPtr, const int value, const size_t count)
  {
    logCudaCall("cudaMemset(%p, %d, %d)", devPtr, value, static_cast<int>(count));
    checkCudaError(cudaMemset(devPtr, value, count));
  }
  void   Runtime::memset2D          (void * const devPtr, const size_t pitch, const int value, const size_t width, const size_t height)
  {
    logCudaCall("cudaMemset2D(%p, %d, %d, %d, %d)", devPtr, static_cast<int>(pitch), value, static_cast<int>(width), static_cast<int>(height));
    checkCudaError(cudaMemset2D(devPtr, pitch, value, width, height));
  }
  void   Runtime::sync              ()
  {
    logCudaCall("cudaThreadSynchronize()");
    checkCudaError(cudaThreadSynchronize());
  }
  void   Runtime::exit              ()
  {
    logCudaCall("cudaThreadExit()");
    checkCudaError(cudaThreadExit());
  }
  void   Runtime::setQuitOnError    (const bool quit)
  {
    quitOnError = quit;
  }
  void   Runtime::setPrintErrors    (const bool print)
  {
    errorsToStderr = print;
  }
  void   Runtime::setLogCudaCalls   (const bool log)
  {
    logCalls = log;
  }
  void   Runtime::printAllocs(const unsigned int hostThresh, const unsigned int deviceThresh)
  {
    AllocMap * host = allocMapPtr(hostAllocs);
    AllocMap * dev  = allocMapPtr(deviceAllocs);
    if (host->size() < hostThresh && dev->size() < deviceThresh) return;
    printf("Allocations: %u host, %u device.\n", host->size(), dev->size());
    printf("Host Allocations:\n");
    for (AllocMap::iterator it = host->begin(); it != host->end(); ++it)
    {
      printf("%p - %u bytes\n%s\n\n", it->first, it->second.first, it->second.second.c_str());
    }
    printf("Device Allocations:\n");
    for (AllocMap::iterator it = dev->begin(); it != dev->end(); ++it)
    {
      printf("%p - %u bytes\n%s\n\n", it->first, it->second.first, it->second.second.c_str());
    }
  }
  void   Runtime::checkCudaError(const Error & error)
  {
    cudaError_t err = static_cast<cudaError_t>(error.getErrorValue());
    if (err != cudaSuccess)
    {
      Error error(static_cast<int>(err));
      if (errorsToStderr)
      {
        String backTrace = getStackTrace();

        std::vector<std::string> * log = reinterpret_cast<std::vector<std::string> * >(cudaCallLog);
        if (logCalls && log->size() > 0) fprintf(stderr, "%s: ", log->back().c_str());
        fprintf(stderr, "%s\n", error.toString().c_str());
        fprintf(stderr, "backtrace:\n%s\n", backTrace.c_str());
        fflush(stderr);
      }
      if (quitOnError)
      {
        fprintf(stderr, "\n\n\n\n");
        printCudaLog(stderr);
        // *reinterpret_cast<int * >(NULL) = 0;
        ::exit(1);
      }
    }
  }
  void   Runtime::logCudaCall(const char * const fmt, ...)
  {
    if (!logCalls) return;
    va_list vl;
    char time[40];
    static char buf[10240];
    va_start(vl, fmt);
    sprintf(time, "%13.8f", timer.getElapsedSeconds());
    vsnprintf(buf, 10239, fmt, vl);
    std::vector<std::string> * log = reinterpret_cast<std::vector<std::string> * >(cudaCallLog);
    log->push_back((String(time) + ": " + buf).c_str());
    va_end(vl);
  }
  void   Runtime::printCudaLog(FILE * fp)
  {
    std::vector<std::string> * log = reinterpret_cast<std::vector<std::string> * >(cudaCallLog);
    fprintf(fp, "CUDA Log:\n");
    for (unsigned int i = 0; i < log->size(); ++i)
    {
      fprintf(fp, "  %s\n", log->at(i).c_str());
    }
  }
}
