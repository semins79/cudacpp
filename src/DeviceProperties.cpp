#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Error.h>
#include <cudacpp/Runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <vector>

namespace cudacpp
{
  DeviceProperties::DeviceProperties()
  {
  }
  DeviceProperties * DeviceProperties::create(const String & pName,
                                        const int pMaxThreadsPerBlock,
                                        const Vector3<int> & pMaxThreadsDim,
                                        const Vector3<int> & pMaxGridSize,
                                        const int pSharedMemPerBlock,
                                        const int pWarpSize,
                                        const int MemPitch,
                                        const int pRegsPerBlock,
                                        const int pClockRate,
                                        const int pMajor,
                                        const int pMinor,
                                        const int pMultiProcessorCount,
                                        const size_t pTotalConstantMemory,
                                        const size_t pTotalMemBytes,
                                        const size_t pTextureAlign)
  {
    DeviceProperties * ret = new DeviceProperties;
    ret->name = pName;
    ret->maxThreadsPerBlock = pMaxThreadsPerBlock;
    ret->maxThreadsDim = pMaxThreadsDim;
    ret->maxGridSize = pMaxGridSize;
    ret->sharedMemPerBlock = pSharedMemPerBlock;
    ret->warpSize = pWarpSize;
    ret->memPitch;
    ret->regsPerBlock = pRegsPerBlock;
    ret->clockRate = pClockRate;
    ret->major = pMajor;
    ret->minor = pMinor;
    ret->multiProcessorCount = pMultiProcessorCount;
    ret->totalConstantMemory = pTotalConstantMemory;
    ret->totalMemBytes = pTotalMemBytes;
    ret->textureAlign = pTextureAlign;
    return ret;
  }

  DeviceProperties * DeviceProperties::get(const int deviceID)
  {
    DeviceProperties * ret = new DeviceProperties;
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, deviceID);
    Runtime::logCudaCall("cudaGetDeviceProperties(ptr, %d)", deviceID);
    Runtime::checkCudaError(err);
    if (err != cudaSuccess)
    {
      delete ret;
      return NULL;
    }

    ret->name = props.name;
    ret->maxThreadsPerBlock   = props.maxThreadsPerBlock;
    ret->maxThreadsDim.x      = props.maxThreadsDim[0];
    ret->maxThreadsDim.y      = props.maxThreadsDim[1];
    ret->maxThreadsDim.z      = props.maxThreadsDim[2];
    ret->maxGridSize.x        = props.maxGridSize[0];
    ret->maxGridSize.y        = props.maxGridSize[1];
    ret->maxGridSize.z        = props.maxGridSize[2];
    ret->sharedMemPerBlock    = props.sharedMemPerBlock;
    ret->totalConstantMemory  = props.totalConstMem;
    ret->warpSize             = props.warpSize;
    ret->memPitch             = props.memPitch;
    ret->regsPerBlock         = props.regsPerBlock;
    ret->clockRate            = props.clockRate;
    ret->textureAlign         = props.textureAlignment;
    ret->totalMemBytes        = props.totalGlobalMem;
    ret->major                = props.major;
    ret->minor                = props.minor;
    ret->multiProcessorCount  = props.multiProcessorCount;

    return ret;
  }
  const String & DeviceProperties::getName() const
  {
    return name;
  }
  int DeviceProperties::getMaxThreadsPerBlock() const
  {
    return maxThreadsPerBlock;
  }
  const Vector3<int> & DeviceProperties::getMaxBlockSize() const
  {
    return maxThreadsDim;
  }
  const Vector3<int> & DeviceProperties::getMaxGridSize() const
  {
    return maxGridSize;
  }
  int DeviceProperties::getSharedMemoryPerBlock() const
  {
    return sharedMemPerBlock;
  }
  size_t DeviceProperties::getTotalConstantMemory() const
  {
    return totalConstantMemory;
  }
  int DeviceProperties::getWarpSize() const
  {
    return warpSize;
  }
  int DeviceProperties::getMemoryPitch() const
  {
    return memPitch;
  }
  int DeviceProperties::getRegistersPerBlock() const
  {
    return regsPerBlock;
  }
  int DeviceProperties::getClockRate() const
  {
    return clockRate;
  }
  size_t DeviceProperties::getTextureAlignment() const
  {
    return textureAlign;
  }
  size_t DeviceProperties::getTotalMemory() const
  {
    return totalMemBytes;
  }
  int DeviceProperties::getMajor() const
  {
    return major;
  }
  int DeviceProperties::getMinor() const
  {
    return minor;
  }
  int DeviceProperties::getMultiProcessorCount() const
  {
    return multiProcessorCount;
  }
}
