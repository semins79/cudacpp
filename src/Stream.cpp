#include <cudacpp/Stream.h>
#include <cudacpp/Error.h>
#include <cudacpp/Runtime.h>

#include <cuda_runtime_api.h>

namespace cudacpp
{
  Stream Stream::nullStream(0);
  Stream::Stream(const int ignored)
  {
    Runtime::logCudaCall("cudaStreamCreate(%p, nullStream)", &handle);
    handle = 0;
  }

  Stream::Stream()
  {
    Runtime::logCudaCall("cudaStreamCreate(%p)", &handle);
    Runtime::checkCudaError(cudaStreamCreate(&handle));
  }
  Stream::~Stream()
  {
    if (handle != 0)
    {
      Runtime::logCudaCall("cudaStreamDestroy(%p)", &handle);
      Runtime::checkCudaError(cudaStreamDestroy(handle));
    }
  }

  void Stream::sync()
  {
    Runtime::logCudaCall("cudaStreamSynchronize(%p)", &handle);
    Runtime::checkCudaError(cudaStreamSynchronize(handle));
  }
  bool Stream::query()
  {
    Runtime::logCudaCall("cudaStreamQuery(%p)", &handle);
    cudaError_t err = cudaStreamQuery(handle);
    if (err != cudaSuccess && err != cudaErrorNotReady) Runtime::checkCudaError(err);
    return err == cudaSuccess;
  }

  cudaStream_t & Stream::getHandle()
  {
    return handle;
  }
}
