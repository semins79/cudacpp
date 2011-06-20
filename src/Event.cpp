#include <cudacpp/Event.h>
#include <cudacpp/Error.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

#include <cuda_runtime_api.h>

namespace cudacpp
{
  Event::Event()
  {
    Runtime::logCudaCall("cudaEventCreate(%p)", &handle);
    Runtime::checkCudaError(cudaEventCreate(&handle));
  }
  Event::~Event()
  {
    Runtime::logCudaCall("cudaEventDestroy(%p)", &handle);
    Runtime::checkCudaError(cudaEventDestroy(handle));
  }

  void Event::record(Stream * stream)
  {
    Runtime::logCudaCall("cudaEventRecord(%p, %p)", &handle, &stream->getHandle());
    Runtime::checkCudaError(cudaEventRecord(handle, stream->getHandle()));
  }
  bool Event::query()
  {
    Runtime::logCudaCall("cudaEventQuery(%p)", &handle);
    cudaError_t val = cudaEventQuery(handle);
    if (val != cudaSuccess && val != cudaErrorNotReady) Runtime::checkCudaError(val);
    return val == cudaSuccess;
  }
  void Event::sync()
  {
    Runtime::logCudaCall("cudaEventSynchronize(%p)", &handle);
    Runtime::checkCudaError(cudaEventSynchronize(handle));
  }
  double Event::getElapsedSeconds(const Event * end)
  {
    float t;
    Runtime::logCudaCall("cudaEventElapsedTime(eventPtr, %p, %p)", &handle, end->handle);
    Runtime::checkCudaError(cudaEventElapsedTime(&t, handle, end->handle));
    return static_cast<double>(t) / 1000.0;
  }

  cudaEvent_t & Event::getHandle()
  {
    return handle;
  }
}
