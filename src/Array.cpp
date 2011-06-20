#include <cudacpp/Array.h>
#include <driver_types.h>

namespace cudacpp
{
  Array::Array(cudaArray * const cudaArrayHandle)
  {
    handle = cudaArrayHandle;
  }

  cudaArray * Array::getHandle()
  {
    return handle;
  }
  const cudaArray * Array::getHandle() const
  {
    return handle;
  }
}
