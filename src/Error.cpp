#include <cudacpp/Error.h>
#include <cudacpp/String.h>
#include <cstring>

#include <cuda_runtime_api.h>

namespace cudacpp
{
  Error::Error(const int errorVal) : error(errorVal)
  {
  }

  int Error::getErrorValue() const
  {
    return error;
  }
  String Error::toString() const
  {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
  }
}
