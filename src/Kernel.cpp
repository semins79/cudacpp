#include <cudacpp/Kernel.h>
#include <cudacpp/KernelConfiguration.h>
#include <cudacpp/KernelParameters.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>

#include <cstdio>

namespace cudacpp
{
  String Kernel::paramToString(const int index, const KernelParameters & params) const
  {
    return "unknown";
  }
  Kernel::Kernel()
  {
  }
  Kernel::~Kernel()
  {
  }
  void Kernel::run(const KernelConfiguration & config, const KernelParameters & params)
  {
    char buf[256];
    sprintf(buf, "<<<{%d, %d}, {%d, %d, %d}, %d, %p>>>", config.getGridSize().x, config.getGridSize().y,
                                                         config.getBlockSize().x, config.getBlockSize().y, config.getBlockSize().z,
                                                         config.getSharedMemoryUsage(),
                                                         (config.getStream() ? config.getStream()->getHandle() : NULL));
    String str = name() + buf + "(";
    for (int i = 0; i < params.getNumParams(); ++i)
    {
      if (i > 0) str += ", ";
      str += paramToString(i, params);
    }
    str += ")";
    Runtime::logCudaCall("%s", str.c_str());
    runSub(config, params);
  }
}
