#include <cudacpp/KernelParameters.h>

#include <cstdio>
#include <cstring>

namespace cudacpp
{
  void KernelParameters::storeParam(const void * const buf, const int size)
  {
    mem[numArgs] = new char[size];
    memcpy(mem[numArgs], buf, size);
    sizes[numArgs] = size;
    ++numArgs;
  }
  KernelParameters::KernelParameters(const int numParams) : numArgs(0)
  {
    paramCount = (numParams < 1 ? 1 : numParams);
    mem = new void*[paramCount];
    sizes = new int[paramCount];
    for (int i = 0; i < paramCount; ++i) mem[i] = NULL;
  }
  KernelParameters::~KernelParameters()
  {
    for (int i = 0; i < numArgs; ++i) if (mem[i] != NULL) delete [] (char * )mem[i];
    delete [] mem;
    delete [] sizes;
  }
}
