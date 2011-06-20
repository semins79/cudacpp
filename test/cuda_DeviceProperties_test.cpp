#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Runtime.h>

#include <cstdio>

int main(int argc, char ** argv)
{
  cudacpp::Runtime::init();
  cudacpp::Runtime::setQuitOnError(false);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::setLogCudaCalls(true);
  int count = cudacpp::Runtime::getDeviceCount();

  for (int i = 0; i < count; ++i)
  {
    cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(i);


    printf("props(%d):\n", i);
    printf("  name:                   %s\n", props->getName().c_str());
    printf("  max threads per block:  %d\n", props->getMaxThreadsPerBlock());
    printf("  max threads:            { %d %d %d }\n", props->getMaxBlockSize()[0], props->getMaxBlockSize()[1], props->getMaxBlockSize()[2]);
    printf("  max blocks:             { %d %d %d }\n", props->getMaxGridSize() [0], props->getMaxGridSize() [1], props->getMaxGridSize() [2]);
    printf("  shared mem:             %d kB\n", props->getSharedMemoryPerBlock() / 1024);
    printf("  constant mem:           %u kB\n", props->getTotalConstantMemory() / 1024);
    printf("  warp size:              %d threads\n", props->getWarpSize());
    printf("  memory pitch:           %d kB\n", props->getMemoryPitch() / 1024);
    printf("  32-bit registers:       %d\n", props->getRegistersPerBlock());
    printf("  clock rate:             %d kHz\n", props->getClockRate());
    printf("  texture alignment:      %d bytes\n", props->getTextureAlignment());
    printf("  total memory:           %d MB\n", props->getTotalMemory() / 1048576);
    printf("  major:                  %d\n", props->getMajor());
    printf("  minor:                  %d\n", props->getMinor());
    printf("  multiprocessor count:   %d\n", props->getMultiProcessorCount());

    delete props;
  }
  cudacpp::DeviceProperties * props = cudacpp::DeviceProperties::get(count + 1);
  if (props != NULL)
  {
    fprintf(stderr, "Error, returned properties for an invalid device.\n");
    fflush(stderr);
  }

  return 0;
}
