#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>

const int NUM_INTS = 128;

int main(int argc, char ** argv)
{
  cudacpp::Runtime::init();
  cudacpp::Runtime::setQuitOnError(false);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::setLogCudaCalls(true);
  printf("Num Devices: %d\n", cudacpp::Runtime::getDeviceCount());
  cudacpp::Runtime::setDevice(0);
  cudacpp::Stream stream;
  int * gpuMem0 = reinterpret_cast<int * >(cudacpp::Runtime::malloc(NUM_INTS * sizeof(int)));
  int * gpuMem1 = reinterpret_cast<int * >(cudacpp::Runtime::malloc(NUM_INTS * sizeof(int)));
  int * cpuMem0 = new int[NUM_INTS];
  int * cpuMem1 = reinterpret_cast<int * >(cudacpp::Runtime::mallocHost(NUM_INTS * sizeof(int)));
  for (int i = 0; i < NUM_INTS; ++i) cpuMem0[i] = i;
  cudacpp::Runtime::memcpyHtoD(gpuMem0, cpuMem0, NUM_INTS * sizeof(int));
  // cudacpp::Runtime::memcpyDtoD(gpuMem1, gpuMem1, NUM_INTS * sizeof(int));
  cudacpp::Runtime::memcpyDtoH(cpuMem1, gpuMem0, NUM_INTS * sizeof(int));
  for (int i = 0; i < NUM_INTS; ++i) if (cpuMem1[i] != i) fprintf(stderr, "%d: %d\n", i, cpuMem1[i]);
  for (int i = 0; i < NUM_INTS; ++i) cpuMem0[i] = cpuMem1[i] = NUM_INTS - i - 1;
  cudacpp::Runtime::memcpyHtoDAsync(gpuMem0, cpuMem0, NUM_INTS * sizeof(int), &stream);
  cudacpp::Runtime::memcpyHtoDAsync(gpuMem0, cpuMem1, NUM_INTS * sizeof(int), &stream);
  // cudacpp::Runtime::memcpyDtoDAsync(gpuMem1, gpuMem0, NUM_INTS * sizeof(int), &stream);
  cudacpp::Runtime::memcpyDtoHAsync(cpuMem1, gpuMem0, NUM_INTS * sizeof(int), &stream);
  for (int i = 0; i < NUM_INTS; ++i) if (cpuMem1[i] != NUM_INTS - i - 1) fprintf(stderr, "%d: %d\n", i, cpuMem1[i]);
  delete [] cpuMem0;
  cudacpp::Runtime::free(gpuMem0);
  cudacpp::Runtime::free(gpuMem1);
  cudacpp::Runtime::freeHost(cpuMem1);

  cudacpp::Runtime::printCudaLog(stderr);
  return 0;
}
