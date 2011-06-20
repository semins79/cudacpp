#include <cudacpp/Event.h>
#include <cudacpp/Kernel.h>
#include <cudacpp/KernelConfiguration.h>
#include <cudacpp/KernelParameters.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>

const int NUM_BLOCKS = 10240;
const int NUM_THREADS = 512;
const int NUM_FLOATS = NUM_BLOCKS * NUM_THREADS;

class SubDiffKernel : public cudacpp::Kernel
{
  protected:
    virtual cudacpp::String paramToString(const int index, const cudacpp::KernelParameters & params) const
    {
      char buf[256];
      if (index < 3)  sprintf(buf, "float*=%p", params.get<float * >(index));
      else            sprintf(buf, "int=%d", params.get<int>(index));
      return buf;
    }
    virtual void runSub(const cudacpp::KernelConfiguration & config, const cudacpp::KernelParameters & params)
    {
      extern void subdiff_runSub(const int, const int, float * , float * , float * , int);
      int gs = config.getGridSize().x;
      int bs = config.getBlockSize().x;
      float * p0 = params.get<float * >(0);
      float * p1 = params.get<float * >(1);
      float * p2 = params.get<float * >(2);
      int p3 = params.get<int>(3);
      subdiff_runSub(gs, bs, p0, p1, p2, p3);
    }
  public:
    SubDiffKernel() { }

    virtual cudacpp::String name() const;
};

cudacpp::String SubDiffKernel::name() const
{
  return "SubDiffKernel";
}

int main(int argc, char ** argv)
{
  cudacpp::Runtime::init();
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setDevice(0);
  cudacpp::Stream stream;
  cudacpp::Event event0, event1, event2, event3, event4, event5, event6;
  float * cpuMem0 = reinterpret_cast<float * >(cudacpp::Runtime::mallocHost(NUM_FLOATS * sizeof(float)));
  float * cpuMem1 = reinterpret_cast<float * >(cudacpp::Runtime::mallocHost(NUM_FLOATS * sizeof(float)));
  float * cpuMem2 = reinterpret_cast<float * >(cudacpp::Runtime::mallocHost(NUM_FLOATS * sizeof(float)));
  float * gpuMem0 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  float * gpuMem1 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  float * gpuMem2 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  cudacpp::KernelConfiguration conf(NUM_BLOCKS, NUM_THREADS, 0, &stream);
  cudacpp::KernelParameters params(4);
  params.set(gpuMem0, gpuMem1, gpuMem2, NUM_FLOATS);
  SubDiffKernel kernel;

  for (int i = 0; i < NUM_FLOATS; ++i)
  {
    cpuMem0[i] = static_cast<float>(i) / static_cast<float>(NUM_FLOATS - 1);
    cpuMem1[i] = 1.0f - cpuMem0[i];
  }
  event0.record(&stream);
  cudacpp::Runtime::memcpyHtoDAsync(gpuMem0, cpuMem0, NUM_FLOATS * sizeof(float), &stream);
  event1.record(&stream);
  cudacpp::Runtime::memcpyHtoDAsync(gpuMem1, cpuMem1, NUM_FLOATS * sizeof(float), &stream);
  event2.record(&stream);
  kernel.run(conf, params);
  event3.record(&stream);
  cudacpp::Runtime::memcpyDtoHAsync(cpuMem0, gpuMem0, NUM_FLOATS * sizeof(float), &stream);
  event4.record(&stream);
  cudacpp::Runtime::memcpyDtoHAsync(cpuMem1, gpuMem1, NUM_FLOATS * sizeof(float), &stream);
  event5.record(&stream);
  cudacpp::Runtime::memcpyDtoHAsync(cpuMem2, gpuMem2, NUM_FLOATS * sizeof(float), &stream);
  event6.record(&stream);
  event6.sync();
  cudacpp::Runtime::free(gpuMem0);
  cudacpp::Runtime::free(gpuMem1);
  cudacpp::Runtime::free(gpuMem2);
  cudacpp::Runtime::freeHost(cpuMem0);
  cudacpp::Runtime::freeHost(cpuMem1);
  cudacpp::Runtime::freeHost(cpuMem2);
  cudacpp::Runtime::printCudaLog(stderr);

  printf("memcpy took %.4f s\n", event0.getElapsedSeconds(&event1));
  printf("memcpy took %.4f s\n", event1.getElapsedSeconds(&event2));
  printf("kernel took %.4f s\n", event2.getElapsedSeconds(&event3));
  printf("memcpy took %.4f s\n", event3.getElapsedSeconds(&event4));
  printf("memcpy took %.4f s\n", event4.getElapsedSeconds(&event5));
  printf("memcpy took %.4f s\n", event5.getElapsedSeconds(&event6));

  return 0;
}
