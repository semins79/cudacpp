#include <cudacpp/Event.h>
#include <cudacpp/Kernel.h>
#include <cudacpp/KernelConfiguration.h>
#include <cudacpp/KernelParameters.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>

const int NUM_KERNELS_TO_STREAM = 200;
const int NUM_STREAMING_KERNELS = 100;
const int NUM_BLOCKS = 2048;
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
  int minStream = 0, maxStream = 0, numStreaming = 0;
  cudacpp::Runtime::init();
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setDevice(0);
  cudacpp::Stream stream;
  cudacpp::Event evts[NUM_KERNELS_TO_STREAM];
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

  for (int i = 0; i < NUM_KERNELS_TO_STREAM; ++i)
  {
    while (numStreaming > 0)
    {
      if (evts[minStream].query())
      {
        ++minStream;
        --numStreaming;
      }
      else
      {
        break;
      }
    }
    while (numStreaming >= NUM_STREAMING_KERNELS)
    {
      evts[minStream++].sync();
      --numStreaming;
    }
    cudacpp::Runtime::memcpyHtoDAsync(gpuMem0, cpuMem0, NUM_FLOATS * sizeof(float), &stream);
    cudacpp::Runtime::memcpyHtoDAsync(gpuMem1, cpuMem1, NUM_FLOATS * sizeof(float), &stream);
    kernel.run(conf, params);
    cudacpp::Runtime::memcpyDtoHAsync(cpuMem2, gpuMem2, NUM_FLOATS * sizeof(float), &stream);
    evts[maxStream++].record(&stream);
    ++numStreaming;
  }
  while (numStreaming > 0)
  {
    evts[minStream++].sync();
    --numStreaming;
  }
  cudacpp::Runtime::free(gpuMem0);
  cudacpp::Runtime::free(gpuMem1);
  cudacpp::Runtime::free(gpuMem2);
  cudacpp::Runtime::freeHost(cpuMem0);
  cudacpp::Runtime::freeHost(cpuMem1);
  cudacpp::Runtime::freeHost(cpuMem2);
  cudacpp::Runtime::printCudaLog(stderr);

  return 0;
}
