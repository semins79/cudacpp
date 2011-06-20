#include <cudacpp/Kernel.h>
#include <cudacpp/KernelConfiguration.h>
#include <cudacpp/KernelParameters.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>

const int NUM_BLOCKS = 10240;
const int NUM_THREADS = 512;
const int NUM_FLOATS = NUM_BLOCKS * NUM_THREADS;

class AddKernel : public cudacpp::Kernel
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
      extern void addkernel_runSub(const int, const int, float * , float * , float * , int);
      int gs = config.getGridSize().x;
      int bs = config.getBlockSize().x;
      float * p0 = params.get<float * >(0);
      float * p1 = params.get<float * >(1);
      float * p2 = params.get<float * >(2);
      int p3 = params.get<int>(3);
      addkernel_runSub(gs, bs, p0, p1, p2, p3);
    }
  public:
    AddKernel() { }

    virtual cudacpp::String name() const;
};

cudacpp::String AddKernel::name() const
{
  return "AddKernel";
}

int main(int argc, char ** argv)
{
  cudacpp::Runtime::init();
  cudacpp::Runtime::setQuitOnError(true);
  cudacpp::Runtime::setPrintErrors(true);
  cudacpp::Runtime::setLogCudaCalls(true);
  cudacpp::Runtime::setDevice(0);
  float * cpuMem0 = new float[NUM_FLOATS];
  float * cpuMem1 = new float[NUM_FLOATS];
  float * cpuMem2 = new float[NUM_FLOATS];
  float * gpuMem0 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  float * gpuMem1 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  float * gpuMem2 = reinterpret_cast<float * >(cudacpp::Runtime::malloc(NUM_FLOATS * sizeof(float)));
  for (int i = 0; i < NUM_FLOATS; ++i)
  {
    cpuMem0[i] = static_cast<float>(i) / static_cast<float>(NUM_FLOATS - 1);
    cpuMem1[i] = 1.0f - cpuMem0[i];
  }
  cudacpp::Runtime::memcpyHtoD(gpuMem0, cpuMem0, NUM_FLOATS * sizeof(float));
  cudacpp::Runtime::memcpyHtoD(gpuMem1, cpuMem1, NUM_FLOATS * sizeof(float));
  cudacpp::KernelConfiguration conf(NUM_BLOCKS, NUM_THREADS);
  cudacpp::KernelParameters params(4);
  params.set(gpuMem0, gpuMem1, gpuMem2, NUM_FLOATS);
  AddKernel kernel;
  kernel.run(conf, params);
  cudacpp::Runtime::sync();
  cudacpp::Runtime::memcpyDtoH(cpuMem0, gpuMem0, NUM_FLOATS * sizeof(float));
  cudacpp::Runtime::memcpyDtoH(cpuMem1, gpuMem1, NUM_FLOATS * sizeof(float));
  cudacpp::Runtime::memcpyDtoH(cpuMem2, gpuMem2, NUM_FLOATS * sizeof(float));
  for (int i = 0; i < NUM_FLOATS; ++i)
  {
    float t = 1.0f - cpuMem2[i];
    if (t < 0) t = -t;
    if (t > 1e-5) fprintf(stderr, "%7d: %50f - %10f %10f\n", i, cpuMem2[i], cpuMem0[i], cpuMem1[i]);
  }
  cudacpp::Runtime::printCudaLog(stderr);

  return 0;
}
