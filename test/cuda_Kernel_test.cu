__global__ void addKernel(const float * a, const float * b, float * res, const int numFloats)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  res[index] = a[index] + b[index];
}
void addkernel_runSub(const int gs, const int bs, float * p0, float * p1, float * p2, int p3)
{
  addKernel<<<gs, bs>>>(p0, p1, p2, p3);
}
