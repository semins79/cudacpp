__shared__ float sha[512];
__shared__ float shb[512];
__global__ void subDiffKernel(const float * a, const float * b, float * res, const int numFloats)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  sha[threadIdx.x] = a[index];
  shb[threadIdx.x] = b[index];
  __syncthreads();
  float temp1 = 0.0f, temp2 = 0.0f;
  for (int i = -8; i <= 8; ++i)
  {
    const int index2 = (threadIdx.x + i + numFloats) % numFloats;
    temp1 += sha[index2] * sha[index2];
    temp2 += shb[index2] * shb[index2];
  }
  res[index] = (temp2 - temp1) / 9.0f;
}
void subdiff_runSub(const int gs, const int bs, float * p0, float * p1, float * p2, int p3)
{
  subDiffKernel<<<gs, bs>>>(p0, p1, p2, p3);
}
