#include <cudacpp/KernelConfiguration.h>
#include <cudacpp/Stream.h>
#include <cudacpp/String.h>
#include <cudacpp/Vector2.h>
#include <cudacpp/Vector3.h>
#include <string>

namespace cudacpp
{
  KernelConfiguration::KernelConfiguration(int numBlocks, int numThreads, int sharedMem, Stream * const cudaStream)
   : gridSize(numBlocks, 1), blockSize(numThreads, 1, 1), shmem(sharedMem), stream(cudaStream)
  {
  }
  KernelConfiguration::KernelConfiguration(int numBlocksX, int numBlocksY, int numThreadsX, int numThreadsY, int numThreadsZ, int sharedMem, Stream * const cudaStream)
   : gridSize(numBlocksX, numBlocksY), blockSize(numThreadsX, numThreadsY, numThreadsZ), shmem(sharedMem), stream(cudaStream)
  {
  }
  KernelConfiguration::KernelConfiguration(const Vector2<int> & pGridSize, const Vector3<int> & pBlockSize, int sharedMem, Stream * const cudaStream)
   : gridSize(pGridSize), blockSize(pBlockSize), shmem(sharedMem), stream(cudaStream)
  {
  }
  KernelConfiguration::KernelConfiguration(const KernelConfiguration & rhs)
   : gridSize(rhs.gridSize), blockSize(rhs.blockSize), shmem(rhs.shmem), stream(rhs.stream)
  {
  }
  KernelConfiguration & KernelConfiguration::operator = (const KernelConfiguration & rhs)
  {
    gridSize = rhs.gridSize;
    blockSize = rhs.blockSize;
    shmem = rhs.shmem;
    stream = rhs.stream;
    return *this;
  }
  String KernelConfiguration::toString() const
  {
    char buf[20];
    sprintf(buf, "shmem=%d", shmem);
    return String("KernelConfiguration(gridSize=") + gridSize.toString() + ",blockSize=" + blockSize.toString() + "," + buf + ")";
  }
}
