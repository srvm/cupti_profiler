#include <cstdio>
#include <vector>
#include <string>
#include <thrust/device_vector.h>

#include <cupti_profiler.h>

template<typename T>
__global__ void kernel(T begin, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < size)
    *(begin + thread_id) += 1;
}

template<typename T>
void call_kernel(T& arg) {
  kernel<<<1, 100>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

int main() {
  using namespace std;
  using namespace thrust;

  constexpr int N = 100;
  device_vector<double> data(N, 0);

  vector<string> events {"active_warps"};
  vector<string> metrics {"flop_count_sp"};

  cupti_profiler::profiler profiler(events, metrics);
  const int passes = profiler.get_passes();

  call_kernel(data);

  thrust::host_vector<double> h_data(data);

  for(int i = 0; i < 10; ++i) {
    printf("%lf ", h_data[i]);
  }
  printf("\n");
  return 0;
}
