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
  //using namespace thrust;

  constexpr int N = 100;
  thrust::device_vector<float> data(N, 0);

  vector<string> event_names {"active_warps"};
  vector<string> metric_names {"flop_count_dp",
                               "flop_count_sp",
                               "inst_executed"};

  cupti_profiler::profiler profiler(event_names, metric_names);

  // Get #passes required to compute all metrics
  const int passes = profiler.get_passes();

  profiler.start();
  for(int i=0; i<passes; ++i) {
    call_kernel(data);
  }
  profiler.stop();

  profiler.print_event_values();
  profiler.print_metric_values();

  thrust::host_vector<float> h_data(data);

  for(int i = 0; i < 10; ++i) {
    printf("%lf ", h_data[i]);
  }
  printf("\n");
  return 0;
}
