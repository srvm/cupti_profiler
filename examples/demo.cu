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
__global__ void kernel2(T begin, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < size)
    *(begin + thread_id) += 2;
}

template<typename T>
void call_kernel(T& arg) {
  kernel<<<1, 100>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

template<typename T>
void call_kernel2(T& arg) {
  kernel2<<<1, 50>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

int main() {
  using namespace std;
  //using namespace thrust;

  vector<string> event_names {
                              "active_warps",
                              "gst_inst_32bit",
                              "active_cycles",
                              "threads_launched",
                              "branch"
                             };
  vector<string> metric_names {
                               "flop_count_dp",
                               "flop_count_sp",
                               "inst_executed",
                               "gst_transactions",
                               "gld_transactions"
                               //"stall_memory_throttle"
                              };

  constexpr int N = 100;
  thrust::device_vector<float> data(N, 0);

  //cupti_profiler::profiler profiler(vector<string>{}, metric_names);

  // XXX: Disabling all metrics seems to change the values
  // of some events. Not sure if this is correct behavior.
  //cupti_profiler::profiler profiler(event_names, vector<string>{});

  cupti_profiler::profiler profiler(event_names, metric_names);
  // Get #passes required to compute all metrics and events
  const int passes = profiler.get_passes();
  printf("Passes: %d\n", passes);

  profiler.start();
  for(int i=0; i<50; ++i) {
    call_kernel(data);
    cudaDeviceSynchronize();
    call_kernel2(data);
    cudaDeviceSynchronize();
  }
  profiler.stop();

  printf("Event Trace\n");
  profiler.print_event_values(std::cout);
  printf("Metric Trace\n");
  profiler.print_metric_values(std::cout);

  thrust::host_vector<float> h_data(data);

  /*printf("\n");
  for(int i = 0; i < 10; ++i) {
    printf("%.2lf ", h_data[i]);
  }*/
  printf("\n");
  return 0;
}
