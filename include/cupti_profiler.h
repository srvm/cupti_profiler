#pragma once

#include <vector>
#include <string>
#include <cassert>

#include <cupti.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

namespace cupti_profiler {
namespace detail {

  // User data for event collection callback
  struct pass_data_t {
    // the device where metric is being collected
    CUdevice device;
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet *event_groups;
    // the current number of events collected in eventIdArray and
    // eventValueArray
    uint32_t current_event_idx;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t num_events;
    // array of event ids
    std::vector<CUpti_EventID> event_ids;
    // array of event values
    std::vector<uint64_t> event_values;
  };

  void CUPTIAPI
  get_value_callback(void *userdata,
                     CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid,
                     const CUpti_CallbackData *cbInfo) {
    static int current_pass = 0;

    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      printf("%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
      exit(-1);
    }

    std::vector<detail::pass_data_t> *pass_vector =
      (std::vector<detail::pass_data_t> *)userdata;
    detail::pass_data_t *pass_data = &(*pass_vector)[current_pass];
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      //printf("In Callback: Enter\n");
      printf("Pass number: %d\n", current_pass);
      cudaDeviceSynchronize();

      CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
            CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      for (int i = 0; i < pass_data->event_groups->numEventGroups; i++) {
        printf("  Enabling group %d\n", i);
        uint32_t all = 1;
        CUPTI_CALL(cuptiEventGroupSetAttribute(
              pass_data->event_groups->eventGroups[i],
              CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
              sizeof(all), &all));
        CUPTI_CALL(cuptiEventGroupEnable(
              pass_data->event_groups->eventGroups[i]));
      }
    } else if(cbInfo->callbackSite == CUPTI_API_EXIT) {
      cudaDeviceSynchronize();

      for (int i = 0; i < pass_data->event_groups->numEventGroups; i++) {
        CUpti_EventGroup group = pass_data->event_groups->eventGroups[i];
        CUpti_EventDomainID group_domain;
        uint32_t numEvents, numInstances, numTotalInstances;
        CUpti_EventID *eventIds;
        size_t groupDomainSize = sizeof(group_domain);
        size_t numEventsSize = sizeof(numEvents);
        size_t numInstancesSize = sizeof(numInstances);
        size_t numTotalInstancesSize = sizeof(numTotalInstances);
        uint64_t *values, normalized, sum;
        size_t valuesSize, eventIdsSize;

        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
              &groupDomainSize, &group_domain));
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
              pass_data->device, group_domain,
              CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
              &numTotalInstancesSize, &numTotalInstances));
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
              &numInstancesSize, &numInstances));
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
              &numEventsSize, &numEvents));
        eventIdsSize = numEvents * sizeof(CUpti_EventID);
        eventIds = (CUpti_EventID *)malloc(eventIdsSize);
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENTS,
              &eventIdsSize, eventIds));

        valuesSize = sizeof(uint64_t) * numInstances;
        values = (uint64_t *)malloc(valuesSize);

        for(int j = 0; j < numEvents; j++) {
          CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                     eventIds[j], &valuesSize, values));
          /*if (metric_data->eventIdx >= metric_data->numEvents) {
            fprintf(stderr, "[error]: Too many events collected, metric expects only %d\n",
                (int)metric_data->numEvents);
            exit(-1);
          }*/

          // sum collect event values from all instances
          sum = 0;
          for(int k = 0; k < numInstances; k++)
            sum += values[k];

          // normalize the event value to represent the total number of
          // domain instances on the device
          normalized = (sum * numTotalInstances) / numInstances;

          pass_data->event_ids.push_back(eventIds[j]);
          pass_data->event_values.push_back(normalized);

          // print collected value
          {
            char eventName[128];
            size_t eventNameSize = sizeof(eventName) - 1;
            CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                       &eventNameSize, eventName));
            eventName[127] = '\0';
            printf("\t%s = %llu (", eventName, (unsigned long long)sum);
            if (numInstances > 1) {
              for (int k = 0; k < numInstances; k++) {
                if (k != 0)
                  printf(", ");
                printf("%llu", (unsigned long long)values[k]);
              }
            }

            printf(")\n");
            printf("\t%s (normalized) (%llu * %u) / %u = %llu\n",
                eventName, (unsigned long long)sum,
                numTotalInstances, numInstances,
                (unsigned long long)normalized);
          }
        }

        free(values);
      }

      for (int i = 0; i < pass_data->event_groups->numEventGroups; i++) {
        printf("  Disabling group %d\n", i);
        CUPTI_CALL(cuptiEventGroupDisable(
                   pass_data->event_groups->eventGroups[i]));
      }
      ++current_pass;
      //printf("In Callback: Exit\n");
    }
  }

} // namespace detail

  struct profiler {
    typedef std::vector<std::string> strvec_t;
    typedef std::vector<double> val_t;

    profiler(const strvec_t& events,
             const strvec_t& metrics,
             const int device_num=0) :
      m_event_names(events),
      m_metric_names(metrics),
      m_device_num(0),
      m_num_metrics(metrics.size()),
      m_num_events(events.size()) {

      int device_count = 0;

      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
      DRIVER_API_CALL(cuInit(0));
      DRIVER_API_CALL(cuDeviceGetCount(&device_count));
      if (device_count == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(1);
      }

      m_metric_id.resize(m_num_metrics);

      DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
      DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
      CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                 (CUpti_CallbackFunc)detail::get_value_callback,
                 &m_data));
      CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

      CUpti_MetricID metric_ids[m_num_metrics];
      for(int i = 0; i < m_num_metrics; ++i) {
        CUPTI_CALL(cuptiMetricGetIdFromName(m_device,
                   m_metric_names[i].c_str(),
                   &metric_ids[i]));
      }
      CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context,
                 sizeof(metric_ids), metric_ids, &m_pass_data));
      m_passes = m_pass_data->numSets;
      m_data.resize(m_passes);

      for(int i = 0; i < m_passes; ++i) {
        int total_events = 0;
        printf("Looking at set (pass) %d\n", i);
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < m_pass_data->sets[i].numEventGroups; ++j) {
          CUPTI_CALL(cuptiEventGroupGetAttribute(
                m_pass_data->sets[i].eventGroups[j],
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &num_events_size, &num_events));
          printf("  Event Group %d, #Events = %d\n", j, num_events);
          total_events += num_events;
        }
        m_data[i].event_groups = m_pass_data->sets + i;
        m_data[i].device = m_device;
        m_data[i].num_events = total_events;
      }
      std::copy(metric_ids, metric_ids + m_num_metrics,
                m_metric_id.begin());
    }

    ~profiler() {
    }

    int get_passes()
    { return m_passes; }

    void start() {
    }

    void stop() {
      int total_events = 0;
      for(int i = 0; i < m_passes; ++i) {
        /*printf("#Event IDs for Pass %d: %d\n",
               i, m_data[i].num_events);*/
        total_events += m_data[i].num_events;
      }
      CUpti_MetricValue metric_value;
      CUpti_EventID *event_ids = new CUpti_EventID[total_events];
      uint64_t *event_values = new uint64_t[total_events];

      int running_sum = 0;
      for(int i = 0; i < m_passes; ++i) {
        std::copy(m_data[i].event_ids.begin(),
                  m_data[i].event_ids.end(),
                  event_ids + running_sum);
        std::copy(m_data[i].event_values.begin(),
                  m_data[i].event_values.end(),
                  event_values + running_sum);
        running_sum += m_data[i].num_events;
      }
      assert(running_sum == total_events);

      for(int i = 0; i < m_num_metrics; ++i) {
        CUPTI_CALL(cuptiMetricGetValue(m_device, m_metric_id[i],
                   total_events * sizeof(CUpti_EventID),
                   event_ids,
                   total_events * sizeof(uint64_t),
                   event_values,
                   0, &metric_value));
        m_metrics.push_back((double)metric_value.metricValueUint64);
      }

      delete[] event_ids;
      delete[] event_values;

      CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    void print_event_values() {
    }

    void print_metric_values() {
      for(int i = 0; i < m_num_metrics; ++i) {
        printf("metric [%s], value = %lf\n",
               m_metric_names[i].c_str(),
               m_metrics[i]);
      }
      printf("\n");
    }

    val_t get_event_values()
    { return m_events; }

    val_t get_metric_values()
    { return m_metrics; }

  private:
    int m_device_num;
    const strvec_t& m_event_names;
    const strvec_t& m_metric_names;
    int m_num_metrics, m_num_events;
    val_t m_events;
    val_t m_metrics;

    int m_passes;

    CUcontext m_context;
    CUdevice m_device;
    CUpti_SubscriberHandle m_subscriber;
    std::vector<detail::pass_data_t> m_data;
    CUpti_EventGroupSets *m_pass_data;
    std::vector<CUpti_MetricID> m_metric_id;
  };

} // namespace cupti_profiler
