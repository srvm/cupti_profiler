#pragma once

#include <vector>
#include <string>
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
  struct metric_data_t {
    // the device where metric is being collected
    CUdevice device;
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet *eventGroups;
    // the current number of events collected in eventIdArray and
    // eventValueArray
    uint32_t eventIdx;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t numEvents;
    // array of event ids
    CUpti_EventID *eventIdArray;
    // array of event values
    uint64_t *eventValueArray;
  };

  void CUPTIAPI
  get_value_callback(void *userdata,
                     CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid,
                     const CUpti_CallbackData *cbInfo) {
    std::vector<detail::metric_data_t> *metricData_vec =
      (std::vector<detail::metric_data_t> *)userdata;
    detail::metric_data_t *metricData = &(*metricData_vec)[0];

    unsigned int i, j, k;

    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
      exit(-1);
    }

    // on entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      cudaDeviceSynchronize();

      CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
            CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
        uint32_t all = 1;
        CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
              CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
              sizeof(all), &all));
        CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
      }
    }

    // on exit, read and record event values
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      cudaDeviceSynchronize();

      // for each group, read the event values from the group and record
      // in metricData
      for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
        CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
        CUpti_EventDomainID groupDomain;
        uint32_t numEvents, numInstances, numTotalInstances;
        CUpti_EventID *eventIds;
        size_t groupDomainSize = sizeof(groupDomain);
        size_t numEventsSize = sizeof(numEvents);
        size_t numInstancesSize = sizeof(numInstances);
        size_t numTotalInstancesSize = sizeof(numTotalInstances);
        uint64_t *values, normalized, sum;
        size_t valuesSize, eventIdsSize;

        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
              &groupDomainSize, &groupDomain));
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
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

        for (j = 0; j < numEvents; j++) {
          CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                eventIds[j], &valuesSize, values));
          if (metricData->eventIdx >= metricData->numEvents) {
            fprintf(stderr, "error: too many events collected, metric expects only %d\n",
                (int)metricData->numEvents);
            exit(-1);
          }

          // sum collect event values from all instances
          sum = 0;
          for (k = 0; k < numInstances; k++)
            sum += values[k];

          // normalize the event value to represent the total number of
          // domain instances on the device
          normalized = (sum * numTotalInstances) / numInstances;

          metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
          metricData->eventValueArray[metricData->eventIdx] = normalized;
          metricData->eventIdx++;

          // print collected value
          {
            char eventName[128];
            size_t eventNameSize = sizeof(eventName) - 1;
            CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                  &eventNameSize, eventName));
            eventName[127] = '\0';
            printf("\t%s = %llu (", eventName, (unsigned long long)sum);
            if (numInstances > 1) {
              for (k = 0; k < numInstances; k++) {
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

      for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
        CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
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

      m_metric_id.reserve(m_num_metrics);
      m_data.reserve(m_num_metrics);
      m_pass_data.reserve(m_num_metrics);

      DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
      DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
      CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                 (CUpti_CallbackFunc)detail::get_value_callback,
                 &m_data));
      CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

      for(int i=0; i < m_num_metrics; ++i) {
        CUPTI_CALL(cuptiMetricGetIdFromName(m_device,
                   m_metric_names[i].c_str(),
                   &m_metric_id[i]));
        CUPTI_CALL(cuptiMetricGetNumEvents(m_metric_id[i],
                   &m_data[i].numEvents));

        m_data[i].device = m_device;
        m_data[i].eventIdArray = (CUpti_EventID *)malloc(
                              m_data[i].numEvents * sizeof(CUpti_EventID));
        m_data[i].eventValueArray = (uint64_t *)malloc(
                                 m_data[i].numEvents * sizeof(uint64_t));
        m_data[i].eventIdx = 0;
        CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context,
                   sizeof(m_metric_id[i]), &m_metric_id[i], &m_pass_data[i]));
      }
      m_passes = m_pass_data[0]->numSets;
      m_data[0].eventGroups = m_pass_data[0]->sets + m_passes - 1;
      //printf("Max events: %d\n", max_events);
    }

    ~profiler() {
    }

    int get_passes()
    { return m_passes; }

    void start() {
    }

    void stop() {
      if (m_data[0].eventIdx != m_data[0].numEvents) {
        fprintf(stderr, "error: expected %u metric events, got %u\n",
            m_data[0].numEvents, m_data[0].eventIdx);
        exit(-1);
      }
      CUpti_MetricValue metric_value;

      CUPTI_CALL(cuptiMetricGetValue(m_device, m_metric_id[0],
                 m_data[0].numEvents * sizeof(CUpti_EventID),
                 m_data[0].eventIdArray,
                 m_data[0].numEvents * sizeof(uint64_t),
                 m_data[0].eventValueArray,
                 0, &metric_value));

      m_metrics.push_back((double)metric_value.metricValueUint64);
      CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    void print_event_values() {
    }

    void print_metric_values() {
      printf("metric [%s], value = %lf\n",
             m_metric_names[0].c_str(),
             m_metrics[0]);
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
    std::vector<detail::metric_data_t> m_data;
    std::vector<CUpti_EventGroupSets *> m_pass_data;
    std::vector<CUpti_MetricID> m_metric_id;
  };

} // namespace cupti_profiler
