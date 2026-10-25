/* Demonstration reference plug-in for the torch_like trusted native anomaly ABI.
 * A simple fitted global-mean absolute deviation, NOT an industrial detector.
 * gcc -std=c11 -O2 -shared -fPIC -I torch_like/native/include
 *     examples/native/demo_mean_deviation.c -o libtl_mean_dev.so
 */
#include "torch_like_native_anomaly_v1.h"
#include <math.h>
#if defined(_MSC_VER)
#  include <float.h>
#  define TL_ISFINITE(x) _finite(x)
#else
#  define TL_ISFINITE(x) isfinite(x)
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double mean;
    uint64_t channels;
    uint32_t fitted;
} demo_state;

TL_NATIVE_EXPORT uint32_t tl_native_abi_version(void) {
#ifdef TL_FORCE_BAD_VERSION
    return 999u;
#else
    return TL_NATIVE_ANOMALY_ABI_VERSION;
#endif
}

TL_NATIVE_EXPORT int32_t tl_native_create(const char *json, void **out_handle) {
    if (!json || !out_handle) return -1;
    demo_state *s = (demo_state *)calloc(1, sizeof(demo_state));
    if (!s) return -2;
    *out_handle = s;
    return 0;
}

TL_NATIVE_EXPORT int32_t tl_native_fit(void *handle, const tl_native_input_v1 *input) {
    if (!handle || !input || !input->values || input->abi_version != TL_NATIVE_ANOMALY_ABI_VERSION)
        return -1;
    if (!input->windows || !input->time_steps || !input->channels) return -2;
    if (input->windows > UINT64_MAX / input->time_steps) return -3;
    uint64_t length = input->windows * input->time_steps;
    if (length > UINT64_MAX / input->channels) return -3;
    length *= input->channels;
    double total = 0.0;
    for (uint64_t i = 0; i < length; ++i) {
        if (!TL_ISFINITE(input->values[i])) return -4;
        total += input->values[i];
    }
    if (!isfinite(total)) return -4;
    demo_state *s = (demo_state *)handle;
    s->mean = total / (double)length;
    s->channels = input->channels;
    s->fitted = 1u;
    return 0;
}

TL_NATIVE_EXPORT int32_t tl_native_score(
    void *handle, const tl_native_input_v1 *input, tl_native_output_v1 *output
) {
    if (!handle || !input || !output || !input->values || !output->scores) return -1;
    if (input->abi_version != TL_NATIVE_ANOMALY_ABI_VERSION
        || output->abi_version != TL_NATIVE_ANOMALY_ABI_VERSION) return -2;
    demo_state *s = (demo_state *)handle;
    if (!s->fitted || input->channels != s->channels) return -3;
    if (!input->windows || !input->time_steps || !input->channels) return -4;
    if (input->windows > UINT64_MAX / input->time_steps) return -5;
    uint64_t length = input->windows * input->time_steps;
    if (length > UINT64_MAX / input->channels) return -5;
    length *= input->channels;
    if (output->capacity != length) return -6;
    for (uint64_t i = 0; i < length; ++i)
        output->scores[i] = fabs(input->values[i] - s->mean);
    return 0;
}

TL_NATIVE_EXPORT int32_t tl_native_state_size(void *handle, uint64_t *out_size) {
    if (!handle || !out_size || !((demo_state *)handle)->fitted) return -1;
    *out_size = (uint64_t)sizeof(demo_state);
    return 0;
}

TL_NATIVE_EXPORT int32_t tl_native_save_state(void *handle, uint8_t *out_bytes, uint64_t capacity) {
    if (!handle || !out_bytes || capacity != sizeof(demo_state)) return -1;
    if (!((demo_state *)handle)->fitted) return -2;
    memcpy(out_bytes, handle, sizeof(demo_state));
    return 0;
}

TL_NATIVE_EXPORT int32_t tl_native_load_state(void *handle, const uint8_t *bytes, uint64_t size) {
    if (!handle || !bytes || size != sizeof(demo_state)) return -1;
    demo_state candidate;
    memcpy(&candidate, bytes, sizeof(candidate));
    if (!candidate.fitted || !candidate.channels || !isfinite(candidate.mean)) return -2;
    memcpy(handle, &candidate, sizeof(candidate)); /* atomic after full validation */
    return 0;
}

TL_NATIVE_EXPORT void tl_native_destroy(void *handle) { free(handle); }
