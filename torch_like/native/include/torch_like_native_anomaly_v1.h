#ifndef TORCH_LIKE_NATIVE_ANOMALY_ABI_V1_H
#define TORCH_LIKE_NATIVE_ANOMALY_ABI_V1_H

/* Versioned C ABI for trusted native anomaly-score plugins.
 * - C calling convention; C++ implementations must export extern "C".
 * - Inputs and output memory are owned by the caller, valid during each call.
 * - Dense row-major double data: [windows][time][channels].
 * - Output scores have EXACTLY the same shape and element count as input.
 * - No host/Python heap allocation may be freed by the plugin.
 * - Every function other than abi_version and destroy returns 0 on success.
 * - Plugin state byte format is plugin-defined; the host treats it as opaque.
 *
 * v1 deliberately excludes classification labels, RUL values, GPU buffers,
 * sparse tensors, hidden input broadcasting, calibration and data provenance.
 */

#include <stdint.h>

#if defined(_WIN32)
#  define TL_NATIVE_EXPORT __declspec(dllexport)
#else
#  define TL_NATIVE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TL_NATIVE_ANOMALY_ABI_VERSION 1u

typedef struct tl_native_input_v1 {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t windows;
    uint64_t time_steps;
    uint64_t channels;
    const double *values;
} tl_native_input_v1;

typedef struct tl_native_output_v1 {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t capacity;
    double *scores;
} tl_native_output_v1;

TL_NATIVE_EXPORT uint32_t tl_native_abi_version(void);
TL_NATIVE_EXPORT int32_t tl_native_create(const char *config_utf8_json, void **out_handle);
TL_NATIVE_EXPORT int32_t tl_native_fit(void *handle, const tl_native_input_v1 *reference);
TL_NATIVE_EXPORT int32_t tl_native_score(
    void *handle, const tl_native_input_v1 *input, tl_native_output_v1 *output
);
TL_NATIVE_EXPORT int32_t tl_native_state_size(void *handle, uint64_t *out_size);
TL_NATIVE_EXPORT int32_t tl_native_save_state(void *handle, uint8_t *out_bytes, uint64_t capacity);
TL_NATIVE_EXPORT int32_t tl_native_load_state(void *handle, const uint8_t *bytes, uint64_t size);
TL_NATIVE_EXPORT void tl_native_destroy(void *handle);

#ifdef __cplusplus
}
#endif
#endif
