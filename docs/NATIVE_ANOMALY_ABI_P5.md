# P5 — Optional native C ABI anomaly plugin (first supported slice)

**Scope:** industrial C/C++ anomaly **score** algorithms with dense
`float64 (N,L,C)` data. This is a versioned, opt-in interoperability layer,
not an alternative PyTorch, experiment runner, UI or device communication
service. It is not included in the previously released algorithm-only 0.2.0
wheel. Existing algorithms and `torch_like.diagnostics` remain unchanged.

## v1 contract

The full framework ships
`torch_like/native/include/torch_like_native_anomaly_v1.h`, a portable
`extern "C"` header usable from C and C++:

- `tl_native_abi_version()` returns exactly 1.
- `tl_native_create(config_json, &handle)` and
  `tl_native_destroy(handle)` own the plug-in's opaque model lifetime.
- `tl_native_fit(handle, input)` receives reference windows chosen by the
  caller.
- `tl_native_score(handle, input, output)` writes finite score values
  **exactly matching (N,L,C)** into a preallocated caller-owned buffer.
- `tl_native_state_size/save_state/load_state` support plug-in-owned
  serializable state with explicit byte size and a host-side size cap.
- Every status-returning function uses 0 for success; negative/nonzero
  values fail closed through a Python exception.

No cross-runtime allocation is freed by the wrong side. There is no silent
flattening of multichannel telemetry or guessing output semantics. The
calling convention is C/cdecl on Windows and Linux. Every plugin must handle
invalid sizes and pointers defensively and be tested against that compiler
and platform.

## Minimal compiled demonstration

```shell
gcc -std=c11 -O2 -shared -fPIC \
    -I torch_like/native/include \
    examples/native/demo_mean_deviation.c \
    -o libtl_mean_dev.so -lm
```

Windows MinGW:

```shell
gcc -std=c11 -O2 -shared \
    -I torch_like/native/include \
    examples/native/demo_mean_deviation.c \
    -o tl_mean_dev.dll
```

The C demo is **not** a production industrial detector: it learns one
global mean and returns absolute distance from that mean. It exists to
validate memory ownership, version handshake and checkpoint calls.

## Python usage

```python
from torch_like.native import NativeAnomalyDetector

with NativeAnomalyDetector(
    "./libtl_mean_dev.so", algorithm_name="demo_mean_dev",
    config={"seed": 5},
) as detector:
    detector.fit(healthy_signal_batch)
    result = detector.score(new_signal_batch)
    snapshot = detector.export_state()  # JSON-serializable host envelope

with NativeAnomalyDetector(
    "./libtl_mean_dev.so", algorithm_name="demo_mean_dev",
    config={"seed": 5},
) as another:
    another.import_state(snapshot)
    restored_result = another.score(new_signal_batch)
```

The wrapper stores sample rate/channel identity and units separately from
the plug-in's opaque bytes. A mismatch fails before scoring. Native scores
return a standard `DiagnosticResult`. Alarm thresholds and labeling remain
separate algorithm decisions; classification, RUL and GPU buffers are **not**
represented by ABI v1.

## Trust, compatibility and limitations

**Only load locally trusted and verified binaries.** Loading a DLL/SO
executes arbitrary native code in the application process. This wrapper
does not sandbox plugins or audit their supply chain. In production,
applications should control allowlists, signatures/hashes, and optionally
isolate native execution in a separate process.

`import_state` validates the host manifest first but cannot guarantee
atomicity inside arbitrary native implementations. The plugin must treat
failed state loads transactionally, and the caller should reinstantiate if
a native load returns error. A binary ABI version of 1 does *not* imply
cross-plugin, cross-compiler, or cross-OS compatibility of opaque checkpoint
bytes. The C demo's raw-struct checkpoint is for same-build local tests,
not a portable persistence format.

The v1 ABI intentionally excludes complex `fit` target semantics, sparse
tensors, GPU memory, multiple typed output heads, hidden feature pipelines,
process isolation and threading guarantees. Do not claim all native
algorithms are supported. Broader typed tasks must receive a new versioned
ABI with explicit compatibility gates.

## Quality gates

`tests/test_native_abi.py` compiles the demonstration native plugin
when a C compiler is present, verifies version negotiation, C
fit-score behavior, JSON checkpoint reloading, caller channel metadata
and close/error handling. Automated Windows compiler/loader validation
must be added and passed before declaring cross-platform Stable support.
