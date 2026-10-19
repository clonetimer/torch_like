# R4b — Explicit candidate source evidence, not causal root cause

FaultSourceRulebook is an expert-reviewed, versioned mapping between an ordered
fault-type label vocabulary and physical candidate source IDs. It supports
ambiguous and many-to-many associations: one fault type may originate from
multiple components, and one component may be associated with several fault
types. The healthy/normal class must have no fault-source mapping.

FaultSourceEvidenceMapper.predict accepts only a fault_classification
DiagnosticResult containing a complete fixed-order class-probability matrix.
It rejects reordered labels or invalid normalization. For each candidate
source, its support is the maximum classifier probability among associated
fault-type classes. The result has semantic task fault_isolation and output
kind fault_source_score, with explicit source names. No training or ground truth
is accessed by the mapper.

The scores are NOT mutually exclusive class probabilities and are NOT
posterior root-cause probabilities. They deliberately need not sum to one;
multiple candidate sources receive equal evidence for ambiguous fault types.
The output is only a transparent candidate list for engineering review.

The caller must provide the association table and its provenance. There is no
implicit fault-to-component inference, physical-causal model, FMEA derivation,
calibrated confidence, unknown-fault reasoning, nor propagation/coupled-fault
diagnosis. Each of these requires separate evidence and validation.

Tests check many-to-many mapping mathematics, schema ordering, immutable
rulebook snapshot, invalid associations, bad probability normalization,
and explicit factory registration.

This R4b building block is independent of R4a's classifier implementation:
it consumes the stable existing DiagnosticResult semantics only.
