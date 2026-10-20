# One-time stable algorithm source Release request

Tag: `algorithms-v0.1.0`

Exact algorithm implementation snapshot: `5a6b15e15f5874b8558145bbffc4df8c1a3133d8`.

Publisher workflow introduced on Main by PR #23: `.github/workflows/release-algorithms-source.yml`.

Release must attach the independently installable *source distribution*
`torch_like_algorithms-0.1.0.tar.gz`, matching Wheel,
`stable-manifest.json`, and `SHA256SUMS`.

This file is only a one-time push trigger. The release workflow checks out
the pinned source SHA; the marker is intentionally excluded from published
algorithm-only sources and the immutable tag target.
