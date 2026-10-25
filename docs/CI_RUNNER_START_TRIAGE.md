# CI runner-start failure — P0 evidence and triage

**Status:** unresolved infrastructure/account/runner-start gate, not a tested
algorithm regression. Do not label failed jobs "passed" or merge a stable
candidate to bypass the gate.

## Reproducible evidence (2026-10-25 UTC)

- Unchanged main revision `580133dddf61cce7c354c648d4885e6caec99418`:
  Quality `40203128634` and DDP `40203128902` both returned failure.
- Stable v0.3.0 candidate revision `867fab12c01d2283b98593fa4dc7274002e2c136`:
  Quality `40203418098`, DDP `40203418074`, and stable-wheel
  `40203418150` returned failure.
- For the affected jobs, `GET /actions/runs/{id}/jobs` returns a completed
  failure with `steps=null`, including on earlier failure retry. Attempted
  job-log download returned an unavailable/nonexistent blob.
- The failure is shared by code-unmodified main and three independent
  workflows and occurs before any shell or dependency installation step.

**Known:** testing did not run. **Unknown:** whether GitHub hosted-runner
availability, repository/org spending limit, account billing, job-queue
capacity, disabled Actions, or enforcement policies caused the failure.
No runner-level error text was accessible from the connected API.

## Required owner-side read-only checks

1. Open the failed runs on GitHub; inspect their run banners/check annotations,
   especially the message shown *above* jobs. A zero-step job failure may
   contain a billing, usage or policy notification outside the runner log.
2. Check repository Settings → Actions → General for Actions enablement,
   allowed workflows and runner policies. On an organization repository also
   check organization settings; avoid changing governance without approval.
3. Check Settings → Billing / Actions usage for monthly minutes/spending
   limits, payment problems and hosted-runner allotments. For private
   repositories, runner eligibility and paid usage may apply.
4. Check GitHub Status for contemporaneous **Actions / Hosted Runners**
   incidents; historical incidents are not proof of this run's cause.
5. If the banner is blank and billing/policy are healthy, provide the five
   run IDs above to GitHub Support. Capture screenshots of the web banners.

Do **not** try to make jobs green by deleting required workflows, removing
coverage thresholds or adding `continue-on-error`.

## CI-independent verification path

If Python with installed NumPy, scikit-learn, PyTorch, setuptools, wheel and
`build` is available on a trusted local Linux/Windows workstation, run:

```shell
python -m pip install build wheel
python packaging/stable/verify_local.py --report stable-v0.3.0-local-report.json
```

The command stages the allowlisted algorithms, builds a fresh v0.3.0 wheel,
installs it in a temporary virtual environment using preinstalled system
dependencies, executes the installed-wheel smoke outside the source tree and
writes a JSON receipt. This is a **fallback test**, not a substitute for
the dedicated GitHub Actions checks or a proof that dependencies are isolated.

## Exit criteria

- First, a hosted-runner job for unchanged main **starts and completes actual
  steps**, not simply a check marked green.
- Then Quality, DDP and Stable Wheel execute successfully against the *exact*
  candidate PR head and report actual steps/artifacts.
- Installed-wheel smoke covers the v0.3 contracts and the existing 0.2 API
  without bringing full-platform features into the slim wheel.
- Only after all evidence is available, mark the PR ready, merge with an
  expected head SHA and consider publishing a new, explicitly versioned
  release. The published v0.2.0 artifact is immutable.
