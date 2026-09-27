# Source-Aware Verifier Experiments

This repository contains code and cached result artifacts for experiments on
fixed-candidate verification, verifier-source dependence, active checking, and
finite-calibration decision rules.

The repository intentionally does not include the manuscript PDF, manuscript
source, submission forms, private cache files, or API credentials.

## Contents

- `experiments/run_verifier_dependence_pilot.py`: controlled verifier
  dependence and active-checking studies.
- `experiments/run_shared_misunderstanding_real.py`: executable
  fixed-candidate replay with shared versus requirement-conditioned
  verification.
- `experiments/run_llm_shared_misunderstanding.py`: hosted-model candidate and
  verifier study. The committed JSON file can be inspected without API access;
  live reruns require an OpenAI-compatible endpoint.
- `experiments/run_prospective_issue_benchmark.py`: public pull-request
  benchmark fixture and scoring script.
- `experiments/run_mined_issue_fixture.py`,
  `experiments/run_cross_repo_source_pattern_fixture.py`, and
  `experiments/run_public_history_prior_stress.py`: public-history replay and
  cross-repository source-pattern stress tests.
- `experiments/results/`: cached JSON summaries and generated LaTeX tables used
  for the reported results.
- `figures/`: rendered PDF figures corresponding to the reported experiments.

## Quick Check

```bash
python3 -m tests.run_smoke
```

The smoke check verifies that the committed cached results contain the expected
main quantities and that no paper PDF is present in the repository.

Equivalent Make target:

```bash
make check
```

This is the safest first command for a fresh clone. It is fully offline and
does not rewrite cached result files.

## Reproducing Cached Tables

The deterministic studies can be regenerated with:

```bash
python3 -m experiments.run_verifier_dependence_pilot
python3 -m experiments.run_shared_misunderstanding_real
python3 -m experiments.run_mined_issue_fixture
python3 -m experiments.run_cross_repo_source_pattern_fixture
python3 -m experiments.run_public_history_prior_stress
python3 -m experiments.generate_locked_public_history_replay
```

These commands rewrite cached JSON and LaTeX table artifacts under
`experiments/results/`. Run them from a clean working tree if you want to compare
regenerated outputs against the committed release artifacts.

The hosted-model and public-source scripts can also be rerun, but external
service behavior, repository state, and issue availability may change over time.
The cached JSON files are included so that the reported numerical summaries
remain inspectable even when live services drift.

## Optional Hosted-Model Reruns

Live hosted-model reruns use an OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_COMPATIBLE_API_KEY='...'
export OPENAI_COMPATIBLE_ENDPOINT='https://example.com/v1/chat/completions'
python3 -m experiments.run_llm_shared_misunderstanding --live --model your-model
```

Alternatively, pass `--auth-source path/to/key.txt`. Do not commit credentials
or private cache files.
