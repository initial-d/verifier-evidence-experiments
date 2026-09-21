.PHONY: smoke controlled executable public-history locked-replay

PYTHON ?= python3

smoke:
	$(PYTHON) -m tests.run_smoke

controlled:
	$(PYTHON) -m experiments.run_verifier_dependence_pilot

executable:
	$(PYTHON) -m experiments.run_shared_misunderstanding_real

public-history:
	$(PYTHON) -m experiments.run_mined_issue_fixture
	$(PYTHON) -m experiments.run_cross_repo_source_pattern_fixture
	$(PYTHON) -m experiments.run_public_history_prior_stress

locked-replay: public-history
	$(PYTHON) -m experiments.generate_locked_public_history_replay

