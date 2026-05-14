# tests/test_smoke.py
import os
os.environ.setdefault("PYTHONHASHSEED", "0")

from bilangsim import BiLangModel


def test_model_constructs():
    model = BiLangModel(200, num_clusters=1)
    assert model.schedule.get_agent_count() > 0


def test_model_runs_short():
    model = BiLangModel(200, num_clusters=1)
    model.run_model(3)
    assert model.schedule.steps == 3


def test_warmup_runs_before_collection():
    model = BiLangModel(200, num_clusters=1, warmup_steps=5)
    model.run_model(3)
    # warmup advances the schedule but is not part of the collected run
    assert model.schedule.steps == 8
    assert len(model.data_process.model_vars['pct_bil']) == 3
