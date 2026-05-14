# tests/test_storage.py
import os
os.environ.setdefault("PYTHONHASHSEED", "0")

from bilangsim import BiLangModel
from bilangsim.dataprocess import DataProcessor


def test_parquet_results_roundtrip(tmp_path):
    model = BiLangModel(200, num_clusters=1)
    model.run_model(4, save_data_freq=2, save_dir=str(tmp_path))
    model_df, agent_df = DataProcessor.load_model_data(save_dir=str(tmp_path))
    assert model_df is not None and len(model_df) == 4
    assert agent_df is not None and len(agent_df) > 0
    assert not any(f.endswith('.h5') for f in os.listdir(tmp_path))
