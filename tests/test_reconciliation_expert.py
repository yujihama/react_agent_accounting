import os
import sys
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.task_agent.reconciliation_expert import ReconciliationExpert


def test_load_and_output(tmp_path, monkeypatch):
    """_load_data と _output_csv の基本動作をテスト"""
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    expert = ReconciliationExpert()
    df = expert._load_data("tests/data/deposit_test.csv")
    assert not df.empty

    out_file = tmp_path / "result.csv"
    message = expert._output_csv(df, str(out_file))

    assert out_file.exists()
    assert "ファイルを出力しました" in message
