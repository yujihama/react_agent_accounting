# LangGraph ReAct型エージェントシステム

## 概要

このシステムは、LangGraphフレームワークを使用した階層型エージェントワークフローの実装です。複雑なビジネスタスク（例：売掛金の消込処理）を、複数の自律的なAIエージェントが協調して解決します。

## アーキテクチャ

### 主要コンポーネント

1. **Supervisor（オーケストレーター）**
   - ワークフロー全体の進行を管理
   - タスクの割り当てと進行管理
   - 動的なエージェント選択

2. **BaseTaskAgent（基底クラス）**
   - 全ての専門エージェントの共通機能を提供
   - ReActループの実装
   - Supervisorとの通信機能

3. **専門エージェント**
   - **InstructionPlanner**: ユーザー指示の解釈とタスク計画生成
   - **FileSearchExpert**: ファイルシステムの検索と特定
   - **ReconciliationExpert**: 財務データの消込処理

## セットアップ

### 1. 環境変数の設定

`.env`ファイルを作成し、OpenAI APIキーを設定します：

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な実行

```bash
python main.py
```

実行モードを選択：
1. LangGraphワークフロー（推奨）
2. Supervisorダイレクト実行
3. テスト実行（売掛金消込の例）

### コマンドライン実行

```bash
python main.py "売掛金と入金データの消込処理を実行してください"
```

### プログラムからの使用

```python
from workflow import create_workflow

# ワークフローを作成
workflow = create_workflow()

# 指示を実行
result = workflow.run("売掛金と入金データの消込処理を実行してください")

# 結果を確認
print(f"ステータス: {result.process_status}")
print(f"成果物: {result.artifacts}")
```

## ディレクトリ構造

```
.
├── agent/                      # エージェント実装
│   ├── states.py              # 状態定義
│   ├── supervisor_agent/      # Supervisorエージェント
│   └── task_agent/           # タスクエージェント
│       ├── base/             # BaseTaskAgent
│       ├── instruction_planner.py
│       ├── file_search_expert.py
│       └── reconciliation_expert.py
├── config/                    # 設定ファイル
│   └── common_config.yaml    # 共通設定
├── workflow/                  # LangGraphワークフロー
│   └── langgraph_workflow.py
├── data/                      # データディレクトリ
│   ├── input/                # 入力ファイル
│   ├── output/               # 出力ファイル
│   └── instructions/         # 手順書
└── main.py                    # メインスクリプト
```

## 設定のカスタマイズ

`config/common_config.yaml`で以下の設定を変更できます：

- LLMモデル（デフォルト: gpt-4.1-mini）
- 温度パラメータ
- ReActループの最大反復回数
- 各エージェント固有の設定

## エージェントの拡張

新しい専門エージェントを追加する場合：

1. `BaseTaskAgent`を継承したクラスを作成
2. `get_tools()`メソッドでツールを定義
3. `get_agent_description()`メソッドで説明を記述
4. Supervisorのレジストリに追加

```python
class NewExpert(BaseTaskAgent):
    def get_agent_description(self) -> str:
        return "新しい専門家の説明"
    
    def get_tools(self) -> List[Tool]:
        return [
            Tool(name="custom_tool", func=self._custom_tool, description="...")
        ]
```

## トラブルシューティング

- **OPENAI_API_KEY エラー**: `.env`ファイルにAPIキーが設定されているか確認
- **インポートエラー**: `pip install -r requirements.txt`を実行
- **ファイルが見つからない**: データディレクトリのパスを確認

## ライセンス

このプロジェクトは内部使用を目的としています。 