"""
LangGraph ReAct型エージェントシステム - メイン実行スクリプト
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

from workflow import create_workflow
from agent.supervisor_agent import Supervisor


def main():
    """メイン関数"""
    # 環境変数をロード
    load_dotenv()
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY が設定されていません。")
        print(".envファイルに OPENAI_API_KEY を設定してください。")
        sys.exit(1)
    
    print("=== LangGraph ReAct型エージェントシステム ===")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 実行モードの選択
    print("実行モードを選択してください:")
    print("1. LangGraphワークフロー（推奨）")
    print("2. Supervisorダイレクト実行")
    print("3. テスト実行（売掛金消込の例）")
    
    mode = input("\n選択 (1-3): ").strip()
    
    if mode == "1":
        # LangGraphワークフローモード
        run_langgraph_workflow()
    elif mode == "2":
        # Supervisorダイレクトモード
        run_supervisor_direct()
    elif mode == "3":
        # テスト実行モード
        run_test_example()
    else:
        print("無効な選択です。")
        sys.exit(1)


def run_langgraph_workflow():
    """LangGraphワークフローを実行"""
    print("\n--- LangGraphワークフローモード ---")
    
    # ユーザーからの指示を取得
    print("\n実行したいタスクを入力してください:")
    print("（例: 売掛金と入金データの消込処理を実行してください）")
    instruction = input("\n指示: ").strip()
    
    if not instruction:
        print("指示が入力されていません。")
        return
    
    # ワークフローを作成・実行
    workflow = create_workflow()
    result = workflow.run(instruction)
    
    # 結果を表示
    print_results(result)


def run_supervisor_direct():
    """Supervisorを直接実行"""
    print("\n--- Supervisorダイレクトモード ---")
    
    # ユーザーからの指示を取得
    print("\n実行したいタスクを入力してください:")
    print("（例: 売掛金と入金データの消込処理を実行してください）")
    instruction = input("\n指示: ").strip()
    
    if not instruction:
        print("指示が入力されていません。")
        return
    
    # Supervisorを作成・実行
    supervisor = Supervisor()
    result = supervisor.execute_workflow(instruction)
    
    # 結果を表示
    print_results(result)


def run_test_example():
    """テスト用の例を実行"""
    print("\n--- テスト実行モード ---")
    print("売掛金消込処理のテストを実行します。")
    
    # テスト用の指示
    test_instruction = """
    以下のタスクを実行してください：
    1. data/inputディレクトリから売掛金データ（billing_202505.csv）を探してください
    2. data/inputディレクトリから入金データ（deposit_202506.csv）を探してください
    3. 見つかったデータファイルを使用して消込処理を実行してください
    4. 消込結果をdata/outputディレクトリに保存してください
    """
    
    # ワークフローを作成・実行
    workflow = create_workflow()
    result = workflow.run(test_instruction.strip())
    
    # 結果を表示
    print_results(result)


def print_results(result):
    """実行結果を表示"""
    print("\n=== 実行結果 ===")
    print(f"最終ステータス: {result.process_status.value}")
    print(f"総タスク数: {len(result.task_plan)}")
    print(f"完了タスク数: {len(result.completed_tasks)}")
    
    if result.task_plan:
        print("\n--- タスク一覧 ---")
        for task_id, task in result.task_plan.items():
            status_mark = "✓" if task.status.value == "completed" else "✗"
            print(f"{status_mark} {task.name} (ID: {task_id[:8]}...) - {task.status.value}")
    
    if result.artifacts:
        print("\n--- 生成された成果物 ---")
        for key, value in result.artifacts.items():
            print(f"- {key}: {type(value).__name__}")
            if isinstance(value, str) and os.path.exists(value):
                print(f"  ファイルパス: {value}")
    
    print("\n実行が完了しました。")


def run_cli():
    """CLIモードで実行（引数付き）"""
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
        print(f"指示: {instruction}")
        
        workflow = create_workflow()
        result = workflow.run(instruction)
        print_results(result)
    else:
        main()


if __name__ == "__main__":
    # コマンドライン引数がある場合はCLIモード
    if len(sys.argv) > 1 and sys.argv[1] != "-i":
        run_cli()
    else:
        # インタラクティブモード
        main() 