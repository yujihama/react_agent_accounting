"""
Supervisor - ワークフロー全体の進行を管理する中央制御エージェント
"""

import yaml
from typing import Dict, Any, Optional, List, Type
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..states import GlobalAgentState, BaseTaskAgentState, Task, ProcessStatus, TaskStatus
from ..task_agent import BaseTaskAgent, InstructionPlanner, FileSearchExpert, ReconciliationExpert


class Supervisor:
    """
    ワークフロー全体の進行を管理する中央制御エージェント。
    LLMの判断力を活用し、動的なタスク割り当てを行う。
    """
    
    def __init__(self, config_path: str = "config/common_config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込む
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # LLMの初期化
        self.llm = ChatOpenAI(
            model=self.config['common']['llm_model'],
            temperature=self.config['common']['llm_temperature']
        )
        
        # 利用可能なタスクエージェントのレジストリ
        self.agent_registry: Dict[str, Type[BaseTaskAgent]] = {
            "InstructionPlanner": InstructionPlanner,
            "FileSearchExpert": FileSearchExpert,
            "ReconciliationExpert": ReconciliationExpert
        }
        
        # タスク選択用のプロンプト
        self.task_selection_prompt = self._create_task_selection_prompt()
        
        # エージェント選択用のプロンプト
        self.agent_selection_prompt = self._create_agent_selection_prompt()
    
    def _create_task_selection_prompt(self) -> ChatPromptTemplate:
        """タスク選択用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_messages([
            ("system", """あなたはワークフローの進行を管理するスーパーバイザーです。
現在の状況を分析し、次に実行すべきタスクを選択してください。

現在の状況:
- 完了済みタスク: {completed_tasks}
- 実行可能なタスク: {executable_tasks}
- 全体の進行状況: {process_status}

タスクの依存関係を考慮し、最も適切な次のタスクを選択してください。
選択したタスクのIDのみを返してください。"""),
            ("human", "次に実行すべきタスクを選択してください。")
        ])
    
    def _create_agent_selection_prompt(self) -> ChatPromptTemplate:
        """エージェント選択用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_messages([
            ("system", """あなたはタスクに最適なエージェントを選択するスーパーバイザーです。

タスク情報:
- タスク名: {task_name}
- タスクの説明: {task_description}
- 期待される成果物: {expected_output}

利用可能なエージェント:
- InstructionPlanner: ユーザー指示の解釈とタスク計画の生成に特化
- FileSearchExpert: ファイルシステムの検索と特定に特化
- ReconciliationExpert: 財務データの消込処理に特化

タスクの内容を分析し、最も適切なエージェントを選択してください。
選択したエージェント名のみを返してください。"""),
            ("human", "このタスクに最適なエージェントを選択してください。")
        ])
    
    def execute_workflow(self, initial_instruction: str) -> GlobalAgentState:
        """
        ワークフロー全体を実行
        
        Args:
            initial_instruction: ユーザーからの初期指示
            
        Returns:
            最終的なワークフロー状態
        """
        # 初期状態を作成
        state = GlobalAgentState(
            input_instruction=initial_instruction,
            process_status=ProcessStatus.PLANNING
        )
        
        # ステップ1: タスク計画の生成
        print("=== ステップ1: タスク計画の生成 ===")
        state = self._execute_planning_phase(state)
        
        if state.process_status == ProcessStatus.FAILED:
            return state
        
        # ステップ2: タスクの実行
        print("\n=== ステップ2: タスクの実行 ===")
        state.process_status = ProcessStatus.RUNNING
        state = self._execute_tasks_phase(state)
        
        # 最終ステータスの設定
        if all(task.status == TaskStatus.COMPLETED for task in state.task_plan.values()):
            state.process_status = ProcessStatus.SUCCESS
        else:
            state.process_status = ProcessStatus.FAILED
        
        return state
    
    def _execute_planning_phase(self, state: GlobalAgentState) -> GlobalAgentState:
        """タスク計画フェーズを実行"""
        # InstructionPlannerを使用してタスク計画を生成
        planner = InstructionPlanner(self.config['__file__'] if '__file__' in self.config else 'config/common_config.yaml')
        
        # プランナー用の状態を作成
        planner_state = BaseTaskAgentState(
            task_to_perform=Task(
                name="タスク計画の生成",
                description=f"以下の指示に基づいて具体的なタスク計画を生成してください: {state.input_instruction}",
                expected_output_description="tasks というキーで、タスクのリストを含む辞書を返してください。各タスクは name, description, dependencies, expected_output_description を持つ必要があります。"
            )
        )
        
        # タスク計画を生成
        result_state = planner.execute_task(planner_state)
        
        if result_state.status_report == "completed" and result_state.final_result:
            if "task_plan" in result_state.final_result:
                state.task_plan = result_state.final_result["task_plan"]
                print(f"タスク計画を生成しました: {len(state.task_plan)}個のタスク")
            else:
                print("エラー: タスク計画の生成に失敗しました")
                state.process_status = ProcessStatus.FAILED
        else:
            print("エラー: InstructionPlannerの実行に失敗しました")
            state.process_status = ProcessStatus.FAILED
        
        return state
    
    def _execute_tasks_phase(self, state: GlobalAgentState) -> GlobalAgentState:
        """タスク実行フェーズ"""
        max_iterations = 100  # 無限ループ防止
        iteration = 0
        
        while iteration < max_iterations:
            # 実行可能なタスクを取得
            executable_tasks = state.get_executable_tasks()
            
            if not executable_tasks:
                # 全てのタスクが完了したか、実行不可能
                break
            
            # 次に実行するタスクを選択
            next_task = self._select_next_task(state, executable_tasks)
            if not next_task:
                print("エラー: 次のタスクを選択できませんでした")
                break
            
            print(f"\n--- タスク実行: {next_task.name} ---")
            state.current_task_id = next_task.id
            
            # タスクに適したエージェントを選択
            agent_name = self._select_agent_for_task(next_task)
            print(f"選択されたエージェント: {agent_name}")
            
            # エージェントを実行
            if agent_name in self.agent_registry:
                agent_class = self.agent_registry[agent_name]
                agent = agent_class()
                
                # エージェント用の状態を準備
                agent_state = BaseTaskAgentState(
                    task_to_perform=next_task,
                    required_artifacts=self._get_required_artifacts(state, next_task)
                )
                
                # タスクを実行
                result_state = agent.execute_task(agent_state)
                
                # 結果を処理
                if result_state.status_report == "completed":
                    # 成果物を保存
                    if result_state.final_result:
                        for key, value in result_state.final_result.items():
                            state.artifacts[f"{next_task.id}_{key}"] = value
                    
                    # タスクを完了済みとしてマーク
                    state.mark_task_completed(next_task.id)
                    print(f"タスク完了: {next_task.name}")
                else:
                    # タスクが失敗
                    next_task.update_status(TaskStatus.FAILED)
                    print(f"タスク失敗: {next_task.name}")
            else:
                print(f"エラー: エージェント '{agent_name}' が見つかりません")
                next_task.update_status(TaskStatus.FAILED)
            
            iteration += 1
        
        return state
    
    def _select_next_task(self, state: GlobalAgentState, executable_tasks: List[Task]) -> Optional[Task]:
        """次に実行するタスクを選択"""
        if len(executable_tasks) == 1:
            return executable_tasks[0]
        
        # LLMを使用してタスクを選択
        completed_info = [f"{tid}: {state.task_plan[tid].name}" for tid in state.completed_tasks]
        executable_info = [f"{t.id}: {t.name}" for t in executable_tasks]
        
        messages = self.task_selection_prompt.format_messages(
            completed_tasks="\n".join(completed_info) if completed_info else "なし",
            executable_tasks="\n".join(executable_info),
            process_status=state.process_status.value
        )
        
        response = self.llm.invoke(messages)
        selected_id = response.content.strip()
        
        # 選択されたタスクを探す
        for task in executable_tasks:
            if task.id == selected_id:
                return task
        
        # 見つからない場合は最初のタスクを返す
        return executable_tasks[0]
    
    def _select_agent_for_task(self, task: Task) -> str:
        """タスクに適したエージェントを選択"""
        messages = self.agent_selection_prompt.format_messages(
            task_name=task.name,
            task_description=task.description,
            expected_output=task.expected_output_description
        )
        
        response = self.llm.invoke(messages)
        agent_name = response.content.strip()
        
        # 有効なエージェント名か確認
        if agent_name in self.agent_registry:
            return agent_name
        
        # デフォルトのマッピング
        if "計画" in task.name or "plan" in task.name.lower():
            return "InstructionPlanner"
        elif "ファイル" in task.name or "file" in task.name.lower() or "検索" in task.name:
            return "FileSearchExpert"
        elif "消込" in task.name or "reconcil" in task.name.lower():
            return "ReconciliationExpert"
        
        # デフォルト
        return "FileSearchExpert"
    
    def _get_required_artifacts(self, state: GlobalAgentState, task: Task) -> Dict[str, Any]:
        """タスクに必要な成果物を取得"""
        required_artifacts = {}
        
        # 依存タスクの成果物を収集
        for dep_id in task.dependencies:
            # 該当するタスクの成果物を探す
            for artifact_key, artifact_value in state.artifacts.items():
                if artifact_key.startswith(f"{dep_id}_"):
                    # プレフィックスを除去したキーで保存
                    clean_key = artifact_key[len(f"{dep_id}_"):]
                    required_artifacts[clean_key] = artifact_value
        
        return required_artifacts 