"""
LangGraph ReAct型エージェント - 状態定義
状態を明確に定義することで、エージェント間の情報伝達を安定させる。
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class TaskStatus(str, Enum):
    """タスクのステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessStatus(str, Enum):
    """ワークフロー全体のステータス"""
    PLANNING = "planning"
    AGENT_ASSIGNMENT = "agent_assignment"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Task(BaseModel):
    """タスクの定義"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    expected_output_description: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def update_status(self, new_status: TaskStatus):
        """ステータスを更新"""
        self.status = new_status
        self.updated_at = datetime.now()


class Action(BaseModel):
    """ReActループにおけるアクション"""
    tool: str
    tool_input: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class Observation(BaseModel):
    """ツール実行結果の観察"""
    result: Any
    timestamp: datetime = Field(default_factory=datetime.now)


# === ReAct Structured Output Models ===

class ReActThought(BaseModel):
    """ReActループの思考フェーズ"""
    thought: str = Field(description="現在の状況分析と次の行動の判断")
    is_complete: bool = Field(default=False, description="タスクが完了したかどうか")


class ReActAction(BaseModel):
    """ReActループのアクションフェーズ"""
    action_name: str = Field(description="実行するツール名")
    action_input: Optional[Dict[str, Any]] = Field(default_factory=dict, description="ツールに渡す引数")


class ReActFinalAnswer(BaseModel):
    """ReActループの最終回答"""
    result: Any = Field(description="最終的な成果物（辞書、リスト、文字列など）")
    summary: Optional[str] = Field(default=None, description="結果の要約説明（省略可能）")


class BaseReActResponse(BaseModel):
    """
    ReActレスポンスの基底クラス
    サブクラスでカスタマイズ可能な構造を提供
    """
    thought: ReActThought
    action: Optional[ReActAction] = None
    final_answer: Optional[ReActFinalAnswer] = None
    
    class Config:
        # サブクラスでの拡張を許可
        extra = "allow"
    
    def is_complete(self) -> bool:
        """タスクが完了しているかチェック"""
        return self.thought.is_complete or self.final_answer is not None
    
    def get_action_details(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """アクションの詳細を取得"""
        if self.action:
            return self.action.action_name, self.action.action_input
        return None
    
    def get_final_result(self) -> Optional[Any]:
        """最終結果を取得（辞書形式で包む）"""
        if self.final_answer:
            result = self.final_answer.result
            # リストや単純な値の場合は、辞書に包む
            if isinstance(result, dict):
                return result
            else:
                return {"result": result}
        return None


# === 専門エージェント向けのカスタマイズ例 ===

class FileSearchReActResponse(BaseReActResponse):
    """ファイル検索エージェント専用のレスポンス形式（例）"""
    found_files_path: List[str] = Field(default=None, description="発見されたファイルパスのリスト")


class ReconciliationReActResponse(BaseReActResponse):
    """照合エージェント専用のレスポンス形式（例）"""
    reconciliation_status: Optional[str] = Field(default=None, description="照合の状況")
    error_count: Optional[int] = Field(default=None, description="エラー件数")


# === Supervisor用の判定モデル ===

class TaskExecutionJudgment(BaseModel):
    """
    Supervisorがエージェントのタスク実行結果を判定するためのStructured Output
    """
    is_successful: bool = Field(
        description="タスクが成功したかどうかの最終判定"
    )
    confidence_level: str = Field(
        description="判定の信頼度 ('high', 'medium', 'low')"
    )
    reasoning: str = Field(
        description="判定の根拠と理由"
    )
    identified_issues: Optional[List[str]] = Field(
        default=None,
        description="特定された問題点のリスト（失敗時またはpartial success時）"
    )
    required_actions: Optional[List[str]] = Field(
        default=None,
        description="失敗時に必要な対応アクション（将来の自動修正で使用予定）"
    )


class OptimizedTask(BaseModel):
    """最適化されたタスクの定義"""
    name: str = Field(description="タスク名")
    description: str = Field(description="詳細な説明")
    dependencies: List[str] = Field(default_factory=list, description="依存タスクIDのリスト")
    expected_output_description: str = Field(description="期待される成果物の説明")
    recommended_agent: Optional[str] = Field(default=None, description="推奨エージェント名")


class TaskOptimizationResult(BaseModel):
    """
    エージェント割り当て最適化の結果を表すStructured Output
    """
    optimized_tasks: Dict[str, OptimizedTask] = Field(
        description="最適化されたタスク計画。タスクIDをキーとする辞書"
    )
    optimization_summary: str = Field(
        description="実行した最適化の概要説明"
    )


class GlobalAgentState(BaseModel):
    """
    Supervisorが管理するワークフロー全体の永続的な状態。
    タスク計画、進行状況、成果物など、すべてがここに集約される。
    """
    input_instruction: str = Field(
        description="人間のユーザーから最初に与えられる、達成したい目標の指示。"
    )
    task_plan: Dict[str, Task] = Field(
        default_factory=dict,
        description="InstructionPlannerによって生成されたタスク計画。タスクIDをキーとする辞書形式で管理される。"
    )
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="完了したタスクのIDを格納するリスト。タスクの依存関係解決に使用される。"
    )
    current_task_id: Optional[str] = Field(
        None,
        description="現在実行中、または次に実行すべきタスクのID。ワークフローはシーケンシャルに進行する。"
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="ワークフロー全体で共有される成果物のストア。ファイルパスやデータフレームなどがタスク間で受け渡される。"
    )
    process_status: ProcessStatus = Field(
        ProcessStatus.PLANNING,
        description="ワークフロー全体の高レベルなステータス。"
    )
    
    def get_executable_tasks(self) -> List[Task]:
        """実行可能なタスクのリストを返す（依存関係が解決済みのもの）"""
        executable = []
        for task_id, task in self.task_plan.items():
            if task.status == TaskStatus.PENDING:
                # 全ての依存タスクが完了しているかチェック
                if all(dep_id in self.completed_tasks for dep_id in task.dependencies):
                    executable.append(task)
        return executable
    
    def mark_task_completed(self, task_id: str):
        """タスクを完了済みとしてマーク"""
        if task_id in self.task_plan:
            self.task_plan[task_id].update_status(TaskStatus.COMPLETED)
            if task_id not in self.completed_tasks:
                self.completed_tasks.append(task_id)


class BaseTaskAgentState(BaseModel):
    """
    全てのTaskAgentが継承するBaseTaskAgentが、内部で持つ一時的な状態。
    この状態はタスクの完了と共に破棄される。
    """
    task_to_perform: Optional[Task] = Field(
        None,
        description="Supervisorから実行を指示されたTaskオブジェクト。Agentの思考の起点となる。"
    )
    required_artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="タスク実行に必要となる、先行タスクが生成した成果物。"
    )
    intermediate_steps: List[Tuple[Action, Observation]] = Field(
        default_factory=list,
        description="ReActループにおける思考（Action）とツール実行結果（Observation）の履歴。"
    )
    final_result: Optional[Dict[str, Any]] = Field(
        None,
        description="タスクの最終成果物。この内容がSupervisorによってGlobalAgentStateのartifactsにマージされる。"
    )
    status_report: str = Field(
        "pending",
        description="タスク実行の最終ステータス報告。"
    )
    communication_to_supervisor: Optional[str] = Field(
        None,
        description="エージェントからSupervisorへの詳細な伝達事項。成功・失敗の詳細、問題点、推奨事項などを記載。"
    )
    iteration_count: int = Field(
        0,
        description="ReActループの現在の反復回数"
    )
    
    def add_step(self, action: Action, observation: Observation):
        """ReActループのステップを追加"""
        self.intermediate_steps.append((action, observation))
        self.iteration_count += 1
    
    def get_history_string(self) -> str:
        """これまでの思考と観察の履歴を文字列として返す"""
        history = []
        for i, (action, observation) in enumerate(self.intermediate_steps):
            history.append(f"Step {i+1}:")
            history.append(f"  Action: {action.tool}({action.tool_input})")
            history.append(f"  Observation: {observation.result}")
        return "\n".join(history) 