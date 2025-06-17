"""
LangGraph ReAct型エージェント - 状態定義
状態を明確に定義することで、エージェント間の情報伝達を安定させる。
"""

from typing import Dict, List, Optional, Any, Tuple
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