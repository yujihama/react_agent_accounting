"""
LangGraph ワークフロー - ReAct型エージェントシステムのワークフロー定義
"""

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.states import GlobalAgentState, ProcessStatus
from agent.supervisor_agent import Supervisor


class WorkflowState(TypedDict):
    """ワークフローの状態定義"""
    global_state: GlobalAgentState
    current_step: str
    error_message: Optional[str]


class LangGraphWorkflow:
    """
    LangGraphを使用したワークフローの実装
    """
    
    def __init__(self, config_path: str = "config/common_config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.supervisor = Supervisor(config_path)
        
        # メモリセーバー（チェックポイント管理）
        self.memory = MemorySaver()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """ワークフローグラフを構築"""
        # StateGraphの作成
        workflow = StateGraph(WorkflowState)
        
        # ノードの追加
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("execution", self._execution_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # エッジの追加（遷移条件）
        workflow.set_entry_point("initialize")
        
        workflow.add_edge("initialize", "planning")
        
        workflow.add_conditional_edges(
            "planning",
            self._planning_routing,
            {
                "execution": "execution",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "execution",
            self._execution_routing,
            {
                "execution": "execution",  # タスクが残っている場合は継続
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        """初期化ノード"""
        print("=== ワークフロー開始 ===")
        state["current_step"] = "initialize"
        state["error_message"] = None
        return state
    
    def _planning_node(self, state: WorkflowState) -> WorkflowState:
        """タスク計画ノード"""
        print("\n=== タスク計画フェーズ ===")
        state["current_step"] = "planning"
        
        try:
            # Supervisorの計画フェーズを実行
            global_state = state["global_state"]
            global_state = self.supervisor._execute_planning_phase(global_state)
            state["global_state"] = global_state
            
            if global_state.process_status == ProcessStatus.FAILED:
                state["error_message"] = "タスク計画の生成に失敗しました"
        except Exception as e:
            state["error_message"] = f"計画フェーズでエラーが発生: {str(e)}"
            state["global_state"].process_status = ProcessStatus.FAILED
        
        return state
    
    def _execution_node(self, state: WorkflowState) -> WorkflowState:
        """タスク実行ノード"""
        state["current_step"] = "execution"
        global_state = state["global_state"]
        
        try:
            # 実行可能なタスクを取得
            executable_tasks = global_state.get_executable_tasks()
            
            if executable_tasks:
                # 最初のタスクを実行（Supervisorの内部ロジックを使用）
                next_task = self.supervisor._select_next_task(global_state, executable_tasks)
                
                if next_task:
                    print(f"\n--- タスク実行: {next_task.name} ---")
                    global_state.current_task_id = next_task.id
                    
                    # エージェントの選択と実行
                    agent_name = self.supervisor._select_agent_for_task(next_task)
                    print(f"選択されたエージェント: {agent_name}")
                    
                    if agent_name in self.supervisor.agent_registry:
                        from agent.states import BaseTaskAgentState, TaskStatus
                        
                        agent_class = self.supervisor.agent_registry[agent_name]
                        agent = agent_class()
                        
                        # エージェント用の状態を準備
                        agent_state = BaseTaskAgentState(
                            task_to_perform=next_task,
                            required_artifacts=self.supervisor._get_required_artifacts(global_state, next_task)
                        )
                        
                        # タスクを実行
                        result_state = agent.execute_task(agent_state)
                        
                        # 結果を処理
                        if result_state.status_report == "completed":
                            # 成果物を保存
                            if result_state.final_result:
                                for key, value in result_state.final_result.items():
                                    global_state.artifacts[f"{next_task.id}_{key}"] = value
                            
                            # タスクを完了済みとしてマーク
                            global_state.mark_task_completed(next_task.id)
                            print(f"タスク完了: {next_task.name}")
                        else:
                            # タスクが失敗
                            next_task.update_status(TaskStatus.FAILED)
                            print(f"タスク失敗: {next_task.name}")
            
            state["global_state"] = global_state
            
        except Exception as e:
            state["error_message"] = f"実行フェーズでエラーが発生: {str(e)}"
            print(f"エラー: {str(e)}")
        
        return state
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """終了処理ノード"""
        print("\n=== ワークフロー終了 ===")
        state["current_step"] = "finalize"
        
        global_state = state["global_state"]
        
        # 最終ステータスの設定
        if all(task.status.value == "completed" for task in global_state.task_plan.values()):
            global_state.process_status = ProcessStatus.SUCCESS
            print("ステータス: 成功")
        else:
            global_state.process_status = ProcessStatus.FAILED
            print("ステータス: 失敗")
            if state["error_message"]:
                print(f"エラー: {state['error_message']}")
        
        # 結果サマリー
        print("\n--- 実行結果サマリー ---")
        print(f"総タスク数: {len(global_state.task_plan)}")
        print(f"完了タスク数: {len(global_state.completed_tasks)}")
        print(f"成果物数: {len(global_state.artifacts)}")
        
        state["global_state"] = global_state
        return state
    
    def _planning_routing(self, state: WorkflowState) -> str:
        """計画フェーズ後のルーティング"""
        if state["global_state"].process_status == ProcessStatus.FAILED:
            return "finalize"
        elif state["global_state"].task_plan:
            return "execution"
        else:
            return "finalize"
    
    def _execution_routing(self, state: WorkflowState) -> str:
        """実行フェーズ後のルーティング"""
        global_state = state["global_state"]
        executable_tasks = global_state.get_executable_tasks()
        
        if executable_tasks: #and not state.get("error_message"):
            return "execution"  # まだタスクが残っている
        else:
            return "finalize"
    
    def run(self, instruction: str) -> GlobalAgentState:
        """
        ワークフローを実行
        
        Args:
            instruction: ユーザーからの指示
            
        Returns:
            最終的なグローバル状態
        """
        # 初期状態を作成
        initial_state = WorkflowState(
            global_state=GlobalAgentState(
                input_instruction=instruction,
                process_status=ProcessStatus.PLANNING
            ),
            current_step="initialize",
            error_message=None
        )
        
        # ワークフローを実行
        config = {"configurable": {"thread_id": "default"}}
        final_state = self.workflow.invoke(initial_state, config)
        
        return final_state["global_state"]


# ワークフローのファクトリ関数
def create_workflow(config_path: str = "config/common_config.yaml") -> LangGraphWorkflow:
    """
    LangGraphワークフローを作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        LangGraphWorkflowインスタンス
    """
    return LangGraphWorkflow(config_path) 