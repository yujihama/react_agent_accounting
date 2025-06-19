"""
BaseTaskAgent - 全ての専門TaskAgentの基底クラス
ReActの思考ループ実行、Supervisorとの通信といった共通機能を提供
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Type
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool

from ...states import BaseTaskAgentState, Task, Action, Observation, BaseReActResponse


class BaseTaskAgent(ABC):
    """
    全ての専門TaskAgentの基底クラスとなる汎用ReActエージェント。
    ReActの思考ループ実行、Supervisorとの通信（状態の受け取り・結果報告）といった共通機能を担う。
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
        
        # 最大反復回数
        self.max_iterations = self.config['common']['max_react_iterations']
        
        # エージェント固有のツールを取得（サブクラスで実装）
        self.tools = self.get_tools()
        
        # ReActプロンプトのテンプレート
        self.react_prompt = self._create_react_prompt()
    
    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """
        エージェント固有のツールを返す
        サブクラスで必ず実装する必要がある
        
        実装時は get_common_tools() を含めることを推奨:
        return self.get_common_tools() + [エージェント固有のツール...]
        """
        pass
    
    def get_common_tools(self) -> List[Tool]:
        """
        全エージェント共通のツールを返す
        サブクラスのget_toolsで利用可能
        
        Returns:
            共通ツールのリスト
        """
        return [
            Tool(
                name="set_communication_to_supervisor_tool",
                func=self._set_communication_wrapper,
                description="Supervisorに伝達事項やメッセージを送信する。重要な情報、エラー、推奨事項などを報告する際に使用。\n引数:\n- message (str): Supervisorに伝達するメッセージ内容"
            )
        ]
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """
        エージェントの役割と能力の説明を返す
        サブクラスで必ず実装する必要がある
        """
        pass
    
    @abstractmethod
    def get_result_format(self) -> str:
        """
        エージェントの結果のフォーマットを返す
        """
        return "成果物title: 成果物の内容"
    
    def get_response_model(self) -> Type[BaseReActResponse]:
        """
        このエージェントが使用するレスポンスモデルを返す
        サブクラスでオーバーライド可能（専門的なフィールドを追加したい場合）
        
        Returns:
            BaseReActResponseまたはそのサブクラス
        """
        return BaseReActResponse
    
    def set_communication_to_supervisor(self, message: str):
        """
        Supervisorへの伝達事項を設定する（エージェントから呼び出し可能）
        
        Args:
            message: Supervisorに伝達したいメッセージ
        """
        if hasattr(self, '_current_state') and self._current_state:
            self._current_state.communication_to_supervisor = message
            print(f"Supervisor伝達事項設定: {message}")
        else:
            print("警告: 現在の状態が設定されていないため、伝達事項を設定できませんでした")
    
    def _create_react_prompt(self) -> ChatPromptTemplate:
        """ReActループ用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_messages([
            ("system", """あなたは{agent_description}

以下のタスクを実行してください：
タスク名: {task_name}
タスクの説明: {task_description}
期待される成果物: {expected_output}

利用可能なツール:
{available_tools}

これまでの思考と観察:
{history}

**重要な指針:**
1. 同じツールを2回以上連続で使用することは避けてください
2. 手順書検索で結果が見つからない場合は、一般的な知識に基づいて進めてください
3. タスクの目的を達成するために、最も直接的で効率的なアプローチを選択してください
4. 既に試行して失敗したアプローチは繰り返さないでください

タスクを完了するために、以下の構造で応答してください：

1. thought: 現在の状況を分析し、次に何をすべきか考える
2. is_complete: タスクが完了した場合（Supervisor伝達事項設定があって先に進められない場合も含む）はtrue、続行する場合はfalse
3. action（タスクが未完了の場合）:
   - action_name: 使用するツール名
   - action_input: ツールに渡す引数（JSON形式）
4. final_answer（タスクが完了した場合、Supervisor伝達事項設定があって先に進められない場合）:
   - result: {result_format}
   - summary: 結果の要約説明

タスクが完了していない場合は、次のアクションを指定してください。
タスクが完了した場合（Supervisor伝達事項設定があって先に進められない場合も含む）は、is_completeをtrueにして、final_answerに結果を含めてください。
"""),
            ("human", "タスクを実行してください。")
        ])
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """指定されたツールを実行"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.func(**tool_input)
        
        return f"エラー: ツール '{tool_name}' が見つかりません。"
    
    def _extract_additional_response_fields(self, response: BaseReActResponse) -> Dict[str, Any]:
        """
        レスポンスモデルから基本フィールド以外の追加フィールドを汎用的に抽出
        
        Args:
            response: レスポンスオブジェクト（BaseReActResponseまたはその派生クラス）
            
        Returns:
            追加フィールドの辞書（値がNoneでないもののみ）
        """
        # BaseReActResponseの基本フィールドを定義
        base_fields = {'thought', 'action', 'final_answer'}
        
        # レスポンスオブジェクトのすべてのフィールドを取得
        additional_fields = {}
        
        # Pydanticモデルのフィールドを取得
        if hasattr(response, '__fields__'):
            # Pydantic v1 互換
            all_fields = set(response.__fields__.keys())
        elif hasattr(response, 'model_fields'):
            # Pydantic v2
            all_fields = set(response.model_fields.keys())
        else:
            # フォールバック: dir()を使用
            all_fields = {attr for attr in dir(response) 
                         if not attr.startswith('_') and not callable(getattr(response, attr))}
        
        # 基本フィールド以外のフィールドを検出
        custom_fields = all_fields - base_fields
        
        # 値が設定されている追加フィールドのみを抽出
        for field_name in custom_fields:
            if hasattr(response, field_name):
                field_value = getattr(response, field_name)
                # 値がNoneでなく、設定されている場合のみ追加
                if field_value is not None:
                    additional_fields[field_name] = field_value
        
        return additional_fields
    
    def execute_task(self, state: BaseTaskAgentState) -> BaseTaskAgentState:
        """
        タスクを実行するメインメソッド
        ReActループを実行し、結果を返す
        """
        # 現在の状態をインスタンス変数として保存（ツールからアクセス可能にする）
        self._current_state = state
        
        task = state.task_to_perform
        if not task:
            state.status_report = "failed"
            state.final_result = {"error": "実行するタスクが指定されていません"}
            return state
        
        # 利用可能なツールの説明を生成
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        # レスポンスモデルを取得
        response_model = self.get_response_model()
        
        # ReActループの実行
        while state.iteration_count < self.max_iterations:
            # プロンプトの準備
            messages = self.react_prompt.format_messages(
                agent_description=self.get_agent_description(),
                task_name=task.name,
                task_description=task.description,
                expected_output=task.expected_output_description,
                available_tools=tools_description,
                history=state.get_history_string(),
                result_format=self.get_result_format()
            )
            
            # LLMに問い合わせ（response_modelを使用したstructured output）
            try:
                # 動的にレスポンスモデルの型を使用
                structured_llm = self.llm.with_structured_output(response_model)
                response = structured_llm.invoke(messages)
                
                # デバッグ: レスポンスを表示
                print(f"\n--- Iteration {state.iteration_count + 1} ---")
                print(f"Thought: {response.thought.thought}")
                print(f"Is Complete: {response.thought.is_complete}")
                print(f"response: {response.dict()}")
                
                # 最終回答が得られた場合
                if response.is_complete():
                    final_result = response.get_final_result()
                    if final_result:
                        state.final_result = final_result
                        state.status_report = "completed"
                        print(f"Final Result: {final_result}")
                    else:
                        # is_completeがTrueだがfinal_answerがない場合
                        # レスポンスから直接結果を抽出（汎用的な追加フィールド検出）
                        additional_fields = self._extract_additional_response_fields(response)
                        
                        # 基本結果に追加フィールドをマージ
                        state.final_result = {
                            "result": "タスクが完了しました", 
                            "thought": response.thought.thought,
                            **additional_fields
                        }
                        state.status_report = "completed"
                    return state
                
                # アクションが指定された場合
                action_details = response.get_action_details()
                if action_details:
                    action_name, action_input = action_details
                    print(f"Action: {action_name}({action_input})")
                    
                    # ツールを実行
                    try:
                        observation = self._execute_tool(action_name, action_input)
                        print(f"Observation: {observation}")
                        
                        # ステップを記録
                        action_obj = Action(tool=action_name, tool_input=action_input)
                        observation_obj = Observation(result=observation)
                        state.add_step(action_obj, observation_obj)
                        
                    except Exception as e:
                        # エラーも観察として記録
                        error_msg = f"エラー: {str(e)}"
                        print(f"Error: {error_msg}")
                        action_obj = Action(tool=action_name, tool_input=action_input)
                        observation_obj = Observation(result=error_msg)
                        state.add_step(action_obj, observation_obj)
                else:
                    # アクションが指定されていない場合
                    error_msg = "アクションが指定されていません"
                    print(f"Error: {error_msg}")
                    action_obj = Action(tool="no_action", tool_input={"response": response.dict()})
                    observation_obj = Observation(result=error_msg)
                    state.add_step(action_obj, observation_obj)
                    
            except Exception as e:
                # structured outputの解析エラー
                error_msg = f"Structured output解析エラー: {str(e)}"
                print(f"Error: {error_msg}")
                action_obj = Action(tool="parse_error", tool_input={"error": str(e)})
                observation_obj = Observation(result=error_msg)
                state.add_step(action_obj, observation_obj)
        
        # 最大反復回数に達した場合
        state.status_report = "failed"
        state.final_result = {"error": f"最大反復回数（{self.max_iterations}）に達しました"}
        return state 

    def _set_communication_wrapper(self, **kwargs) -> str:
        """
        set_communication_to_supervisorのラッパー関数（LLMからの呼び出しに対応）
        
        Args:
            **kwargs: message パラメータを含む辞書
            
        Returns:
            実行結果のメッセージ
        """
        message = kwargs.get('message', '')
        if not message:
            return "エラー: メッセージが指定されていません。"
        
        self.set_communication_to_supervisor(message)
        return f"Supervisorに以下のメッセージを送信しました: {message}" 