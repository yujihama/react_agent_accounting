"""
BaseTaskAgent - 全ての専門TaskAgentの基底クラス
ReActの思考ループ実行、Supervisorとの通信といった共通機能を提供
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool

from ...states import BaseTaskAgentState, Task, Action, Observation


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
        """
        pass
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """
        エージェントの役割と能力の説明を返す
        サブクラスで必ず実装する必要がある
        """
        pass
    
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

タスクを完了するために、次に何をすべきか考えてください。
思考プロセスは以下の形式に従ってください：

Thought: 現在の状況を分析し、次に何をすべきか考える
Action: 使用するツール名
Action Input: ツールに渡す引数（JSON形式）

ツールの実行結果が返ってきたら、その結果を観察し、次のステップを考えます。
タスクが完了したと判断したら、以下の形式で終了してください：

Thought: タスクが完了した
Final Answer: {{最終的な成果物の説明とデータ}}
"""),
            ("human", "タスクを実行してください。")
        ])
    
    def _parse_llm_output(self, output: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        LLMの出力を解析して、アクションまたは最終回答を抽出
        
        Returns:
            (action_name, action_input, final_answer) のタプル
        """
        lines = output.strip().split('\n')
        
        thought = None
        action = None
        action_input = None
        final_answer = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Thought:"):
                thought = line[8:].strip()
                # 最終回答のチェック
                if "完了" in thought or "complete" in thought.lower():
                    # Final Answerを探す
                    for j in range(i+1, len(lines)):
                        if lines[j].strip().startswith("Final Answer:"):
                            final_answer_text = lines[j][13:].strip()
                            # JSON形式の場合はパース
                            try:
                                import json
                                final_answer = json.loads(final_answer_text)
                            except:
                                final_answer = {"result": final_answer_text}
                            break
                    return None, None, final_answer
            
            elif line.startswith("Action:"):
                action = line[7:].strip()
            
            elif line.startswith("Action Input:"):
                # 複数行のJSONに対応
                input_start = i
                input_lines = [lines[i][13:].strip()]
                i += 1
                
                # JSONの終了を検出
                while i < len(lines) and not lines[i].strip().startswith(("Thought:", "Action:", "Final Answer:")):
                    input_lines.append(lines[i])
                    i += 1
                
                try:
                    import json
                    action_input = json.loads('\n'.join(input_lines))
                except:
                    action_input = {"input": '\n'.join(input_lines)}
                
                return action, action_input, None
            
            i += 1
        
        return action, action_input, final_answer
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """指定されたツールを実行"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.func(**tool_input)
        
        return f"エラー: ツール '{tool_name}' が見つかりません。"
    
    def execute_task(self, state: BaseTaskAgentState) -> BaseTaskAgentState:
        """
        タスクを実行するメインメソッド
        ReActループを実行し、結果を返す
        """
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
        
        # ReActループの実行
        while state.iteration_count < self.max_iterations:
            # プロンプトの準備
            messages = self.react_prompt.format_messages(
                agent_description=self.get_agent_description(),
                task_name=task.name,
                task_description=task.description,
                expected_output=task.expected_output_description,
                available_tools=tools_description,
                history=state.get_history_string()
            )
            
            # LLMに問い合わせ
            response = self.llm.invoke(messages)
            llm_output = response.content
            
            # 出力を解析
            action_name, action_input, final_answer = self._parse_llm_output(llm_output)
            
            # 最終回答が得られた場合
            if final_answer:
                state.final_result = final_answer
                state.status_report = "completed"
                return state
            
            # アクションが指定された場合
            if action_name and action_input:
                # ツールを実行
                try:
                    observation = self._execute_tool(action_name, action_input)
                    
                    # ステップを記録
                    action_obj = Action(tool=action_name, tool_input=action_input)
                    observation_obj = Observation(result=observation)
                    state.add_step(action_obj, observation_obj)
                    
                except Exception as e:
                    # エラーも観察として記録
                    action_obj = Action(tool=action_name, tool_input=action_input)
                    observation_obj = Observation(result=f"エラー: {str(e)}")
                    state.add_step(action_obj, observation_obj)
            else:
                # 解析できなかった場合もエラーとして記録
                action_obj = Action(tool="parse_error", tool_input={"output": llm_output})
                observation_obj = Observation(result="LLMの出力を解析できませんでした")
                state.add_step(action_obj, observation_obj)
        
        # 最大反復回数に達した場合
        state.status_report = "failed"
        state.final_result = {"error": f"最大反復回数（{self.max_iterations}）に達しました"}
        return state 