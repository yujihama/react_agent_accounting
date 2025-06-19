"""
Supervisor - ワークフロー全体の進行を管理する中央制御エージェント
"""

import yaml
from typing import Dict, Any, Optional, List, Type
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..states import GlobalAgentState, BaseTaskAgentState, Task, ProcessStatus, TaskStatus, TaskExecutionJudgment, TaskOptimizationResult
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
        
        # エージェント設定のキャッシュを初期化
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # 利用可能なタスクエージェントのレジストリを動的に構築
        self.agent_registry: Dict[str, Type[BaseTaskAgent]] = self._build_agent_registry()
        
        # タスク選択用のプロンプト
        self.task_selection_prompt = self._create_task_selection_prompt()
        
        # エージェント選択用のプロンプト
        self.agent_selection_prompt = self._create_agent_selection_prompt()
        
        # タスク実行結果判定用のプロンプト
        self.task_judgment_prompt = self._create_task_judgment_prompt()
        
        # エージェント割り当て用のプロンプト
        self.agent_assignment_prompt = self._create_agent_assignment_prompt()
    
    def _build_agent_registry(self) -> Dict[str, Type[BaseTaskAgent]]:
        """
        YAML設定ファイルからエージェントレジストリを動的に構築
        
        Returns:
            エージェント名とクラスのマッピング辞書
        """
        registry = {}
        
        # config key名からクラス名への変換マッピング
        config_to_class_mapping = {
            'instruction_planner': ('InstructionPlanner', InstructionPlanner),
            'filesearch_expert': ('FileSearchExpert', FileSearchExpert),
            'reconciliation_expert': ('ReconciliationExpert', ReconciliationExpert)
        }
        
        # agentsセクションから設定を読み込み
        agents_config = self.config.get('agents', {})
        
        for config_key, config_value in agents_config.items():
            if config_key in config_to_class_mapping:
                agent_name, agent_class = config_to_class_mapping[config_key]
                
                # エージェントレジストリに登録
                registry[agent_name] = agent_class
                
                # エージェント設定をキャッシュ
                self.agent_configs[agent_name] = config_value
                
                print(f"エージェント登録完了: {agent_name}")
            else:
                print(f"警告: 不明なエージェント設定キー: {config_key}")
        
        if not registry:
            raise ValueError("エージェントレジストリが空です。common_config.yamlのagentsセクションを確認してください。")
        
        print(f"登録されたエージェント数: {len(registry)}")
        return registry
    
    def _get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        エージェント設定を効率的に取得（キャッシュ利用）
        
        Args:
            agent_name: エージェント名
            
        Returns:
            エージェント設定辞書
        """
        return self.agent_configs.get(agent_name, {})
    
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
        
        # キャッシュされたエージェント情報を使用して動的にプロンプトを構築
        agents_info = []
        
        for agent_name in self.agent_registry.keys():
            # キャッシュされた設定を効率的に取得
            agent_config = self._get_agent_config(agent_name)
            
            if agent_config:
                description = agent_config.get('description', f'{agent_name}エージェント')
                skills = agent_config.get('skills', [])
                suitable_for = agent_config.get('suitable_for', [])
                not_suitable_for = agent_config.get('not_suitable_for', [])
                tools = agent_config.get('tools', [])
                
                agent_info = f"""
**{agent_name}**:
- 説明: {description}
- 専門スキル: {', '.join(skills) if skills else '設定なし'}
- 適用分野: {', '.join(suitable_for) if suitable_for else '設定なし'}
- 非適用分野: {', '.join(not_suitable_for) if not_suitable_for else '設定なし'}
- 利用可能ツール: {', '.join(tools) if tools else '設定なし'}"""
            else:
                # configがない場合のフォールバック
                agent_info = f"**{agent_name}**: 基本的なタスクエージェント"
            
            agents_info.append(agent_info)
        
        agents_description = '\n'.join(agents_info)
        
        # エージェント選択例も動的に生成
        selection_examples = []
        for agent_name in self.agent_registry.keys():
            agent_config = self._get_agent_config(agent_name)
            if agent_config and agent_config.get('suitable_for'):
                suitable_for = agent_config.get('suitable_for', [])
                for area in suitable_for[:1]:  # 最初の1つの例を使用
                    selection_examples.append(f"{len(selection_examples) + 4}. {area} → {agent_name}")
        
        examples_text = '\n'.join(selection_examples) if selection_examples else ""
        
        return ChatPromptTemplate.from_messages([
            ("system", f"""あなたはタスクに最適なエージェントを選択するスーパーバイザーです。

タスク情報:
- タスク名: {{task_name}}
- タスクの説明: {{task_description}}
- 期待される成果物: {{expected_output}}

利用可能なエージェント:
{agents_description}

**選択指針:**
1. タスクの内容を詳細に分析する
2. 各エージェントの「適用分野」と「非適用分野」を慎重に確認する
3. 最も専門性が高く、「非適用分野」に該当しないエージェントを選択する
{examples_text}

選択したエージェント名のみを返してください。"""),
            ("human", "このタスクに最適なエージェントを選択してください。")
        ])
    
    def _create_task_judgment_prompt(self) -> ChatPromptTemplate:
        """タスク実行結果判定用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_messages([
            ("system", """あなたはタスクエージェントの実行結果を評価するスーパーバイザーです。

以下の情報を総合的に分析し、タスクの成功/失敗を判定してください：

**タスク情報:**
- タスク名: {task_name}
- タスクの説明: {task_description}
- 期待される成果物: {expected_output}
- 実行エージェント: {agent_name}

**実行結果情報:**
- エージェントの報告ステータス: {status_report}
- エージェントからの詳細伝達事項: {communication_to_supervisor}
- 最終成果物の有無: {has_final_result}
- 最終成果物の概要: {final_result_summary}

**判定基準:**
1. 期待される成果物が得られているか
2. エージェントからの伝達事項に問題や懸念事項がないか  
3. 成果物の品質や完全性
4. 後続タスクに影響する要素はないか

**注意事項:**
- status_reportだけでなく、communication_to_supervisorの内容を重視してください
- エージェントが「部分的成功」や「課題あり」を報告している場合は慎重に判定してください
- 完全に失敗していなくても、品質に問題がある場合は適切に評価してください

以下の構造で判定結果を返してください：
- is_successful: bool (最終的な成功/失敗判定)
- confidence_level: str ('high', 'medium', 'low')
- reasoning: str (判定の根拠)
- identified_issues: List[str] (特定された問題点、あれば)
- required_actions: List[str] (必要な対応アクション、あれば)"""),
            ("human", "このタスク実行結果を判定してください。")
        ])
    
    def _create_agent_assignment_prompt(self) -> ChatPromptTemplate:
        """エージェント割り当て用のプロンプトテンプレートを作成"""
        
        # キャッシュされたエージェント情報を使用して動的にプロンプトを構築
        agents_info = []
        agent_examples = []  # エージェント推奨例を動的に生成
        
        for agent_name in self.agent_registry.keys():
            agent_config = self._get_agent_config(agent_name)
            
            if agent_config:
                description = agent_config.get('description', f'{agent_name}エージェント')
                skills = agent_config.get('skills', [])
                suitable_for = agent_config.get('suitable_for', [])
                tools = agent_config.get('tools', [])
                
                agent_info = f"""
**{agent_name}**:
- 説明: {description}
- 専門スキル: {', '.join(skills) if skills else '設定なし'}
- 適用分野: {', '.join(suitable_for) if suitable_for else '設定なし'}
- 利用可能ツール: {', '.join(tools) if tools else '設定なし'}"""
                
                # 適用分野から推奨例を生成
                if suitable_for:
                    for area in suitable_for[:2]:  # 最大2つの例を生成
                        agent_examples.append(f"   - {area} → {agent_name}")
            else:
                agent_info = f"**{agent_name}**: 基本的なタスクエージェント"
            
            agents_info.append(agent_info)
        
        agents_description = '\n'.join(agents_info)
        examples_description = '\n'.join(agent_examples) if agent_examples else "   - 設定から推奨例を生成できませんでした"
        
        return ChatPromptTemplate.from_messages([
            ("system", f"""あなたはタスクとエージェントの最適な割り当てを行うスーパーバイザーです。

**目標**: 各タスクが1つのエージェントで完結するように、タスク計画を最適化してください。

**利用可能なエージェント:**
{agents_description}

**最適化のルール:**
1. **タスク分解**: 複数の異なる作業が含まれるタスクは、エージェント単位に分解する
   - 例：「ファイルAとファイルBを取得する」→「ファイルAを取得する」「ファイルBを取得する」
   
2. **タスク集約**: 同一エージェントで効率的に実行できる複数タスクは集約する
   - 例：複数の類似ファイル検索タスク → 1つの包括的な検索タスク

3. **推奨エージェント**: 各タスクに最適なエージェントを推奨として付与する
{examples_description}

4. **依存関係の保持**: タスク分解・集約時も依存関係を適切に保持する

**現在のタスク計画:**
{{current_tasks}}"""),
            ("human", "タスク計画を最適化してください。optimized_tasksには各タスクの情報を、optimization_summaryには最適化の概要を含めてください。")
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
        
        # ステップ2: エージェント割り当て最適化
        print("\n=== ステップ2: エージェント割り当て最適化 ===")
        state.process_status = ProcessStatus.AGENT_ASSIGNMENT
        state = self._execute_agent_assignment_phase(state)
        
        if state.process_status == ProcessStatus.FAILED:
            return state
        
        # ステップ3: タスクの実行
        print("\n=== ステップ3: タスクの実行 ===")
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
        try:
            # InstructionPlannerを使用してタスク計画を生成
            planner = InstructionPlanner(self.config.get('__file__', 'config/common_config.yaml'))
            
            # プランナー用の状態を作成
            planner_state = BaseTaskAgentState(
                task_to_perform=Task(
                    name="タスク計画の生成",
                    description=f"以下の指示に基づいて具体的なタスク計画を生成してください: {state.input_instruction}",
                    expected_output_description="tasks というキーで、タスクのリストを含む辞書を返してください。各タスクは name, description, dependencies, expected_output_description を持つ必要があります。"
                )
            )
            
            # タスク計画を生成
            print("InstructionPlannerを実行中...")
            result_state = planner.execute_task(planner_state)
            
            print(f"プランナーの実行結果: status={result_state.status_report}")
            if result_state.final_result:
                print(f"最終結果のキー: {list(result_state.final_result.keys())}")
            
            if result_state.status_report == "completed" and result_state.final_result:
                if "task_plan" in result_state.final_result:
                    state.task_plan = result_state.final_result["task_plan"]
                    print(f"タスク計画を生成しました: {len(state.task_plan)}個のタスク")
                elif "tasks" in result_state.final_result:
                    # "tasks"キーがある場合の処理
                    print("タスクリストを変換中...")
                    tasks_data = result_state.final_result["tasks"]
                    task_plan = planner._create_task_plan(tasks_data)
                    state.task_plan = task_plan
                    print(f"タスク計画を生成しました: {len(state.task_plan)}個のタスク")
                else:
                    print(f"エラー: 期待されるキーが見つかりません。結果: {result_state.final_result}")
                    state.process_status = ProcessStatus.FAILED
            else:
                print(f"エラー: InstructionPlannerの実行に失敗しました。ステータス: {result_state.status_report}")
                if result_state.final_result:
                    print(f"エラー詳細: {result_state.final_result}")
                state.process_status = ProcessStatus.FAILED
                
        except Exception as e:
            print(f"計画フェーズでエラーが発生しました: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            state.process_status = ProcessStatus.FAILED
        
        return state
    
    def _execute_agent_assignment_phase(self, state: GlobalAgentState) -> GlobalAgentState:
        """
        エージェント割り当てフェーズを実行
        タスクとエージェントが1:1になるようにタスク計画を最適化
        """
        try:
            print("エージェント割り当て最適化を実行中...")
            
            # 現在のタスク計画を文字列に変換
            current_tasks_info = []
            for task_id, task in state.task_plan.items():
                task_info = {
                    "id": task_id,
                    "name": task.name,
                    "description": task.description,
                    "dependencies": task.dependencies,
                    "expected_output_description": task.expected_output_description
                }
                current_tasks_info.append(f"{task_id}: {task_info}")
            
            current_tasks_str = '\n'.join(current_tasks_info)
            
            # structured outputを使用するLLMを作成
            structured_llm = self.llm.with_structured_output(TaskOptimizationResult)
            
            # LLMでタスク計画を最適化
            messages = self.agent_assignment_prompt.format_messages(
                current_tasks=current_tasks_str
            )
            
            # structured outputで最適化結果を取得
            optimization_result = structured_llm.invoke(messages)
            
            print(f"最適化概要: {optimization_result.optimization_summary}")
            
            if optimization_result.optimized_tasks:
                # 新しいタスク計画を構築
                new_task_plan = {}
                
                for task_id, optimized_task in optimization_result.optimized_tasks.items():
                    # 既存のタスクから継承する情報
                    base_task = None
                    for original_id, original_task in state.task_plan.items():
                        if optimized_task.name == original_task.name or task_id == original_id:
                            base_task = original_task
                            break
                    
                    # 新しいタスクを作成
                    new_task = Task(
                        id=task_id,
                        name=optimized_task.name,
                        description=optimized_task.description,
                        dependencies=optimized_task.dependencies,
                        expected_output_description=optimized_task.expected_output_description,
                        status=TaskStatus.PENDING if base_task is None else base_task.status
                    )
                    
                    # 推奨エージェント情報を保存
                    if optimized_task.recommended_agent:
                        new_task.description += f"\n\n推奨エージェント: {optimized_task.recommended_agent}"
                    
                    new_task_plan[task_id] = new_task
                
                # タスク計画を更新
                state.task_plan = new_task_plan
                
                print(f"タスク計画の最適化が完了しました")
                print(f"最適化後のタスク数: {len(new_task_plan)}")
                
                # 各タスクの推奨エージェントを表示
                for task_id, task in new_task_plan.items():
                    agent_info = ""
                    if "推奨エージェント:" in task.description:
                        agent_info = task.description.split("推奨エージェント:")[-1].strip()
                    print(f"- {task.name} (推奨: {agent_info})")
            else:
                print("警告: 最適化されたタスクが空です。元のタスク計画を維持します。")
                
        except Exception as e:
            print(f"エージェント割り当てフェーズでエラーが発生しました: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # エラー時は元のタスク計画を維持し、処理を継続
        
        return state
    
    def _execute_tasks_phase(self, state: GlobalAgentState) -> GlobalAgentState:
        """タスク実行フェーズ"""
        max_iterations = 20  # 無限ループ防止
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
                
                # LLMを使用してタスク実行結果を判定
                judgment = self._judge_task_execution(result_state, next_task, agent_name)
                
                # 判定結果に基づいて処理
                if judgment.is_successful:
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
                    print(f"失敗理由: {judgment.reasoning}")
                    if judgment.identified_issues:
                        print(f"特定された問題: {', '.join(judgment.identified_issues)}")
                    # 将来的にはここでタスク修正ロジックを追加予定
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
        print(f"エラー: 次のタスクを選択できませんでした: {selected_id}")   
        return executable_tasks[0]
    
    def _select_agent_for_task(self, task: Task) -> str:
        """タスクに適したエージェントを選択"""
        # まず、推奨エージェント情報をチェック
        if "推奨エージェント:" in task.description:
            recommended_agent = task.description.split("推奨エージェント:")[-1].strip()
            if recommended_agent in self.agent_registry:
                print(f"推奨エージェントを使用: {recommended_agent}")
                return recommended_agent
            else:
                print(f"警告: 推奨エージェント '{recommended_agent}' が見つかりません。代替エージェントを選択します。")
        
        # 推奨エージェントがない場合は、従来のLLM選択を使用
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
        else:
            print(f"エラー: エージェント '{agent_name}' が見つかりません")
            return None
    
    def _get_required_artifacts(self, state: GlobalAgentState, task: Task) -> Dict[str, Any]:
        """タスクに必要な成果物を取得"""
        required_artifacts = {}
        
        # 依存タスクの成果物を収集
        for dep_id in task.dependencies:
            # 該当するタスクの成果物を探す
            for artifact_key, artifact_value in state.artifacts.items():
                required_artifacts[artifact_key] = artifact_value
        
        return required_artifacts
    
    def _judge_task_execution(self, result_state: BaseTaskAgentState, task: Task, agent_name: str) -> TaskExecutionJudgment:
        """
        エージェントのタスク実行結果をLLMで判定する
        
        Args:
            result_state: エージェントからの実行結果
            task: 実行されたタスク
            agent_name: 実行したエージェント名
            
        Returns:
            TaskExecutionJudgment: 判定結果
        """
        # structured outputを使用するLLMを作成
        structured_llm = self.llm.with_structured_output(TaskExecutionJudgment)
        
        # 判定用の情報を準備
        has_final_result = result_state.final_result is not None
        final_result_summary = "なし"
        if has_final_result:
            # final_resultの概要を作成（長すぎる場合は要約）
            result_str = str(result_state.final_result)
            if len(result_str) > 200:
                final_result_summary = f"{result_str[:200]}... (総文字数: {len(result_str)})"
            else:
                final_result_summary = result_str
        
        communication = result_state.communication_to_supervisor or "伝達事項なし"
        
        # プロンプトを生成
        messages = self.task_judgment_prompt.format_messages(
            task_name=task.name,
            task_description=task.description,
            expected_output=task.expected_output_description,
            agent_name=agent_name,
            status_report=result_state.status_report,
            communication_to_supervisor=communication,
            has_final_result=has_final_result,
            final_result_summary=final_result_summary
        )
        
        try:
            # LLMで判定を実行
            judgment = structured_llm.invoke(messages)
            
            print(f"\n=== タスク実行結果の判定 ===")
            print(f"タスク: {task.name}")
            print(f"エージェント: {agent_name}")
            print(f"判定結果: {'成功' if judgment.is_successful else '失敗'}")
            print(f"信頼度: {judgment.confidence_level}")
            print(f"判定理由: {judgment.reasoning}")
            if judgment.identified_issues:
                print(f"特定された問題: {judgment.identified_issues}")
            if judgment.required_actions:
                print(f"推奨アクション: {judgment.required_actions}")
            
            return judgment
            
        except Exception as e:
            print(f"タスク判定中にエラーが発生しました: {str(e)}")
            # エラー時はデフォルトで失敗判定
            return TaskExecutionJudgment(
                is_successful=False,
                confidence_level="high",
                reasoning=f"判定処理中にエラーが発生しました: {str(e)}",
                identified_issues=["判定処理エラー"],
                required_actions=["判定エラーの原因調査が必要"]
            ) 