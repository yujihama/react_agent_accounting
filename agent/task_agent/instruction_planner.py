"""
InstructionPlanner - ユーザー指示の解釈とタスク計画の生成に特化した専門家
"""

import os
import glob
from typing import List, Dict, Any
from langchain.tools import Tool

from .base import BaseTaskAgent
from ..states import Task


class InstructionPlanner(BaseTaskAgent):
    """
    ユーザー指示の解釈とタスク計画の生成に特化した専門家エージェント
    手順書を参照しながら、具体的で実行可能なタスクリストを生成する
    """
    
    def get_agent_description(self) -> str:
        """エージェントの説明を返す"""
        return """ユーザーの指示を解釈し、実行可能なタスク計画を効率的に生成する専門家です。

重要な原則：
1. 手順書検索は1-2回まで。関連ファイルが見つからなければ、一般的な知識に基づいてタスク計画を作成する
2. タスク計画は具体的で実行可能なステップに分解する
3. 過度に複雑な計画は避け、シンプルで効果的なアプローチを取る
4. 依存関係を明確にして、並行実行可能なタスクは分離する"""
    
    def get_result_format(self) -> str:
        """
        エージェントの結果のフォーマットを返す
        """
        return "タスク計画: タスクIDをキーとしたタスク辞書"
    
    def get_tools(self) -> List[Tool]:
        """InstructionPlanner固有のツールを返す（共通ツールを含む）"""
        return self.get_common_tools() + [
            Tool(
                name="search_manuals_tool",
                func=self._search_manuals_wrapper,
                description="指定されたキーワードに基づき、手順書ディレクトリ内から関連性の高い手順書ファイルを検索し、ファイルパスのリストを返す。\n引数:\n- keywords (list): 検索キーワードのリスト\n- keyword (str): 単一の検索キーワード（keywordsの代替）"
            ),
            Tool(
                name="read_file_content_tool",
                func=self._read_file_content,
                description="指定されたファイルのテキスト内容をすべて読み込んで返す。\n引数:\n- file_path (str): 読み込むファイルのパス"
            ),
            Tool(
                name="create_task_plan_tool",
                func=self._create_task_plan_wrapper,
                description="タスクのリストを受け取り、適切な形式のタスク計画を生成する。\n引数:\n- tasks (list): タスクデータのリスト\n- tasks_data (list): タスクデータのリスト（tasksの代替）"
            )
        ]
    
    def _search_manuals_wrapper(self, **kwargs) -> List[str]:
        """
        手順書検索のラッパー関数（LLMからの呼び出しに対応）
        """
        # 複数の引数形式に対応
        keywords = kwargs.get('keywords')
        if not keywords:
            keyword = kwargs.get('keyword', '')
            if isinstance(keyword, str):
                keywords = [keyword] if keyword else []
            else:
                keywords = keyword if isinstance(keyword, list) else []
        
        if not keywords:
            return ["エラー: 検索キーワードが指定されていません"]
        
        return self._search_manuals(keywords)
    
    def _search_manuals(self, keywords: List[str]) -> List[str]:
        """
        手順書ディレクトリから関連ファイルを検索
        
        Args:
            keywords: 検索キーワードのリスト
            
        Returns:
            関連するファイルパスのリスト
        """
        manuals_dir = self.config['environment']['manuals_directory']
        
        # ディレクトリが存在しない場合は空のリストを返す
        if not os.path.exists(manuals_dir):
            return []
        
        # サポートするファイル拡張子
        extensions = ['*.txt', '*.md', '*.pdf', '*.docx']
        found_files = []
        
        # 各拡張子でファイルを検索
        for ext in extensions:
            pattern = os.path.join(manuals_dir, '**', ext)
            files = glob.glob(pattern, recursive=True)
            
            # キーワードでフィルタリング
            for file_path in files:
                file_name = os.path.basename(file_path).lower()
                # いずれかのキーワードがファイル名に含まれていれば追加
                if any(keyword.lower() in file_name for keyword in keywords):
                    found_files.append(file_path)
        
        # 重複を除去して返す
        return list(set(found_files))
    
    def _read_file_content(self, file_path: str) -> str:
        """
        ファイルの内容を読み込む
        
        Args:
            file_path: 読み込むファイルのパス
            
        Returns:
            ファイルの内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            return f"エラー: ファイル '{file_path}' が見つかりません。"
        except Exception as e:
            return f"エラー: ファイルの読み込み中にエラーが発生しました: {str(e)}"
    
    def _create_task_plan_wrapper(self, **kwargs) -> Dict[str, Task]:
        """タスク計画作成のラッパー関数"""
        try:
            # 引数からタスクリストを取得
            tasks_data = kwargs.get('tasks', kwargs.get('tasks_data', []))
            
            if not tasks_data:
                return {"error": "タスクデータが提供されていません"}
            
            return self._create_task_plan(tasks_data)
            
        except Exception as e:
            return {"error": f"タスク計画の作成中にエラー: {str(e)}"}
    
    def _create_task_plan(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Task]:
        """
        タスクデータからタスク計画を生成
        
        Args:
            tasks_data: タスクデータのリスト
        
        Returns:
            タスクIDをキーとしたタスク辞書
        """
        task_dict = {}
        name_to_id_map = {}  # タスク名からIDへのマッピング
        
        # 最初にすべてのタスクを作成してIDマッピングを構築
        for task_data in tasks_data:
            task = Task(
                name=task_data.get('name', ''),
                description=task_data.get('description', ''),
                dependencies=[],  # 一時的に空にしておく
                expected_output_description=task_data.get('expected_output_description', '')
            )
            task_dict[task.id] = task
            name_to_id_map[task.name] = task.id
        
        # 依存関係を解決
        for task_data in tasks_data:
            task_name = task_data.get('name', '')
            if task_name in name_to_id_map:
                task_id = name_to_id_map[task_name]
                dependencies = task_data.get('dependencies', [])
                
                # 依存関係をタスクIDに変換
                resolved_dependencies = []
                for dep_name in dependencies:
                    if dep_name in name_to_id_map:
                        resolved_dependencies.append(name_to_id_map[dep_name])
                    else:
                        print(f"警告: 依存タスク '{dep_name}' が見つかりません")
                
                task_dict[task_id].dependencies = resolved_dependencies
        
        return task_dict
    
    def execute_task(self, state) -> Any:
        """
        タスクを実行（親クラスのメソッドをオーバーライド）
        InstructionPlannerの場合、最終成果物はタスク計画となる
        """
        # 親クラスのReActループを実行
        result_state = super().execute_task(state)
        
        # 成功した場合、タスク計画を適切な形式で返す
        if result_state.status_report == "completed" and result_state.final_result:
            # final_resultがタスクリストの場合、タスク計画に変換
            if "tasks" in result_state.final_result:
                tasks_data = result_state.final_result["tasks"]
                task_plan = self._create_task_plan(tasks_data)
                result_state.final_result = {"task_plan": task_plan}
        
        return result_state 