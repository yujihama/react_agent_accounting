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
        return "ユーザーの指示を解釈し、手順書を参照しながら具体的で実行可能なタスク計画を生成する専門家です。"
    
    def get_tools(self) -> List[Tool]:
        """InstructionPlanner固有のツールを返す"""
        return [
            Tool(
                name="search_manuals_tool",
                func=self._search_manuals,
                description="指定されたキーワードに基づき、手順書ディレクトリ内から関連性の高い手順書ファイルを検索し、ファイルパスのリストを返す。"
            ),
            Tool(
                name="read_file_content_tool",
                func=self._read_file_content,
                description="指定されたファイルのテキスト内容をすべて読み込んで返す。"
            ),
            Tool(
                name="create_task_plan_tool",
                func=self._create_task_plan,
                description="タスクのリストを受け取り、適切な形式のタスク計画を生成する。"
            )
        ]
    
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
    
    def _create_task_plan(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Task]:
        """
        タスクデータからタスク計画を生成
        
        Args:
            tasks_data: タスクデータのリスト。各要素は以下の形式：
                {
                    "name": "タスク名",
                    "description": "タスクの説明",
                    "dependencies": ["依存タスクID", ...],
                    "expected_output_description": "期待される成果物の説明"
                }
        
        Returns:
            タスクIDをキーとしたタスク辞書
        """
        task_plan = {}
        
        for i, task_data in enumerate(tasks_data):
            task = Task(
                name=task_data.get("name", f"Task_{i+1}"),
                description=task_data.get("description", ""),
                dependencies=task_data.get("dependencies", []),
                expected_output_description=task_data.get("expected_output_description", "")
            )
            task_plan[task.id] = task
        
        return task_plan
    
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