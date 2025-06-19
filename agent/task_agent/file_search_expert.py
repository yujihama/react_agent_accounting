"""
FileSearchExpert - ファイルシステムの検索と特定に特化した専門家
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional, Type
from langchain.tools import Tool

from .base import BaseTaskAgent
from ..states import FileSearchReActResponse


class FileSearchExpert(BaseTaskAgent):
    """
    ファイルシステムの検索と特定に特化した専門家エージェント
    パターンマッチングやファイル内容のプレビューを通じて、必要なファイルを見つける
    """
    
    def get_agent_description(self) -> str:
        """エージェントの説明を返す"""
        return "フォルダを検索し、パターンマッチングやファイル内容の確認を通じて必要なファイルを特定し、ファイルパスを返す専門家です。"
    
    def get_result_format(self) -> str:
        """
        エージェントの結果のフォーマットを返す
        """
        return "<ファイルタイトル>:<対象ファイルパス>"

    def get_response_model(self) -> Type[FileSearchReActResponse]:
        """
        FileSearchExpert専用のレスポンスモデルを返す
        発見されたファイルリストを含む追加フィールドを持つ
        """
        return FileSearchReActResponse
    
    def _normalize_directory_path(self, directory: str) -> str:
        """
        ディレクトリパスを正規化し、data/inputをルートディレクトリとして扱う
        
        Args:
            directory: 入力されたディレクトリパス
            
        Returns:
            正規化されたディレクトリパス
        """
        # data/inputをベースディレクトリとして設定
        base_dir = os.path.join(os.getcwd(), 'data', 'input')
        
        # 「/」や空文字列の場合はdata/inputをそのまま返す
        if directory in ['/', '', '.']:
            return base_dir
        
        # 絶対パスの場合はそのまま返す
        if os.path.isabs(directory):
            return directory
        
        # 相対パスの場合はdata/inputをベースとして結合
        return os.path.join(base_dir, directory.lstrip('./'))
    
    def _normalize_file_path(self, file_path: str) -> str:
        """
        ファイルパスを正規化し、data/inputをルートディレクトリとして扱う
        
        Args:
            file_path: 入力されたファイルパス
            
        Returns:
            正規化されたファイルパス
        """
        # data/inputをベースディレクトリとして設定
        base_dir = os.path.join(os.getcwd(), 'data', 'input')
        
        # 絶対パスの場合はそのまま返す
        if os.path.isabs(file_path):
            return file_path
        
        # ファイルパスが「/」で始まる場合は、data/inputをベースとして結合
        if file_path.startswith('/'):
            return os.path.join(base_dir, file_path.lstrip('/'))
        
        # 相対パスの場合はdata/inputをベースとして結合
        return os.path.join(base_dir, file_path.lstrip('./'))
    
    def get_tools(self) -> List[Tool]:
        """FileSearchExpert固有のツールを返す（共通ツールを含む）"""
        return self.get_common_tools() + [
            Tool(
                name="file_search_tool",
                func=self._file_search_wrapper,
                description="指定されたディレクトリ内でパターンに一致するファイルを再帰的に検索する。\n引数:\n- directory (str): 検索対象のディレクトリパス\n- filename_pattern (str): ファイル名のパターン（ワイルドカード使用可能、例：'*.csv'）"
            ),
            Tool(
                name="file_content_preview_tool",
                func=self._file_content_preview_wrapper,
                description="ファイルの先頭数行を読み込み、内容のプレビューを返す。\n引数:\n- file_path (str): プレビューするファイルのパス\n- lines (int, 省略可能): 読み込む行数（デフォルト: 10）"
            ),
            Tool(
                name="list_directory_tool",
                func=self._list_directory,
                description="指定されたディレクトリの内容をリストする。\n引数:\n- directory (str): リストするディレクトリのパス"
            )
        ]
    
    def _file_search_wrapper(self, **kwargs) -> List[str]:
        """
        ファイル検索のラッパー関数（LLMからの呼び出しに対応）
        """
        directory = kwargs.get('directory', '/')
        filename_pattern = kwargs.get('filename_pattern') or kwargs.get('pattern') or kwargs.get('keyword', '*')
        
        # ディレクトリパスを正規化（data/inputをルートとして扱う）
        normalized_directory = self._normalize_directory_path(directory)
        
        return self._file_search(normalized_directory, filename_pattern)
    
    def _file_search(self, directory: str, filename_pattern: str) -> List[str]:
        """
        指定されたディレクトリ内でパターンに一致するファイルを検索
        
        Args:
            directory: 検索対象のディレクトリ（すでに正規化済み）
            filename_pattern: ファイル名のパターン（ワイルドカード使用可）
            
        Returns:
            マッチしたファイルパスのリスト
        """
        # ディレクトリパスはすでに_normalize_directory_pathで正規化されているため、
        # 絶対パスでない場合のみ絶対パスに変換
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)
        
        # ディレクトリが存在しない場合
        if not os.path.exists(directory):
            return [f"エラー: ディレクトリ '{directory}' が見つかりません。"]
        
        # パターンに基づいてファイルを検索
        search_pattern = os.path.join(directory, '**', filename_pattern)
        matched_files = glob.glob(search_pattern, recursive=True)
        
        # 結果が空の場合
        if not matched_files:
            return [f"パターン '{filename_pattern}' に一致するファイルが見つかりませんでした。"]
        
        # ファイルの情報を追加（相対パスで返す）
        file_info = []
        for file_path in matched_files:
            try:
                # 指定されたディレクトリからの相対パスを計算
                relative_path = os.path.relpath(file_path, directory)
                stat = os.stat(file_path)
                size_mb = stat.st_size / (1024 * 1024)
                file_info.append(f"{relative_path} (サイズ: {size_mb:.2f}MB)")
            except:
                # エラーの場合も相対パスで返す
                relative_path = os.path.relpath(file_path, directory)
                file_info.append(relative_path)
        
        return file_info
    
    def _file_content_preview_wrapper(self, **kwargs) -> str:
        """
        ファイルプレビューのラッパー関数（LLMからの呼び出しに対応）
        """
        file_path = kwargs.get('file_path', '')
        lines = kwargs.get('lines', 10)
        
        # linesが文字列の場合は整数に変換
        if isinstance(lines, str):
            try:
                lines = int(lines)
            except ValueError:
                lines = 10
        
        return self._file_content_preview(file_path, lines)
    
    def _file_content_preview(self, file_path: str, lines: int = 10) -> str:
        """
        ファイルの先頭数行をプレビュー
        
        Args:
            file_path: プレビューするファイルのパス
            lines: 読み込む行数（デフォルト: 10）
            
        Returns:
            ファイルのプレビュー内容
        """
        try:
            # ファイルパスを正規化（data/inputをルートとして扱う）
            file_path = self._normalize_file_path(file_path)
            
            # ファイルの存在確認
            if not os.path.exists(file_path):
                return f"エラー: ファイル '{file_path}' が見つかりません。"
            
            # ファイルサイズの確認
            file_size = os.path.getsize(file_path)
            size_info = f"ファイルサイズ: {file_size / 1024:.2f}KB\n"
            
            # CSVファイルの場合、pandasで読み込み
            if file_path.endswith(('.csv', '.CSV')):
                try:
                    df = pd.read_csv(file_path, nrows=lines)
                    preview = f"{size_info}\n=== CSVファイルのプレビュー ===\n"
                    preview += f"列名: {', '.join(df.columns)}\n"
                    preview += f"行数: {len(df)} (プレビュー)\n"
                    preview += f"\n最初の{lines}行:\n{df.to_string()}"
                    return preview
                except Exception as e:
                    # CSVとして読み込めない場合は通常のテキストとして処理
                    pass
            
            # Excelファイルの場合
            if file_path.endswith(('.xlsx', '.xls', '.XLSX', '.XLS')):
                try:
                    df = pd.read_excel(file_path, nrows=lines)
                    preview = f"{size_info}\n=== Excelファイルのプレビュー ===\n"
                    preview += f"列名: {', '.join(df.columns)}\n"
                    preview += f"行数: {len(df)} (プレビュー)\n"
                    preview += f"\n最初の{lines}行:\n{df.to_string()}"
                    return preview
                except Exception as e:
                    return f"エラー: Excelファイルの読み込みに失敗しました: {str(e)}"
            
            # テキストファイルとして読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content_lines = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.rstrip())
                
                preview = f"{size_info}\n=== ファイルの先頭{lines}行 ===\n"
                preview += '\n'.join(content_lines)
                
                # ファイルがさらに続く場合
                try:
                    next(f)
                    preview += f"\n\n... (ファイルはさらに続きます)"
                except StopIteration:
                    pass
                
                return preview
                
        except UnicodeDecodeError:
            return f"エラー: ファイル '{file_path}' はテキストファイルではないようです。"
        except Exception as e:
            return f"エラー: ファイルの読み込み中にエラーが発生しました: {str(e)}"
    
    def _list_directory(self, directory: str) -> List[str]:
        """
        ディレクトリの内容をリスト
        
        Args:
            directory: リストするディレクトリのパス
            
        Returns:
            ディレクトリ内のファイルとサブディレクトリのリスト
        """
        try:
            # ディレクトリパスを正規化（data/inputをルートとして扱う）
            directory = self._normalize_directory_path(directory)
            
            if not os.path.exists(directory):
                return [f"エラー: ディレクトリ '{directory}' が見つかりません。"]
            
            if not os.path.isdir(directory):
                return [f"エラー: '{directory}' はディレクトリではありません。"]
            
            items = []
            # ディレクトリの内容を取得
            for item in sorted(os.listdir(directory)):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    # ファイルサイズを取得
                    try:
                        size = os.path.getsize(item_path)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size/1024:.1f}KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f}MB"
                        items.append(f"[FILE] {item} ({size_str})")
                    except:
                        items.append(f"[FILE] {item}")
            
            if not items:
                return ["ディレクトリは空です。"]
            
            return items
            
        except PermissionError:
            return [f"エラー: ディレクトリ '{directory}' へのアクセス権限がありません。"]
        except Exception as e:
            return [f"エラー: ディレクトリのリスト中にエラーが発生しました: {str(e)}"] 