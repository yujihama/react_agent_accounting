"""
ReconciliationExpert - 財務データの消込処理に特化した専門家
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Type
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from .base import BaseTaskAgent
from ..states import ReconciliationReActResponse


class ReconciliationExpert(BaseTaskAgent):
    """
    財務データの消込処理に特化した専門家エージェント
    売掛金と入金データのマッチング、消込処理を行う
    """
    
    def get_agent_description(self) -> str:
        """エージェントの説明を返す"""
        return """財務データの消込処理を専門とし、売掛金と入金データの突合、マッチング、消込処理を行う専門家です。

**重要な作業手順:**
1. まず `list_available_artifacts_tool` で利用可能なデータを確認
2. `get_artifact_data_tool` で売掛金データと入金データを取得（キーワード: '売掛金', 'billing', '入金', 'deposit'）
3. データの内容を確認し、突合キーを決定
4. `perform_reconciliation_tool` で消込処理を実行
5. 必要に応じて結果をCSVで出力

**データ取得のコツ:**
- 前のタスクで取得されたデータはartifactsに保存されています
- '売掛金データ'、'売掛金'、'billing'のいずれかで売掛金データを取得
- '入金データ'、'入金'、'deposit'のいずれかで入金データを取得"""
    
    def get_result_format(self) -> str:
        """
        エージェントの結果のフォーマットを返す
        """
        return "消込済みデータと未消込データのパスまたはデータ"
    
    def get_response_model(self) -> Type[ReconciliationReActResponse]:
        """
        ReconciliationExpert専用のレスポンスモデルを返す
        消込状況やエラー件数を含む追加フィールドを持つ
        """
        return ReconciliationReActResponse
    
    def get_tools(self) -> List[Tool]:
        """ReconciliationExpert固有のツールを返す（共通ツールを含む）"""
        return self.get_common_tools() + [
            Tool(
                name="load_data_tool",
                func=self._load_data,
                description="指定されたパスのCSVまたはExcelファイルからデータをpandas DataFrameとしてロードする。\n引数:\n- file_path (str): ロードするファイルのパス（.csv、.xlsx、.xls形式対応）"
            ),
            Tool(
                name="get_artifact_data_tool",
                func=self._get_artifact_data,
                description="前のタスクで取得されたデータをartifactsから取得する。\n引数:\n- artifact_key (str): 取得したいデータのキー（例：'売掛金データ', '入金データ', 'billing', 'deposit'）"
            ),
            Tool(
                name="list_available_artifacts_tool", 
                func=self._list_available_artifacts,
                description="利用可能なartifacts（前のタスクで生成されたデータ）の一覧を表示する。\n引数: なし"
            ),
            Tool(
                name="find_matching_keys_tool",
                func=self._find_matching_keys,
                description="二つのDataFrameのプレビュー（列名、データ型、サンプル値）をLLMに提示し、意味的に最も一致する可能性が高い突合キーのペアを推論して返す。\n引数:\n- df1_preview (dict): 第一のDataFrameのプレビューデータ\n- df2_preview (dict): 第二のDataFrameのプレビューデータ"
            ),
            Tool(
                name="perform_reconciliation_tool",
                func=self._perform_reconciliation,
                description="二つのCSVファイルを指定されたキーでマージし、消込処理を実行する。消込済みデータと未消込データを返す。\n引数:\n- deposit_file_path (str): 入金データのCSVファイルパス（data/input配下の相対パス）\n- billing_file_path (str): 売掛金データのCSVファイルパス（data/input配下の相対パス）\n- join_keys (list): 結合に使用するキーのリスト"
            ),
            Tool(
                name="output_csv_tool",
                func=self._output_csv,
                description="DataFrameを指定されたパスにCSV形式で出力し、出力したパスを返す。\n引数:\n- df (DataFrame): 出力するpandas DataFrame\n- output_path (str): 出力先のファイルパス"
            ),
            Tool(
                name="analyze_data_tool",
                func=self._analyze_data,
                description="DataFrameの基本的な統計情報と構造を分析し、サマリーを返す。\n引数:\n- df (DataFrame): 分析対象のpandas DataFrame"
            )
        ]
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        CSVまたはExcelファイルからデータをロード
        
        Args:
            file_path: ロードするファイルのパス
            
        Returns:
            ロードしたDataFrame
        """
        try:
            file_path = os.path.join(os.getcwd(), "data", "input", file_path.lstrip('/'))
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイル '{file_path}' が見つかりません。")
            
            # ファイル拡張子に基づいて適切な読み込み方法を選択
            if file_path.endswith(('.csv', '.CSV')):
                # エンコーディングを試行
                encodings = ['utf-8', 'shift_jis', 'cp932', 'utf-8-sig']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"ファイル '{file_path}' を {encoding} エンコーディングで読み込みました。")
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"ファイル '{file_path}' のエンコーディングを特定できませんでした。")
                
            elif file_path.endswith(('.xlsx', '.xls', '.XLSX', '.XLS')):
                df = pd.read_excel(file_path)
                print(f"Excelファイル '{file_path}' を読み込みました。")
                return df
            else:
                raise ValueError(f"サポートされていないファイル形式です: {file_path}")
                
        except Exception as e:
            raise Exception(f"データのロード中にエラーが発生しました: {str(e)} ※ファイルのパスを渡しているか確認してください。")
    
    def _get_artifact_data(self, artifact_key: str) -> Any:
        """
        required_artifactsから指定されたキーのデータを取得
        
        Args:
            artifact_key: 取得したいデータのキー
            
        Returns:
            指定されたキーに対応するデータ
        """
        try:
            # BaseTaskAgent内からrequired_artifactsにアクセス
            if hasattr(self, '_current_state') and self._current_state:
                artifacts = self._current_state.required_artifacts
                print(f"全artifacts取得: {list(artifacts.keys())}")
                
                # 現在のタスクの依存関係を確認
                if self._current_state.task_to_perform and self._current_state.task_to_perform.dependencies:
                    print(f"現在のタスクの依存関係: {self._current_state.task_to_perform.dependencies}")
                    
                    # 依存タスクIDを基にartifactsを探す
                    for dep_id in self._current_state.task_to_perform.dependencies:
                        dep_artifact_key = f"{dep_id}_result"
                        if dep_artifact_key in artifacts:
                            dep_data = artifacts[dep_artifact_key]
                            print(f"依存タスク {dep_id} のデータ: タイプ {type(dep_data)}")
                            
                            if isinstance(dep_data, list) and len(dep_data) > 0:
                                sample = dep_data[0]
                                if isinstance(sample, dict):
                                    sample_keys = list(sample.keys())
                                    print(f"依存タスクデータのサンプルキー: {sample_keys}")
                                    
                                    # 売掛金データの特徴チェック  
                                    if artifact_key in ['売掛金データ', '売掛金', 'billing']:
                                        if any(k in sample_keys for k in ['billing_id', 'invoice_id']) and 'amount' in sample_keys:
                                            print(f"売掛金データを依存タスクから発見: {dep_id}")
                                            self.set_communication_to_supervisor(f"売掛金データを依存タスク {dep_id} から正常に取得しました（{len(dep_data)}件）")
                                            return pd.DataFrame(dep_data)
                                    
                                    # 入金データの特徴チェック
                                    elif artifact_key in ['入金データ', '入金', 'deposit']:
                                        if any(k in sample_keys for k in ['deposit_id', 'payment_id']) and 'amount' in sample_keys:
                                            print(f"入金データを依存タスクから発見: {dep_id}")
                                            return pd.DataFrame(dep_data)
                
                # キーに完全一致するものを探す
                if artifact_key in artifacts:
                    return artifacts[artifact_key]
                
                # 部分一致でも探す（売掛金、入金などのキーワードで）
                for key, value in artifacts.items():
                    if artifact_key in key or any(keyword in key for keyword in ['売掛', '入金', 'billing', 'deposit']):
                        if artifact_key in ['売掛金データ', '売掛金', 'billing'] and ('billing' in key or '売掛' in key):
                            print(f"売掛金データを発見: キー '{key}' からデータを取得")
                            return value
                        elif artifact_key in ['入金データ', '入金', 'deposit'] and ('deposit' in key or '入金' in key):
                            print(f"入金データを発見: キー '{key}' からデータを取得")
                            return value
                
                # リストの形式で保存されている場合の処理
                for key, value in artifacts.items():
                    if isinstance(value, list) and len(value) > 0:
                        # 売掛金データの特徴を確認
                        if artifact_key in ['売掛金データ', '売掛金', 'billing']:
                            sample = value[0]
                            if isinstance(sample, dict) and any(k in sample for k in ['billing_id', 'amount', 'due_date']):
                                print(f"売掛金データを発見: キー '{key}' からリストデータを変換")
                                return pd.DataFrame(value)
                        
                        # 入金データの特徴を確認
                        elif artifact_key in ['入金データ', '入金', 'deposit']:
                            sample = value[0]
                            if isinstance(sample, dict) and any(k in sample for k in ['deposit_id', 'deposit_date', 'amount']):
                                print(f"入金データを発見: キー '{key}' からリストデータを変換")
                                return pd.DataFrame(value)
                
                # タスクIDベースでartifactsを検索（より詳細な検索）
                print(f"詳細検索: 利用可能なartifacts keys: {list(artifacts.keys())}")
                for key, value in artifacts.items():
                    print(f"キー '{key}': タイプ {type(value)}, サイズ {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if isinstance(value, list) and len(value) > 0:
                        sample = value[0]
                        print(f"  サンプル: {sample}")
                        if isinstance(sample, dict):
                            sample_keys = list(sample.keys())
                            print(f"  サンプルキー: {sample_keys}")
                            
                            # 売掛金データの詳細チェック
                            if artifact_key in ['売掛金データ', '売掛金', 'billing']:
                                if any(k in sample_keys for k in ['billing_id', 'invoice_id']) and 'amount' in sample_keys:
                                    print(f"売掛金データを発見: キー '{key}' からリストデータを変換")
                                    return pd.DataFrame(value)
                            
                            # 入金データの詳細チェック
                            elif artifact_key in ['入金データ', '入金', 'deposit']:
                                if any(k in sample_keys for k in ['deposit_id', 'payment_id']) and 'amount' in sample_keys:
                                    print(f"入金データを発見: キー '{key}' からリストデータを変換")
                                    return pd.DataFrame(value)
                
                # 利用可能なキーを表示
                available_keys = list(artifacts.keys())
                
                # Supervisorに状況を報告
                self.set_communication_to_supervisor(
                    f"データ取得に問題があります。期待されたキー '{artifact_key}' がartifactsに見つかりません。"
                    f"利用可能なキー: {available_keys}。前のタスクでデータ取得が適切に行われていない可能性があります。"
                )
                
                return f"指定されたキー '{artifact_key}' が見つかりません。利用可能なキー: {available_keys}"
            else:
                return "required_artifactsにアクセスできません。"
                
        except Exception as e:
            return f"artifact取得中にエラーが発生しました: {str(e)}"
    
    def _list_available_artifacts(self) -> str:
        """
        利用可能なartifactsの一覧を表示
        
        Returns:
            利用可能なartifactsの説明
        """
        try:
            if hasattr(self, '_current_state') and self._current_state:
                artifacts = self._current_state.required_artifacts
                
                if not artifacts:
                    return "利用可能なartifactsがありません。"
                
                print(f"[debug]利用可能なartifacts: {artifacts}")
                result = "利用可能なartifacts:\n"
                for key, value in artifacts.items():
                    if isinstance(value, list):
                        result += f"- {key}: リスト形式のデータ (件数: {len(value)})\n"
                        if len(value) > 0 and isinstance(value[0], dict):
                            sample_keys = list(value[0].keys())
                            result += f"  サンプル列名: {sample_keys}\n"
                    elif isinstance(value, pd.DataFrame):
                        result += f"- {key}: DataFrame (行数: {len(value)}, 列数: {len(value.columns)})\n"
                        result += f"  列名: {list(value.columns)}\n"
                    else:
                        result += f"- {key}: {value[:100]}\n"
                
                return result
            else:
                return "required_artifactsにアクセスできません。"
                
        except Exception as e:
            return f"artifacts一覧取得中にエラーが発生しました: {str(e)}"
    
    def _find_matching_keys(self, df1_preview: Dict, df2_preview: Dict) -> List[Tuple[str, str]]:
        """
        二つのDataFrameのプレビューから突合キーを推論
        
        Args:
            df1_preview: 最初のDataFrameのプレビュー情報
            df2_preview: 二番目のDataFrameのプレビュー情報
            
        Returns:
            推奨される突合キーのペアのリスト [(df1_column, df2_column), ...]
        """
        # LLMを使用してキーのマッチングを推論
        llm = ChatOpenAI(
            model=self.config['common']['llm_model'],
            temperature=0
        )
        
        prompt = f"""
        以下の2つのデータセットの情報から、突合（マッチング）に使用すべき列のペアを推論してください。
        
        データセット1:
        列名: {df1_preview.get('columns', [])}
        データ型: {df1_preview.get('dtypes', {})}
        サンプル値: {df1_preview.get('sample_values', {})}
        
        データセット2:
        列名: {df2_preview.get('columns', [])}
        データ型: {df2_preview.get('dtypes', {})}
        サンプル値: {df2_preview.get('sample_values', {})}
        
        売掛金と入金データの突合では、通常以下のような列が使用されます：
        - 取引先コード、顧客コード
        - 請求番号、伝票番号
        - 金額
        - 日付
        
        最も適切な突合キーのペアを、以下の形式で返してください：
        [(データセット1の列名, データセット2の列名), ...]
        
        例: [("顧客コード", "取引先コード"), ("請求金額", "入金額")]
        """
        
        response = llm.invoke(prompt)
        
        # レスポンスをパース
        try:
            import ast
            key_pairs = ast.literal_eval(response.content.strip())
            return key_pairs
        except:
            # パースに失敗した場合、デフォルトの推論を行う
            return self._default_key_matching(df1_preview, df2_preview)
    
    def _default_key_matching(self, df1_preview: Dict, df2_preview: Dict) -> List[Tuple[str, str]]:
        """デフォルトのキーマッチングロジック"""
        df1_cols = df1_preview.get('columns', [])
        df2_cols = df2_preview.get('columns', [])
        
        key_pairs = []
        
        # よくあるキー名のマッピング
        common_mappings = {
            '顧客コード': ['取引先コード', '得意先コード', 'customer_code'],
            '取引先コード': ['顧客コード', '得意先コード', 'customer_code'],
            '請求番号': ['伝票番号', 'invoice_no', 'bill_no'],
            '伝票番号': ['請求番号', 'invoice_no', 'bill_no'],
            '金額': ['請求金額', '入金額', 'amount'],
            '請求金額': ['金額', '入金額', 'amount'],
            '入金額': ['金額', '請求金額', 'amount']
        }
        
        # 列名の類似性でマッチング
        for col1 in df1_cols:
            if col1 in common_mappings:
                for col2 in df2_cols:
                    if col2 in common_mappings[col1]:
                        key_pairs.append((col1, col2))
                        break
        
        return key_pairs if key_pairs else [(df1_cols[0], df2_cols[0])]
    
    def _perform_reconciliation(self, deposit_file_path: str, billing_file_path: str, 
                              join_keys: List[str]) -> Dict[str, pd.DataFrame]:
        """
        消込処理を実行
        
        Args:
            deposit_file_path: 入金データのCSVファイルパス（data/input配下の相対パス）
            billing_file_path: 売掛金データのCSVファイルパス（data/input配下の相対パス）
            join_keys: 突合に使用するキーのリスト
            
        Returns:
            消込済みと未消込のDataFrameを含む辞書
        """
        try:
            # ファイルパスからDataFrameを読み込み
            deposit_df = self._load_data(deposit_file_path)
            billing_df = self._load_data(billing_file_path)
            
            # タイムスタンプを追加
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # データのコピーを作成
            deposit_work = deposit_df.copy()
            billing_work = billing_df.copy()
            
            # 処理フラグを追加
            deposit_work['_消込状態'] = '未消込'
            billing_work['_消込状態'] = '未消込'
            
            # インデックスを保持
            deposit_work['_入金側インデックス'] = deposit_work.index
            billing_work['_売掛側インデックス'] = billing_work.index
            
            # キーが複数の場合の処理
            if isinstance(join_keys, list) and len(join_keys) > 0:
                if isinstance(join_keys[0], tuple):
                    # [(df1_col, df2_col), ...] 形式の場合
                    left_keys = [k[0] for k in join_keys]
                    right_keys = [k[1] for k in join_keys]
                else:
                    # [key1, key2, ...] 形式の場合
                    left_keys = right_keys = join_keys
            else:
                raise ValueError("突合キーが指定されていません。")
            
            # マージ実行
            merged = pd.merge(
                deposit_work,
                billing_work,
                left_on=left_keys,
                right_on=right_keys,
                how='outer',
                indicator=True,
                suffixes=('_入金', '_売掛')
            )
            
            # 消込状態を更新
            reconciled_mask = merged['_merge'] == 'both'
            merged.loc[reconciled_mask, '_消込状態_入金'] = '消込済'
            merged.loc[reconciled_mask, '_消込状態_売掛'] = '消込済'
            
            # 消込済みデータ
            reconciled = merged[reconciled_mask].copy()
            reconciled['_消込日時'] = timestamp
            
            # 未消込データ（入金側）
            unreconciled_deposit = merged[merged['_merge'] == 'left_only'].copy()
            deposit_cols = [col for col in deposit_df.columns] + ['_消込状態', '_消込理由']
            unreconciled_deposit['_消込理由'] = '対応する売掛データなし'
            
            # 未消込データ（売掛側）
            unreconciled_billing = merged[merged['_merge'] == 'right_only'].copy()
            billing_cols = [col for col in billing_df.columns] + ['_消込状態', '_消込理由']
            unreconciled_billing['_消込理由'] = '対応する入金データなし'
            
            # 結果を整理
            result = {
                'reconciled': reconciled,
                'unreconciled_deposit': unreconciled_deposit[deposit_cols] if deposit_cols else unreconciled_deposit,
                'unreconciled_billing': unreconciled_billing[billing_cols] if billing_cols else unreconciled_billing,
                'summary': {
                    'total_deposit': len(deposit_df),
                    'total_billing': len(billing_df),
                    'reconciled_count': len(reconciled),
                    'unreconciled_deposit_count': len(unreconciled_deposit),
                    'unreconciled_billing_count': len(unreconciled_billing),
                    'reconciliation_rate': len(reconciled) / max(len(deposit_df), len(billing_df)) * 100
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"消込処理中にエラーが発生しました: {str(e)}")
    
    def _output_csv(self, df: pd.DataFrame, output_path: str) -> str:
        """
        DataFrameをCSVファイルとして出力
        
        Args:
            df: 出力するDataFrame
            output_path: 出力先のパス
            
        Returns:
            出力したファイルのパス
        """
        try:
            # 出力ディレクトリが存在しない場合は作成
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # CSVとして出力
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            # ファイルサイズを確認
            file_size = os.path.getsize(output_path)
            
            return f"ファイルを出力しました: {output_path} (サイズ: {file_size/1024:.2f}KB, 行数: {len(df)})"
            
        except Exception as e:
            raise Exception(f"CSV出力中にエラーが発生しました: {str(e)}")
    
    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        DataFrameの基本的な分析を実行
        
        Args:
            df: 分析するDataFrame
            
        Returns:
            分析結果の辞書
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_values': {}
        }
        
        # 各列のサンプル値を取得
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                # 文字列型の場合、ユニーク値の上位5個
                unique_vals = df[col].value_counts().head(5).index.tolist()
                analysis['sample_values'][col] = unique_vals
            else:
                # 数値型の場合、統計情報
                analysis['sample_values'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean() if df[col].dtype in ['int64', 'float64'] else None
                }
        
        return analysis 