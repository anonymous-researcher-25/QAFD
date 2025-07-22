from copy import deepcopy
import threading
import sqlite3
import pymysql
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut

class ExecutionAccuracy:
    """
    Tool for evaluating the predicted SQL queries against the ground truth SQL query.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self, pred_sqls, gold_results, db_names):
        self.evaluation_results = {}
        evaluation_keys = list(pred_sqls.keys())

        for question in evaluation_keys:
            try:
                predicted_sql = pred_sqls[question]
                gold_result = pd.read_csv(gold_results[question])
                db_name = db_names[question]

                response = self.compare_results(
                    predicted_sql=predicted_sql,
                    ground_truth=gold_result,
                    db_name=db_name
                )

                evaluation_result = {
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                }

            except Exception as e:
                predicted_sql = "--error--"
                evaluation_result = {
                    "exec_res": "error",
                    "exec_err": str(e),
                }

            evaluation_result.update({
                "gold_result": gold_result,
                "PREDICTED_SQL": predicted_sql
            })
            self.evaluation_results[question] = evaluation_result
            
        return self.evaluation_results

    def compare_results(self, predicted_sql, ground_truth, db_name):
        if self.args['data_name'] == 'bird':
            predicted_sql = predicted_sql.replace('\n', ' ').replace('"', "'").strip("`.")
        
        try:
            if self.args['gold_type'] == 'sql' and predicted_sql == ground_truth:
                res = 1
            else:
                res = func_timeout(60, self._compare_results_outcomes, args=(predicted_sql, ground_truth, db_name))
            error = "incorrect answer" if res == 0 else "--"
        except FunctionTimedOut:
            print("Comparison timed out.")
            error = "timeout"
            res = 0
        except Exception as e:
            print(f"Error in compare_results: {e}")
            error = str(e)
            res = 0
        return {'exec_res': res, 'exec_err': error}
    
    def _compare_results_outcomes(self, predicted_sql, ground_truth, db_name):
        try:
            predicted_res = self.execute_sql(predicted_sql, db_name)
            if self.args['gold_type'] == 'sql':
                ground_truth_res = self.execute_sql(ground_truth, db_name)
            elif self.args['gold_type'] == 'csv':
                ground_truth_res = ground_truth

            if predicted_res.shape != ground_truth_res.shape:
                return False
            else:
                return self.check_dataframes(predicted_res, ground_truth_res)
        
        except Exception as e:
            print(f"Error comparing SQL outcomes: {e}")
            raise e

    def execute_sql(self, sql, db_name, timeout = 60):
        class QueryThread(threading.Thread):
            def __init__(self, args, sql, db_name):
                threading.Thread.__init__(self)
                self.args = args
                self.sql = sql
                self.db_name = db_name
                self.result = None
                self.exception = None

            def run(self):
                try:
                    conn, curs = self.connect_db(self.db_name)
                    curs.execute(self.sql)
                    self.result = pd.DataFrame(curs.fetchall())

                    curs.close()
                    conn.close()
                except Exception as e:
                    self.exception = e
                    
            def connect_db(self, db_name):
                if self.args['db_type'] == 'Mariadb':
                    conn = pymysql.connect(
                        host=self.args['host'],
                        user=self.args['userid'],
                        password=self.args['pwd'],
                        db=db_name,
                        charset='utf8'
                    )
                    curs = conn.cursor(pymysql.cursors.DictCursor)
                elif self.args['db_type'] == 'sqlite3':
                    if self.args['data_name'] == 'bird':
                        db = f"{self.args['db_path']}{db_name}/{db_name}.sqlite"
                    elif self.args['data_name'] == 'spider2':
                        db = f"{self.args['db_path']}{db_name}.sqlite"

                    conn = sqlite3.connect(db)
                    curs = conn.cursor()
                
                return conn, curs

        query_thread = QueryThread(self.args, sql, db_name)
        query_thread.start()
        query_thread.join(timeout)
        if query_thread.is_alive():
            raise TimeoutError(f"SQL query execution exceeded the timeout of {timeout} seconds.")
        if query_thread.exception:
            raise query_thread.exception
        return query_thread.result
    
    def check_dataframes(self, df1, df2):
        cols1 = df1.columns
        cols2 = df2.columns
        used_cols2 = set()
        for col1 in cols1:
            res = 0
            if df1[col1].dtype == 'float':
                s1 = sorted(deepcopy(df1[col1].round(2)))
            elif df1[col1].dtype == 'object':
                try:
                    s1 = sorted(deepcopy(df1[col1].astype('int64')))
                except:
                    s1 = sorted(deepcopy(df1[col1]))
            else:
                s1 = sorted(deepcopy(df1[col1]))

            for col2 in cols2:
                if col2 in used_cols2:
                    res += 1
                    continue

                if df2[col2].dtype == 'float':
                    s2 = sorted(deepcopy(df2[col2].round(2)))
                elif df2[col2].dtype == 'object':
                    try:
                        s2 = sorted(deepcopy(df2[col2].astype('int64')))
                    except:
                        s2 = sorted(deepcopy(df2[col2]))
                else:
                    s2 = sorted(deepcopy(df2[col2]))
                    
                if s1 == s2:
                    used_cols2.add(col2)
                    break
                else:
                    res += 1

            if res >= len(cols2):
                return False
        return True
