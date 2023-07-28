import gradio as gr
from paddlenlp import Taskflow
import pandas as pd
import pymysql
from tqdm import tqdm
import time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


use_unicode = True
charset = 'utf8mb4'
# extractor = Taskflow('information_extraction', model="uie-medical-base")
model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
extractor = pipeline('question-answering', model=model, tokenizer=tokenizer)


class DB:

    def __init__(self, host, port, user, password, db) -> None:
        '''
        数据库连接
        '''
        self.conn = pymysql.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=db,
            cursorclass=pymysql.cursors.SSCursor,
            charset=charset,
            use_unicode=use_unicode
        )

    def fetch_data(self, read_table, p_k, column):
        sql = f"""select {p_k},{column} from {read_table}"""
        print(sql)
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall_unbuffered()
            all_dfs = []
            for record in tqdm(result, desc='fetching'):
                all_dfs.append(record)
                # pbar.update(1)
            print("finish fetching")
            return pd.DataFrame(all_dfs)
            # return pd.Series(data=all_dfs)
            # return pd.DataFrame(data=result, columns=columns)

    def write_result(self, series, table, p_k, column):
        # sql = f''' INSERT INTO {table}''' \
        #       + f''' ({column})''' + f''' VALUES (%s)'''
        sql = f'''UPDATE {table} SET {column}=%s WHERE {p_k}= %s '''
        print(sql)
        # Create a cursor and begin a transaction
        cursor = self.conn.cursor()
        cursor.execute('BEGIN')
        # Split the data into batches of 100 rows
        batch_size = 100
        num_batches = series.shape[0] // batch_size + (1 if series.shape[0] % batch_size != 0 else 0)
        # Use tqdm to add a progress bar to the insertion process
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, series.shape[0])
            batch = series.iloc[start_idx:end_idx]
            to_write = []
            for row in batch.values.tolist():
                to_write.append(row)
            cursor.executemany(sql, to_write)

            # commit the changes
            self.conn.commit()
        # close the connection
        self.conn.close()





def fun(host, user, password, port, db, schema, read_table, p_k, read_column, write_table, write_column,
          progress=gr.Progress()):
    database = DB(host, port, user, password, db)
    QA_input = {'question': schema,
                'context': None}
    progress(0, desc="Start Fetching")
    time.sleep(2)
    print("start fetching")
    data = database.fetch_data(read_table, p_k, read_column)
    print(data.info())
    results = []
    for item in progress.tqdm(data[1], desc='processing'):
        try:
            QA_input['context'] = item
            results.append(extractor(QA_input)['answer'])
        except (KeyError, TypeError) as err:
            results.append(None)
    results = pd.Series(results)
    new_df = pd.concat([results, data[0]], axis=1)
    print(new_df.info())
    database.write_result(new_df, write_table, p_k, write_column)
    return "successfull"


if __name__ == "__main__":
    # Input
    # sql = "select id,txt from ner_test"
    host = gr.Textbox(label='host', value='127.0.0.1')
    user = gr.Textbox(label='user', value='root')
    password = gr.Textbox(label='password', value='xxxxxx')
    port = gr.Textbox(label='port', value='3306')
    db = gr.Textbox(label='database', value='test')
    schema = gr.Textbox(label='提问信息', value='结节尺寸是多少？')
    read_table = gr.Textbox(label='read table')
    p_k = gr.Textbox(label='primary key')
    read_column = gr.Textbox(label='read column')
    write_table = gr.Textbox(label='write table')
    write_column = gr.Textbox(label='write column')
    inputs = [host, user, password, port, db, schema, read_table, p_k, read_column, write_table, write_column]

    # Output
    text = gr.Textbox(label='result')

    outputs = [text]

    # iface = gr.Interface(predict_image, inputs=inputs_1, outputs="label", title=title, description=description, article=article)
    iface = gr.Interface(fun, inputs=inputs, outputs=outputs, title="Info extract tool")
    gr.close_all()
    iface.queue().launch(server_name='0.0.0.0', server_port=15534)
    # iface.launch()
