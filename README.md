# 数据库表字段信息提取

---

### 描述
基于gradio、QA模型的的端到端的信息提取web工具，用以批量提取在数据表某个文本字段的关键信息；例如：身高、体重、年龄、检验检查数据（结节尺寸、含量）等等
### 使用方法
### 1.安装依赖
- gradio
- pandas
- pymysql
- tqdm
- transformers
### 2.运行命令
```bash
python main.py
```
浏览器进入127.0.0.1:15534或者 本机ip:155534

### 3.界面信息填充
在web界面依次填入以下信息：
- 数据库配置信息（host,user,password,port,database)
- 数据表名(read table)
- 主键(primary key)
- 提取的源字段名（read column)
- 结果写入的表名(write table)
- 结果写入的字段名(write column)

**注意：需提前将提取结果表/字段提前创建好，再运行；若结果需写入新表，必须保证新表(write table)与
源表(read table)在同一个数据库(database)中**

### 4.运行
点击Submit后,等待出现'successfull'后即运行成功；若信息填写错误，点击Clear清空全部字段重新填入