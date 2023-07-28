# -*- coding: utf-8 -*-
import gradio as gr
from paddlenlp import Taskflow
import pandas as pd
import pymysql
from tqdm import tqdm
import time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import os

os.environ['CURL_CA_BUNDLE'] = ''

model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
QA = pipeline('question-answering', model=model, tokenizer=tokenizer)


# ie = Taskflow('information_extraction', model="uie-medical-base")
# model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
# tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
# QA = pipeline('question-answering', model=model, tokenizer=tokenizer)


def fun(qa, text):
    # ie.set_schema(schema)
    QA_input = {'question': qa,
                'context': text}
    result = QA(QA_input)
    # result = ie(text)[0][schema]
    return result


eg_text1 = '科别:二病区（妇产科）性别：女年龄：32岁职业：家务 主诉：发现外阴赘生物1+月。入院情况：患者既往月经规则，周期30+/-天，经期7天，月经量中，无痛经，末次月经2022年11月20日，量及性状如常，现月经干净3天。1+月前发现外阴赘生物，来院检查行宫颈HPV+TCT无殊，后渐增大，无瘙痒等不适，今来院检查，考虑外阴赘生物性质待查，建议行赘生物摘除，既往体健，否认肝炎、菌痢、结核等传染性疾病史，否认心脏病、糖尿病、哮喘等慢性疾病史。既往剖宫产3次，恢复可，已结扎，否认其他手术、外伤及输血史，否认药物、食物等过敏史，饮酒史1-年，吸烟史1-年，每天半包，否认冶游史，2个性伴侣，19岁开始性生活，生育史：3-0-0-3(2008年及2015年及2018年各剖宫产一孩，均体健)。否认近期与新型冠状病毒感染者，或与疑似患者接触史；否认近半月离开玉环，内有云南德宏州、江苏南京、湖南张家界等新冠中高风险地区或境外旅居史。否认本人在医学隔离期内；否认冷链食物接触史。体格检查：体温36.7℃,脉搏81次/分,呼吸19次/分,血压96/61mmHg,身高160CM,体重50KG，心肺腹无殊，双下肢无浮肿，NS（-）。妇检：会阴后联合见散在粟粒样疣样赘生物，质软，无压痛，色淡红，阴道壁未见明显，宫颈尚光，无触血及赘生物，未内诊；辅助检查：2022年11月宫颈筛查无殊；门诊2022-11-29白带常规+BV(12)清洁度：Ⅱ度，白带霉菌：未见，白带滴虫：未见。2022-11-28日核酸结果阴性。初步诊断：外阴尖锐湿疣；瘢痕子宫（三次）；双输卵管结扎术后诊疗经过：入院完善各项相关检查，于2022年11月29日行外阴赘生物摘除术，术程顺利，出血不多标本送病检，术后予抗炎及中药清热凉血止血治疗。出院情况：一般情况良好，无腹痛，腹胀，生命体征平稳，心肺听诊无殊，无阴道出血。出院诊断：外阴尖锐湿疣；瘢痕子宫（三次）；双输卵管结扎术后出院医嘱：1.注意事项：注意休息，注意体温变化。加强营养，不适随诊，术后如发现阴道出血多于月经，阴道分泌物有异味，应立即到医院诊治。禁止性生活及盆浴1个月。2.复诊时间：1周后我院查询病理检查结果；87255311。3.出院带药：头孢呋辛酯片1盒1片，一日二次，口服;中药三贴1包外洗'
# iface = gr.Interface(predict_image, inputs=inputs_1, outputs="label", title=title, description=description, article=article)


with gr.Blocks() as demo:
    # Input
    question = gr.Textbox(label='问题')
    text = gr.Textbox(label='文本')

    # Output
    ans = gr.Textbox(label='提取结果')
    btn = gr.Button(value="Submit")
    btn.click(fun, inputs=[question, text], outputs=[ans])

    gr.Markdown("## Text Examples")
    gr.Examples(
        [["生育史是什么", eg_text1],
         ["做了什么手术?", eg_text1],
         ['出院带了什么药？', eg_text1],
         ['清洁度是几度', eg_text1]],
        [question, text],
        ans,
        fun,
        cache_examples=False,
    )
if __name__ == "__main__":
    gr.close_all()
    demo.launch(server_name='0.0.0.0', server_port=15534)
