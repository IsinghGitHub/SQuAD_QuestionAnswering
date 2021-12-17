# -*- coding: utf-8 -*-
"""
Created on Friday 17th Dec- 2021

@author: Indrajit Singh
"""

import pandas as pd
import numpy as np 
import torch 
import streamlit as st


from transformers import pipeline,QuestionAnsweringPipeline, DistilBertForQuestionAnswering,AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# The model_path here would be the directory in which you saved the model using the HuggingFace model.save_pretrained() function
model_path = "QA_Model"
myQAModel = DistilBertForQuestionAnswering.from_pretrained(model_path)

QAPipeline = QuestionAnsweringPipeline(model = myQAModel,tokenizer = tokenizer)

# This is a markdown message at the beginning of my application in which I'm introducing myself and explaining the question. You should add whatever message you want to. 
st.markdown("Project Devdoot testing")

st.markdown("In America Vivekananda's mission was the interpretation of India's spiritual culture,especially in its Vedantic setting.")


context = st.text_area("Context Paragraph", "")
question = st.text_input("Question", "")

if context:
    # Execute question against paragraph
    if question:
        outputs = QAPipeline(question = question,context = context,topk = 3, max_seq_len = 512)
        answer = outputs[0]["answer"]
        output_answer = st.text_area("Answer",answer)
        
