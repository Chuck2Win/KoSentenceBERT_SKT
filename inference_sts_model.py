# inference sts model

from sentence_transformers import SentenceTransformer, SentenceDataset, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

# evaluation
pretrained_model_weight = '/home/work/ok/KoSentenceBERT_SKT/output/training_stsbenchmark_skt_bert-2021-09-03_01-55-05/0_Transformer/result.pt'
model_path = './kobert'
tokenizer_path = './TOKENIZER'

word_embedding_model = models.Transformer(model_path, tokenizer_path)
word_embedding_model.auto_model.load_state_dict()

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


test_samples = []
with open('./KorNLUDatasets/KorSTS/tune_test.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        test_samples.append(InputExample(texts= [s1,s2], label=score))
# model 저장됬던 위치.
model_save_path = model_save_path = os.path.split(os.path.split(pretrained_model_weight)[0])[0]
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)ㅇㅇ

