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
from tqdm import tqdm
import pickle

# inference
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

data_path = '/home/work/ok/preprocessed_data'
gom = [i for i in os.listdir(data_path) if i.endswith('txt')]

# file read
Gom = []
for i in tqdm(gom):
    try:
        f = open(os.path.join('./preprocessed_data',i),'r',encoding='utf-8')
        d = f.readlines()  
        dd = ''.join(d).replace('\n',' ').strip()
        Gom.append(dd)
    except:
        continue

        
# sentence embedding
# 2분소요
corpus_embeddings = embedder.encode(Gom, convert_to_tensor=True, show_progress_bar=True, is_pretokenized = False, num_workers=64, batch_size = 512)

# cosine similarity 계산
now = time.time()
Cos_scores = []
for query,name in tqdm(zip(corpus_embeddings,gom)):
    cos_scores = util.pytorch_cos_sim(query, corpus_embeddings)[0]
    Cos_scores.append(cos_scores.cpu().tolist())
print(time.time()-now) # 20분 소요


pickle.dump(Cos_scores,open('preprocessed_data_cosine_sim','wb'))

# graph화 하기
# 시간 복잡도 N**2

def make_graph(Cos_scores,gom,threshold):
    graph = {}
    for i in tqdm(range(len(Cos_scores))):
        graph[int(gom[i])]=[int(gom[j]) for j in np.where(np.array(Cos_scores[i])>threshold)[0].tolist()]
        graph[int(gom[i])].remove(int(gom[i]))
    return graph
threshold = 0.9
graph = make_graph(Cos_scores, gom,threshold)
pickle.dump(graph,open('preprocessed_data_graph_threshold_0.9','wb'))
