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

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# pretrained model + tokenizer
model_path = "./kobert"
tokenizer_path = './TOKENIZER'

word_embedding_model = models.Transformer(model_path, tokenizer_path)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)


train_batch_size = 16
num_epochs = 4
# path 잘 맞춰라.

model_save_path = 'output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# sentence transformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

with open('./KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        dev_samples.append(InputExample(texts= [s1,s2], label=score))

with open('./KorNLUDatasets/KorSTS/tune_test.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        test_samples.append(InputExample(texts= [s1,s2], label=score))

with open('./KorNLUDatasets/KorSTS/tune_train.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        train_samples.append(InputExample(texts= [s1,s2], label=score))

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
# call back이 없음. -> iteration마다 저장해도 될 듯. -> 그래서 평가
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
