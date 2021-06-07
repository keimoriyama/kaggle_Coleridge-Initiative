import pandas as pd
from utils import clean_text
import glob
from tqdm import tqdm

from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize

from more_itertools import chunked

MAX_LENGTH = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
path = "../input/train.csv"

df = pd.read_csv(path)

dataset_labels = list(set([clean_text(x) for x in df['dataset_title']]))
dataset_labels += list(set([clean_text(x) for x in df['dataset_label']]))
dataset_labels += list(set([clean_text(x) for x in df['cleaned_label']]))
dataset_labels = list(set(dataset_labels))

tokenized_dataset_labels = [tokenizer.tokenize(x) for x in dataset_labels]
train_fiels = glob.glob("../input/train/*.json")
# print(pd.read_json(train_fiels[0]))

data = []
df_t = pd.DataFrame()
for json_file in tqdm(train_fiels,total=len(train_fiels)):
    # テキストの入手
    df = pd.read_json(json_file)
    texts = df['text']
    for text in texts:
        # 文章単位に分割
        sentences = sent_tokenize(text)
        # 文章毎の処理
        for sentence in sentences:
            #print(sentence)
            sentence = clean_text(sentence)
            # データセットのラベル名が含まれるかどうかを判定
            tokenized_sentence = tokenizer.tokenize(sentence)
            # subword後の文章にラベルが含まれるかどうかを探索したい
            for i in range(len(tokenized_dataset_labels)):
                tokenized_dataset_label = tokenized_dataset_labels[i]
                dataset_label = dataset_labels[i]
                find = False
                bio = []
                label_index = 0
                if dataset_label in sentence:
                    for i in range(len(tokenized_sentence)):
                        if label_index == len(tokenized_dataset_label):
                            find = True
                        elif label_index < len(tokenized_dataset_label):
                            if tokenized_sentence[i] == tokenized_dataset_label[label_index]:
                                if label_index == 0:
                                    bio.append("B")
                                else:
                                    bio.append("I")
                                label_index += 1
                            else:
                                bio.append("O")
                                label_index = 0
                    while(len(bio) < len(tokenized_sentence)):
                        bio.append("O")
                    if len(tokenized_sentence) > MAX_LENGTH:
                        for idx in range(0, len(tokenized_sentence), MAX_LENGTH):
                            split = tokenized_sentence[idx:idx + MAX_LENGTH]
                            labels = bio[idx:idx+MAX_LENGTH]
                            if "B" in labels or "I" in labels:
                                data.append([" ".join(split) , " ".join(labels)])
                    else:
                        data.append([" ".join(tokenized_sentence)," ".join(bio)])
                    t = pd.DataFrame(data, columns=['string', 'label'])
                    # print(t)
                    df_t = df_t.append(t)
                    data = []

df_t.to_csv("../input/data_for_bert.csv", index = False)
