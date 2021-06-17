import os

import pandas as pd
import torch
import time

import mlflow

# 自作モジュール
from models import get_model, train_model, val_model, predict_labels
from utils import train_data_pairs, splits_sentence, get_time, get_cfg, get_device, get_idx2tag, get_tag2idx, get_tokenizer
from data import prepare_dataloader
from predict import predict


def main():

    CFG = get_cfg()
    mlflow.start_run()
    data_name = CFG['csv_name']

    if 'COLAB_GPU' in set(os.environ.keys()):
        BIO_LABEL = "/content/" + data_name
    elif "KAGGLE_URL_BASE" in set(os.environ.keys()):
        BIO_LABEL = "/kaggle/input/train-dataset-coleridge/" + data_name
    else:
        BIO_LABEL = "../input/" + dvata_name

    df = pd.read_csv(BIO_LABEL)

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tag_to_idx = get_tag2idx()

    mlflow.set_tracking_uri("./mlruns")

    mlflow.log_dict(CFG, "config.json")

    mlflow.log_params(CFG)

    tokenizer = get_tokenizer()
    train_data = train_data_pairs(df)

    tokenized_sentences, labels = splits_sentence(train_data, tokenizer)

    model, optimizer, scheduler = get_model(tag_to_idx, device, CFG)

    train_dataloader, test_dataloader = prepare_dataloader(
        tokenized_sentences,
        labels,
        tokenizer,
        tag_to_idx,
        batch_size=CFG['batch_size'],
        debug=CFG['debug'])

    d = df.iloc[5]
    sample_sentence = d['string']
    ans_label = d['label']
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    epochs = CFG["epoch"]

    for epoch in range(epochs):
        start = time.time()

        model, train_loss = train_model(model,
                                        optimizer,
                                        train_dataloader,
                                        device,
                                        scheduler=scheduler)
        val_loss = val_model(model, test_dataloader, device)
        end = time.time()
        elapsed_time = get_time(start, end)
        print(f"epoch:{epoch+1}")
        print("time: {} train_loss:{}    val_loss:{}".format(
            elapsed_time, train_loss, val_loss))
        predict_labels(model, sample_sentence, ans_label, idx_to_tag,
                       tokenizer, device)
        mlflow.log_metric("train loss", train_loss, epoch + 1)
        mlflow.log_metric("validation loss", val_loss, epoch + 1)

        # mlflow.log_artifact(model, '/model')

    mlflow.end_run()
    if CFG['bert_type'] == 0:
        torch.save(model.state_dict(),
                   './model/model_{}_layers.pth'.format(CFG['hidden_layers']))
    else:
        torch.save(model.state_dict(), './model/pretrained_model.pth')
    """
    model.load_state_dict(torch.load('./model.pth', map_location=device))

    path = '../input/test/'
    predict(model, path, tokenizer, device, idx_to_tag)
    """


if __name__ == '__main__':
    main()
