import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils import set_seed, collate_fn
from prepro import TACREDProcessor
from evaluation import get_f1
from model import REModel
from torch.cuda.amp import GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'epoch:{epoch}', leave=True, position=0)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      'entity_mask': batch[5].to(args.device),
                      'entity_label': batch[6].to(args.device),
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    f1, precision, recall, accuracy = evaluate(args, model, features, tag=tag)
                    print(
                        f'f1:{f1 * 100},  precision:{precision * 100},  recall:{recall * 100}, accuracy:{accuracy * 100}')


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for _, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  'entity_mask': batch[5].to(args.device),
                  'entity_label': batch[6].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()


    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)

    accuracy = accuracy_score(keys, preds)
    preci_score, recal_score, max_f1 = get_f1(keys, preds)
    return max_f1, preci_score, recal_score, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="../pretrain_model/roberta-base", type=str)
    parser.add_argument("--input_format", default="entity_marker_punct", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=7, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=70,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="tacred")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--k_size", type=int, default=2)
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )


    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )


    config.num_class = args.num_class
    config.dropout_prob = args.dropout_prob
    config.k_size = args.k_size

    model = REModel.from_pretrained(args.model_name_or_path, config=config)

    model.to(device)

    train_file = os.path.join(args.data_dir, "train.txt")
    test_file = os.path.join(args.data_dir, "test.txt")

    if not args.debug:
        processor = TACREDProcessor(args, tokenizer)
        train_features = processor.read(train_file)
        test_features = processor.read(test_file)


    benchmarks = (
        ("test", test_features),
    )

    # if train
    train(args, model, train_features, benchmarks)

    # if test
    # model.load_state_dict(torch.load('Model_path'))
    # model.eval()
    # for tag, features in benchmarks:
    #     f1, precision, recall, accuracy = evaluate(args, model, features, tag=tag)
    #     print(
    #         f'f1:{f1 * 100},  precision:{precision * 100},  recall:{recall * 100}, accuracy:{accuracy * 100}')


if __name__ == "__main__":
    main()










