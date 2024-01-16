import torch
import os
import dill
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, AdamW, get_linear_schedule_with_warmup

from util import parse_cmg_args, set_seed, build_model_tokenizer_config, get_msg_label
from process_data.process_fira import load_fira_data
from process_data.process_jitcmg import load_jitfine_data
from process_data.process_mcmd import load_mcmd_data
from models.SingleModel import SingleModel
from metric import smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge


logger = logging.getLogger(__name__)


def train(args, train_examples, train_dataset, eval_examples, eval_dataset, mymodel, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0

    optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    best_bleu = 0
    best_ppl = 1e6
    patience = 0
    mymodel.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            diff_ids, diff_mask, msg_ids, msg_mask, manual_feature = [x.to(args.device) for x in batch]
            mymodel.train()
            loss = mymodel(diff_ids, diff_mask, msg_ids, msg_mask)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.warning(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            # truncate the gradient, used to prevent exploding gradient.
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, eval_examples, eval_dataset, mymodel, tokenizer)

                # checkpoint_prefix = f"epoch_{idx}_step_{step}"
                # output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)
                # model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                # output_file = os.path.join(output_dir, "model.bin")
                # save_content = {
                #     "model_state_dict": model_to_save.state_dict(),
                #     "optimizer_state_dict": optimizer.state_dict(),
                #     "scheduler": scheduler.state_dict()
                # }
                # torch.save(save_content, output_file)

                if results["eval_ppl"] < best_ppl:
                    patience = 0
                    best_ppl = results["eval_ppl"]
                    checkpoint_prefix = "checkpoint-best-f1"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    output_file = os.path.join(output_dir, "model.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    torch.save(save_content, output_file)

                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, eval_examples, eval_dataset, mymodel, tokenizer):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    eval_loss = 0
    batch_num = 0
    pred_ids = []
    msg_labels = eval_examples
    mymodel.eval()

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        diff_ids, diff_mask, msg_ids, msg_mask, manual_feature = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss = mymodel(diff_ids, diff_mask, msg_ids, msg_mask)
            if args.n_gpu > 1:
                loss = loss.mean()

            # if hasattr(mymodel, "module"):
            #     preds = mymodel.module.generate_preds(diff_ids, diff_mask)
            # else:
            #     preds = mymodel.generate_preds(diff_ids, diff_mask)

        eval_loss += loss.item()
        batch_num += 1
        # pred_ids.extend(preds)

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    # pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    #
    # eval_accs = []
    # predictions = []
    # labels = []
    #
    # for pred_nl, gold_nl in zip(pred_nls, eval_examples):
    #     eval_accs.append(pred_nl.strip() == gold_nl.target.strip())
    #     predictions.append(str(gold_nl.idx) + '\t' + pred_nl)
    #     labels.append(str(gold_nl.idx) + '\t' + gold_nl.target.strip() + '\n')
    #
    # (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, labels)
    # eval_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    #
    # result = {
    #     "eval_ppl": eval_ppl,
    #     "eval_bleu": eval_bleu,
    #     "em": np.mean(eval_accs),
    # }

    result = {
        "eval_ppl": eval_ppl
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_examples, test_dataset, mymodel, tokenizer):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    eval_loss = 0
    batch_num = 0
    pred_ids = []
    msg_labels = test_examples
    mymodel.eval()

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        diff_ids, diff_mask, msg_ids, msg_mask, manual_feature = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss = mymodel(diff_ids, diff_mask, msg_ids, msg_mask)
            if args.n_gpu > 1:
                loss = loss.mean()

            if hasattr(mymodel, "module"):
                preds = mymodel.module.generate_preds(diff_ids, diff_mask)
            else:
                preds = mymodel.generate_preds(diff_ids, diff_mask)

        eval_loss += loss.item()
        batch_num += 1
        pred_ids.extend(preds)

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    refs_dict = {}
    preds_dict = {}
    for i in range(len(pred_nls)):
        preds_dict[i] = [pred_nls[i]]
        refs_dict[i] = [test_examples[i].target.strip()]

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)

    eval_accs = []
    predictions = []
    labels = []

    for pred_nl, gold_nl in zip(pred_nls, test_examples):
        eval_accs.append(pred_nl.strip() == gold_nl.target.strip())
        predictions.append(str(gold_nl.idx) + '\t' + pred_nl)
        labels.append(str(gold_nl.idx) + '\t' + gold_nl.target.strip() + '\n')

    (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, labels)
    eval_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

    result = {
        "eval_ppl": eval_ppl,
        "eval_bleu": eval_bleu,
        "meteor": round(score_Meteor * 100, 2),
        "rouge-l": round(score_Rouge * 100, 2),
        "em": np.mean(eval_accs),
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])
    cpu_cont = multiprocessing.cpu_count()
    args.cpu_cont = cpu_cont
    pool = multiprocessing.Pool(args.cpu_cont)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    mymodel = SingleModel(model, config, tokenizer, args).to(device)
    print(mymodel)

    # store_path = "../datasets/jitfine"
    # with open(os.path.join(store_path, "train.pkl"), 'rb') as frb1:
    #     train_dataset = dill.load(frb1)
    # with open(os.path.join(store_path, "eval.pkl"), 'rb') as frb2:
    #     eval_dataset = dill.load(frb2)
    # with open(os.path.join(store_path, "test.pkl"), 'rb') as frb3:
    #     test_dataset = dill.load(frb3)

    if args.do_train:
        train_examples, train_dataset = load_mcmd_data(args, tokenizer, "train", pool)
        eval_examples, eval_dataset = load_mcmd_data(args, tokenizer, "eval", pool)

        train(args, train_examples, train_dataset, eval_examples, eval_dataset, mymodel, tokenizer)

    if args.do_test:

        test_examples, test_dataset = load_mcmd_data(args, tokenizer, "test", pool)

        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        checkpoint = torch.load(output_dir)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        test(args, test_examples, test_dataset, mymodel, tokenizer)


if __name__ == "__main__":
    args = parse_cmg_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
