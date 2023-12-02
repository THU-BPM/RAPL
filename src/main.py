import argparse
import os
import random
import string

import torch
import wandb
from apex import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data import parse_episodes, parse_episodes_from_index, collate_fn, parse_relations
from evaluation import get_f1, get_f1_macro
from model import Encoder
from utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="dataset/train_redocred.json", help="training file path")
    parser.add_argument("--dev_file", type=str, default="dataset/dev_redocred.json", help="dev file path")
    parser.add_argument("--indomain_test_file", type=str, default="dataset/test_redocred.json",
                        help="indomain test file path")
    parser.add_argument("--scierc_test_file", type=str, default="dataset/test_scierc.json",
                        help="scierc test file path")
    parser.add_argument("--indomain_test_indices_file", type=str, default="dataset/test_redocred_3_doc_indices.json",
                        help="indomain test indices file path")
    parser.add_argument("--scierc_test_indices_file", type=str, default="dataset/test_scierc_3_doc_indices.json",
                        help="scierc test indices file path")
    parser.add_argument("--cache_data", type=str, default="dataset/cache", help="cache dir of parsed data")
    parser.add_argument("--indomain_relation_info_file", type=str, default="meta/rel_info_with_descriptions.json",
                        help="indomain relation names and descriptions file path")
    parser.add_argument("--scierc_relation_info_file", type=str, default="meta/scierc_rel_info_with_descriptions.json",
                        help="scierc relation names and descriptions file path")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased",
                        help="pretrained model name or path")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="path to directory of checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--num_epochs", type=int, default=25, help="number of epochs to train")
    parser.add_argument("--support_docs_train", type=int, default=3, help="number of support documents during training")
    parser.add_argument("--support_docs_eval", type=int, default=3, help="number of support documents during eval")
    parser.add_argument("--query_docs_train", type=int, default=1, help="number of query documents during training")
    parser.add_argument("--query_docs_eval", type=int, default=1, help="number of query documents during eval")
    parser.add_argument("--samples_per_ep", type=int, default=2000, help="number of training episodes per epoch")
    parser.add_argument("--samples_data_train", type=int, default=50000, help="number of training episodes to generate")
    parser.add_argument("--samples_data_dev", type=int, default=500, help="number of dev episodes to generate")
    parser.add_argument("--balancing_train", type=str, default="single",
                        help="balancing (hard, soft, single) for training data")
    parser.add_argument("--balancing_dev", type=str, default="soft", help="balancing (hard, soft, single) for dev data")
    parser.add_argument("--train_batch_size", type=int, default=4, help="training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="eval batch size")
    parser.add_argument("--use_markers", action="store_true", help="use entity marker")
    parser.add_argument("--ensure_positive", action="store_true", help="ensure positive example query")
    parser.add_argument("--seed_model", type=int, default=42, help="random seed for model")
    parser.add_argument("--seed_data", type=int, default=42, help="random seed for data")
    parser.add_argument("--k_percentage", type=int, default=15, help="manipulate top k percentage attention")
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature for contrastive loss")
    parser.add_argument("--lamda", type=float, default=0.1, help="weight of contrastive loss")
    parser.add_argument("--train_num_classes", type=int, default=1, help="classes number upper bound for training")
    parser.add_argument("--dev_num_classes", type=int, default=2, help="classes number upper bound for dev")
    parser.add_argument("--test_indomain_num_classes", type=int, default=2,
                        help="classes number upper bound for indomain test")
    parser.add_argument("--test_scierc_num_classes", type=int, default=2,
                        help="classes number upper bound for scierc test")
    parser.add_argument("--normalize_prototype", action="store_true", help="normalize prototypes")
    parser.add_argument("--normalize_query", action="store_true", help="normalize query embeddings")
    parser.add_argument("--distance_metric", type=str, default="dot_product",
                        help="distance metric (dot_product or square_euclidean)")
    parser.add_argument("--base_nota_num", type=int, default=15, help="number of base NOTA vectors")
    parser.add_argument("--nota_rectification_factor", type=float, default=0.1,
                        help="base NOTA vectors rectification factor")
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="learning rate for encoder module")
    parser.add_argument("--attention_lr", type=float, default=1e-5, help="learning rate for attention modules")
    parser.add_argument("--mlp_lr", type=float, default=1e-5, help="learning rate for mlp modules")
    parser.add_argument("--nota_lr", type=float, default=1e-5, help="learning rate for nota embeddings")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="warmup epochs")
    parser.add_argument("--test_domain", type=str, default="indomain", help="test domain")
    parser.add_argument("--project", type=str, default="RAPL", help="project name for wandb")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
    print(random_string)

    wandb.init(project=args.project)
    wandb.config.update(args)
    wandb.config.identifier = random_string

    if not os.path.exists(args.cache_data):
        os.makedirs(args.cache_data)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    indomain_relation_tokens, indomain_relation_mask, indomain_relation_map = parse_relations(
        args.indomain_relation_info_file, tokenizer)
    if args.test_domain == "crossdomain":
        scierc_relation_tokens, scierc_relation_mask, scierc_relation_map = parse_relations(
            args.scierc_relation_info_file, tokenizer)

    if args.num_epochs != 0:
        training_episodes = parse_episodes(args.train_file,
                                           tokenizer,
                                           K=args.support_docs_train,
                                           n_queries=args.query_docs_train,
                                           n_samples=args.samples_data_train,
                                           markers=args.use_markers,
                                           balancing=args.balancing_train,
                                           seed=args.seed_data,
                                           ensure_positive=args.ensure_positive,
                                           cache=args.cache_data)
        dev_episodes = parse_episodes(args.dev_file,
                                      tokenizer,
                                      K=args.support_docs_eval,
                                      n_queries=args.query_docs_eval,
                                      n_samples=args.samples_data_dev,
                                      markers=args.use_markers,
                                      balancing=args.balancing_dev,
                                      seed=args.seed_data,
                                      ensure_positive=args.ensure_positive,
                                      cache=args.cache_data)
    if args.test_domain == "indomain":
        indomain_test_episodes = parse_episodes_from_index(args.indomain_test_file,
                                                           args.indomain_test_indices_file,
                                                           tokenizer,
                                                           markers=args.use_markers,
                                                           cache=args.cache_data)
    if args.test_domain == "crossdomain":
        scierc_test_episodes = parse_episodes_from_index(args.scierc_test_file,
                                                         args.scierc_test_indices_file,
                                                         tokenizer,
                                                         markers=args.use_markers,
                                                         cache=args.cache_data)

    g = torch.Generator()
    g.manual_seed(args.seed_data)
    set_seed(args)

    if args.num_epochs != 0:
        train_loader = DataLoader(training_episodes,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  generator=g)
        dev_loader = DataLoader(dev_episodes,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                collate_fn=collate_fn,
                                drop_last=False)
    else:
        train_loader = []
        dev_loader = []
    if args.test_domain == "indomain":
        indomain_test_loader = DataLoader(indomain_test_episodes,
                                          batch_size=args.eval_batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          drop_last=False)
    if args.test_domain == "crossdomain":
        scierc_test_loader = DataLoader(scierc_test_episodes,
                                        batch_size=args.eval_batch_size,
                                        shuffle=False,
                                        collate_fn=collate_fn,
                                        drop_last=False)

    lm_config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10)
    lm_model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=False, config=lm_config)
    relation_model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=False, config=lm_config)

    encoder = Encoder(
        config=lm_config,
        model=lm_model,
        relation_model=relation_model,
        cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token),
        sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
        markers=args.use_markers,
        k_percentage=args.k_percentage,
        temperature=args.temperature,
        lamda=args.lamda,
        normalize_prototype=args.normalize_prototype,
        normalize_query=args.normalize_query,
        distance_metric=args.distance_metric,
        base_nota_num=args.base_nota_num,
        nota_rectification_factor=args.nota_rectification_factor,
        device=args.device
    )

    encoder.to(args.device)

    if args.load_checkpoint is not None:
        print(f'loading model from {args.load_checkpoint}')
        encoder.load_state_dict(torch.load(f"{args.load_checkpoint}"))

    mlp_layer = ["head_extractor", "tail_extractor"]
    attention_layer = ["relation_to_context_attention"]
    nota_layer = ["nota_embeddings"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder.named_parameters() if
                    not any(nd in n for nd in mlp_layer + attention_layer + nota_layer)]},
        {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in mlp_layer)], "lr": args.mlp_lr},
        {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in attention_layer)],
         "lr": args.attention_lr},
        {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in nota_layer)], "lr": args.nota_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.encoder_lr, eps=1e-6)
    encoder, optimizer = amp.initialize(encoder, optimizer, opt_level="O1", verbosity=0)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(
        args.warmup_epochs * args.samples_per_ep / args.train_batch_size),
                                                   int(args.samples_per_ep / args.train_batch_size * args.num_epochs))
    step_global = -1
    train_iter = iter(train_loader)
    best_f1 = 0.0

    for i in tqdm(range(args.num_epochs)):
        true_positives, false_positives, false_negatives = {}, {}, {}
        encoder.train()
        loss_agg = 0
        count = 0
        with tqdm(range(int(args.samples_per_ep / args.train_batch_size))) as pbar:
            for _ in pbar:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                step_global += 1
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                output, loss = encoder(exemplar_tokens.to('cuda'),
                                       exemplar_mask.to('cuda'),
                                       exemplar_positions,
                                       exemplar_labels,
                                       query_tokens.to('cuda'),
                                       query_mask.to('cuda'),
                                       query_positions,
                                       query_labels,
                                       label_types,
                                       indomain_relation_tokens.to('cuda'),
                                       indomain_relation_mask.to('cuda'),
                                       indomain_relation_map,
                                       args.train_num_classes)

                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        lbs_align = []
                        for rel_type, rel_hts in lbs.items():
                            if rel_type != "NOTA":
                                for ht in rel_hts:
                                    lbs_align.append([ht[0], ht[1], rel_type])

                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs_align:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs_align:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1

                count += 1
                loss_agg += loss.item()
                pbar.set_postfix({"Loss": f"{loss_agg / count:.2f}"})

                wandb.log({"loss": loss.item()}, step=step_global)
                wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]}, step=step_global)

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

                optimizer.step()
                lr_scheduler.step()
                encoder.zero_grad()
                del loss, output

        p, r, f = get_f1(true_positives, false_positives, false_negatives)
        p_train, r_train, f1_train = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
        wandb.log({"precision_train": p_train}, step=step_global)
        wandb.log({"recall_train": r_train}, step=step_global)
        wandb.log({"f1_macro_train": f1_train}, step=step_global)
        wandb.log({"f1_micro_train": f}, step=step_global)

        true_positives, false_positives, false_negatives = {}, {}, {}
        encoder.eval()
        with tqdm(dev_loader) as pbar:
            for batch in pbar:
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                output = encoder(exemplar_tokens.to('cuda'),
                                 exemplar_mask.to('cuda'),
                                 exemplar_positions,
                                 exemplar_labels,
                                 query_tokens.to('cuda'),
                                 query_mask.to('cuda'),
                                 query_positions,
                                 None,
                                 label_types,
                                 indomain_relation_tokens.to('cuda'),
                                 indomain_relation_mask.to('cuda'),
                                 indomain_relation_map,
                                 args.dev_num_classes)

                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        lbs_align = []
                        for rel_type, rel_hts in lbs.items():
                            if rel_type != "NOTA":
                                for ht in rel_hts:
                                    lbs_align.append([ht[0], ht[1], rel_type])

                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs_align:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs_align:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1

        p, r, f = get_f1(true_positives, false_positives, false_negatives)
        p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

        if f1_dev >= best_f1:
            wandb.log({"best_precision_dev": p_dev}, step=step_global)
            wandb.log({"best_recall_dev": r_dev}, step=step_global)
            wandb.log({"best_f1_macro_dev": f1_dev}, step=step_global)
            wandb.log({"Best_f1_micro_dev": f}, step=step_global)
            best_f1 = f1_dev
            torch.save(encoder.state_dict(), os.path.join(args.checkpoints_dir, f"{args.project}_{random_string}.pt"))

        wandb.log({"precision_dev": p_dev}, step=step_global)
        wandb.log({"recall_dev": r_dev}, step=step_global)
        wandb.log({"f1_macro_dev": f1_dev}, step=step_global)
        wandb.log({"f1_micro_dev": f}, step=step_global)

    if args.num_epochs > 0:
        encoder.to("cpu")
        del encoder
        torch.cuda.empty_cache()

        lm_config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10)
        lm_model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=False, config=lm_config)
        relation_model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=False, config=lm_config)

        encoder = Encoder(
            config=lm_config,
            model=lm_model,
            relation_model=relation_model,
            cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token),
            sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
            markers=args.use_markers,
            k_percentage=args.k_percentage,
            temperature=args.temperature,
            lamda=args.lamda,
            normalize_prototype=args.normalize_prototype,
            normalize_query=args.normalize_query,
            distance_metric=args.distance_metric,
            base_nota_num=args.base_nota_num,
            nota_rectification_factor=args.nota_rectification_factor,
            device=args.device
        )

        encoder.to(args.device)

        encoder.load_state_dict(torch.load(os.path.join(args.checkpoints_dir, f"{args.project}_{random_string}.pt")))
    else:
        step_global = 0
    encoder.eval()

    if args.test_domain == "indomain":
        print("---- INDOMAIN TEST EVAL -----")
        true_positives, false_positives, false_negatives = {}, {}, {}
        with tqdm(indomain_test_loader) as pbar:
            for batch in pbar:
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                output = encoder(exemplar_tokens.to('cuda'),
                                 exemplar_mask.to('cuda'),
                                 exemplar_positions,
                                 exemplar_labels,
                                 query_tokens.to('cuda'),
                                 query_mask.to('cuda'),
                                 query_positions,
                                 None,
                                 label_types,
                                 indomain_relation_tokens.to('cuda'),
                                 indomain_relation_mask.to('cuda'),
                                 indomain_relation_map,
                                 args.test_indomain_num_classes)

                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        lbs_align = []
                        for rel_type, rel_hts in lbs.items():
                            if rel_type != "NOTA":
                                for ht in rel_hts:
                                    lbs_align.append([ht[0], ht[1], rel_type])

                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs_align:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs_align:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1

        p, r, f = get_f1(true_positives, false_positives, false_negatives)
        p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

        wandb.log({"precision_test_indomain": p_dev}, step=step_global)
        wandb.log({"recall_test_indomain": r_dev}, step=step_global)
        wandb.log({"f1_macro_test_indomain": f1_dev}, step=step_global)
        wandb.log({"f1_micro_test_indomain": f}, step=step_global)

    if args.test_domain == "crossdomain":
        print("---- SCIERC TEST EVAL -----")
        true_positives, false_positives, false_negatives = {}, {}, {}
        with tqdm(scierc_test_loader) as pbar:
            for batch in pbar:
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                output = encoder(exemplar_tokens.to('cuda'),
                                 exemplar_mask.to('cuda'),
                                 exemplar_positions,
                                 exemplar_labels,
                                 query_tokens.to('cuda'),
                                 query_mask.to('cuda'),
                                 query_positions,
                                 None,
                                 label_types,
                                 scierc_relation_tokens.to('cuda'),
                                 scierc_relation_mask.to('cuda'),
                                 scierc_relation_map,
                                 args.test_scierc_num_classes)

                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        lbs_align = []
                        for rel_type, rel_hts in lbs.items():
                            if rel_type != "NOTA":
                                for ht in rel_hts:
                                    lbs_align.append([ht[0], ht[1], rel_type])

                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs_align:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs_align:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1

        p, r, f = get_f1(true_positives, false_positives, false_negatives)
        p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)

        wandb.log({"precision_test_scierc": p_dev}, step=step_global)
        wandb.log({"recall_test_scierc": r_dev}, step=step_global)
        wandb.log({"f1_macro_test_scierc": f1_dev}, step=step_global)
        wandb.log({"f1_micro_test_scierc": f}, step=step_global)
