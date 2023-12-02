import json
import os
import random
from pathlib import Path

import torch
import transformers
from tqdm import tqdm


def intersection(lst1, lst2):
    # https://www.geeksforgeeks.org/python-intersection-two-lists/
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def tokenize_and_reindex(sentences, entity_markers, tokenizer, em_tokens=("*", "*")):
    sentence_lengths = [len(s) for s in sentences]
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(sent)

    if type(tokenizer) == transformers.BertTokenizer or type(tokenizer) == transformers.BertTokenizerFast:
        l_offset = 1
        all_tokens = ["[CLS]"] + all_tokens + ["[SEP]"]
    else:
        l_offset = 1
        all_tokens = ["<s>"] + all_tokens + ["</s>"]

    start_tokens = [[] for _ in all_tokens]
    end_tokens = [[] for _ in all_tokens]

    sentence_offsets = [0]
    for s in sentence_lengths[:-1]:
        sentence_offsets.append(s + sentence_offsets[-1])

    m = 0
    mention_map = []
    for i, mentions in enumerate(entity_markers):
        for ment in mentions:
            start_tokens[sentence_offsets[ment['sent_id']] + ment['pos'][0] + l_offset].append(m)
            end_tokens[sentence_offsets[ment['sent_id']] + ment['pos'][1] + l_offset].append(m)
            mention_map.append(i)
            m += 1

    tokenized_text = []
    open_spans = [[-1, -1] for _ in range(m)]

    for tk, span_starts, span_ends in zip(all_tokens, start_tokens, end_tokens):
        for sp_e in span_ends:
            if em_tokens is not None:
                tokenized_text.extend(tokenizer.tokenize(em_tokens[1]))
            open_spans[sp_e][1] = len(tokenized_text)

        for sp_s in span_starts:
            open_spans[sp_s][0] = len(tokenized_text)
            if em_tokens is not None:
                tokenized_text.extend(tokenizer.tokenize(em_tokens[0]))

        tokenized_text.extend(tokenizer.tokenize(tk))

    for e in open_spans:
        if -1 in e:
            print("warning!")

    entity_positions = [[] for _ in entity_markers]
    for i, span in enumerate(open_spans):
        entity_positions[mention_map[i]].append(span)

    return tokenizer.convert_tokens_to_ids(tokenized_text), entity_positions


def parse_document(document, tokenizer, markers=True, no_processing=False):
    if no_processing:
        return {
            "labels": document['labels'],
            "sents": document['sents'],
            "vertexSet": document['vertexSet'],
        }
    else:
        # parse doc
        em = None
        if markers:
            em = ("*", "*")
        tokens, entity_positions = tokenize_and_reindex(document['sents'], document['vertexSet'], tokenizer, em)

        return {
            "tokens": tokens,
            "entity_positions": entity_positions,
            "labels": document['labels'],
        }


def select_labels(document, relation_types, no_processing=False):
    if no_processing:
        labeled_relations = []
        for rel in document['labels']:
            if rel['r'] in relation_types:
                labeled_relations.append(rel)
        return {
            "labels": labeled_relations,
            "sents": document['sents'],
            "vertexSet": document['vertexSet'],
        }
    else:
        triples = {}
        for rel in document['labels']:
            if rel['r'] in relation_types:
                if (rel['h'], rel['t']) not in triples:
                    triples[(rel['h'], rel['t'])] = [rel['r']]
                else:
                    triples[(rel['h'], rel['t'])].append(rel['r'])

        labeled_relations = {"NOTA": []}
        for relation_type in relation_types:
            labeled_relations[relation_type] = []

        entity_num = len(document["entity_positions"])
        for h in range(entity_num):
            for t in range(entity_num):
                if h != t:
                    if (h, t) in triples:
                        for r in triples[(h, t)]:
                            labeled_relations[r].append([h, t])
                    else:
                        labeled_relations["NOTA"].append([h, t])

        return {
            "tokens": document["tokens"],
            "entity_positions": document["entity_positions"],
            "labels": labeled_relations,
        }


def parse_episodes(input_file, tokenizer, K=1, n_queries=3, n_samples=100, markers=True, balancing="soft", seed=123,
                   ensure_positive=True, cache=None, eval_single=False, no_processing=False):
    if balancing not in ["hard", "soft", "single"]:
        raise

    # Set random seed
    if seed is not None:
        random.seed(seed)

    input_filename = input_file
    input_file = json.load(open(input_file, "r", encoding="utf-8"))

    # ----- Indexing -----
    # Build a dictionary containing the indices of all documents containing an instance of each relation type

    documents_for_type = {}
    types_for_document = []

    document_blacklist = []

    for i, document in tqdm(enumerate(input_file), desc="Indexing"):

        relation_types_in_document = []

        if len(document['vertexSet']) < 2:
            document_blacklist.append(i)

        for rel in document['labels']:

            if rel['r'] not in relation_types_in_document:
                # mark as added
                relation_types_in_document.append(rel['r'])
            else:
                # already added document to dictionary -> skip
                continue

            if rel['r'] not in documents_for_type.keys():
                # add relation type to keys
                documents_for_type[rel['r']] = []

            # add entry for document
            documents_for_type[rel['r']].append(i)
        types_for_document.append(relation_types_in_document)

    # Filter out types where K>=available documents
    i = 0
    for rtype in [key for key in documents_for_type.keys()]:
        if len(documents_for_type[rtype]) <= K:
            for doc in documents_for_type[rtype]:
                types_for_document[doc].remove(rtype)
            documents_for_type.pop(rtype)
        else:
            i += 1

    # ----- Support sampling -----
    # Choose n_samples support samples and save the type of interest

    episodes = []

    # Lists to keep track of how often we've added each relation type to the support corpus
    relation_types = [x for x in documents_for_type.keys()]
    rep_count = [0 for _ in relation_types]

    for i in tqdm(range(n_samples), desc="Choosing support examples"):
        # find least represented types
        min_rep = min(rep_count)
        all_min = [rtype for count, rtype in zip(rep_count, relation_types) if count == min_rep]

        # randomly pick one
        r_s = random.choice(all_min)

        # pick a support document
        support_doc_index = random.sample(documents_for_type[r_s], k=K)

        # index all types in support documents
        relation_types_in_sup = []
        for doc_id in support_doc_index:
            relation_types_in_sup.extend(types_for_document[doc_id])
        relation_types_in_sup = list(set(relation_types_in_sup))

        # apply hard balancing
        if balancing == "hard":
            selected_relation_types = []
            for rel_type in relation_types_in_sup:
                if rep_count[relation_types.index(rel_type)] == min_rep:
                    selected_relation_types.append(rel_type)
            relation_types_in_sup = selected_relation_types

        # single relation type only for 3-shot setting
        if balancing == "single":
            relation_types_in_sup = [r_s]

        # add to samples
        episodes.append({
            "support": support_doc_index,
            "labeled_relations": relation_types_in_sup,
            "r_s": r_s,
        })

        for rel_type in relation_types_in_sup:
            rep_count[relation_types.index(rel_type)] += 1

    # ----- Query sampling -----
    # choose n_queries for each episode, ensuring that the support document is not one of them

    for i in tqdm(range(n_samples), desc="Choosing query examples"):
        episode = episodes[i]
        query_documents = random.sample(
            [x for x in range(len(types_for_document)) if x not in episode["support"] and x not in document_blacklist],
            k=n_queries)

        # ensure we have a positive examples for eval sets
        if ensure_positive:
            if len(intersection(query_documents, documents_for_type[episode["r_s"]])) == 0:
                query_documents[0] = random.choice(
                    [x for x in documents_for_type[episode["r_s"]] if x not in episode["support"]])

        episodes[i]["query"] = query_documents

    # ----- Document Parsing & Caching -----

    if cache is not None:
        if not os.path.isdir(cache):
            print("no cache directory found. making folder.")
            os.makedirs(cache)

        filename = f"cached-{Path(input_filename).stem}.json"
        filepath = os.path.join(cache, filename)

        if os.path.exists(filepath):
            try:
                print("found cached dataset. loading from", filepath)
                parsed_docs = json.load(open(filepath, "r", encoding="utf-8"))
            except:
                print("ran into problems loading file. rebuilding dataset.")
                parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                               tqdm(input_file, desc="Parsing documents")]
                json.dump(parsed_docs, open(filepath, "w", encoding="utf-8"))
        else:
            parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                           tqdm(input_file, desc="Parsing documents")]
            json.dump(parsed_docs, open(filepath, "w", encoding="utf-8"))
    else:
        parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                       tqdm(input_file, desc="Parsing documents")]

    # ----- Episode Construction -----

    output = []
    for i in tqdm(range(n_samples), desc="Constructing episodes"):
        if eval_single:
            for rel_type in episodes[i]["labeled_relations"]:
                episode = episodes[i].copy()

                # pselect labels
                episode["exemplars"] = [select_labels(parsed_docs[doc_id], [rel_type], no_processing=no_processing) for
                                        doc_id in episode["support"]]
                episode["test_examples"] = [select_labels(parsed_docs[doc_id], [rel_type], no_processing=no_processing)
                                            for doc_id in episode["query"]]

                episode.pop("support")
                episode.pop("labeled_relations")
                episode.pop("query")
                episode.pop("r_s")

                output.append(episode)
        else:
            episode = episodes[i]

            # pselect labels
            episode["exemplars"] = [
                select_labels(parsed_docs[doc_id], episode["labeled_relations"], no_processing=no_processing) for doc_id
                in episode["support"]]
            episode["test_examples"] = [
                select_labels(parsed_docs[doc_id], episode["labeled_relations"], no_processing=no_processing) for doc_id
                in episode["query"]]

            episode.pop("support")
            episode.pop("labeled_relations")
            episode.pop("query")
            episode.pop("r_s")

            output.append(episode)

    return output


def parse_episodes_from_index(input_file, index_file, tokenizer, markers=True, cache=None, eval_single=False,
                              no_processing=False):
    episodes = json.load(open(index_file, "r", encoding="utf-8"))

    input_filename = input_file
    input_file = json.load(open(input_file, "r", encoding="utf-8"))

    # ----- Document Parsing & Caching -----

    if cache is not None:
        if not os.path.isdir(cache):
            print("no cache directory found. making folder.")
            os.makedirs(cache)

        filename = f"cached-{Path(input_filename).stem}.json"
        filepath = os.path.join(cache, filename)

        if os.path.exists(filepath):
            try:
                print("found cached dataset. loading from", filepath)
                parsed_docs = json.load(open(filepath, "r", encoding="utf-8"))
            except:
                print("ran into problems loading file. rebuilding dataset.")
                parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                               tqdm(input_file, desc="Parsing documents")]
                json.dump(parsed_docs, open(filepath, "w", encoding="utf-8"))
        else:
            parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                           tqdm(input_file, desc="Parsing documents")]
            json.dump(parsed_docs, open(filepath, "w", encoding="utf-8"))
    else:
        parsed_docs = [parse_document(doc, tokenizer, markers=markers, no_processing=no_processing) for doc in
                       tqdm(input_file, desc="Parsing documents")]

    # ----- Episode Construction -----

    output = []
    for i in tqdm(range(len(episodes)), desc="Constructing episodes"):
        if eval_single:
            for rel_type in episodes[i]["labeled_relations"]:
                episode = episodes[i].copy()

                # pselect labels
                episode["exemplars"] = [select_labels(parsed_docs[doc_id], [rel_type], no_processing=no_processing) for
                                        doc_id in episode["support"]]
                episode["test_examples"] = [select_labels(parsed_docs[doc_id], [rel_type], no_processing=no_processing)
                                            for doc_id in episode["query"]]

                episode.pop("support")
                episode.pop("labeled_relations")
                episode.pop("query")
                episode.pop("r_s")

                output.append(episode)
        else:
            episode = episodes[i]

            # pselect labels
            episode["exemplars"] = [
                select_labels(parsed_docs[doc_id], episode["labeled_relations"], no_processing=no_processing) for doc_id
                in episode["support"]]
            episode["test_examples"] = [
                select_labels(parsed_docs[doc_id], episode["labeled_relations"], no_processing=no_processing) for doc_id
                in episode["query"]]

            episode.pop("support")
            episode.pop("labeled_relations")
            episode.pop("query")
            episode.pop("r_s")

            output.append(episode)

    return output


def collate_fn(batch):
    n_tokens = [len(e["tokens"]) for f in batch for e in f["exemplars"]] + [len(e["tokens"]) for f in batch for e in
                                                                            f["test_examples"]]
    # get max dimensions for padding
    max_len = max(n_tokens)

    # exemplars
    # pad token_ids and create a mask to indicate padding
    exemplar_tokens = []
    exemplar_mask = []
    for x in [f["exemplars"] for f in batch]:
        exemplar_tokens.append([e["tokens"] + [0] * (max_len - len(e["tokens"])) for e in x])
        exemplar_mask.append([[1.0] * len(e["tokens"]) + [0.0] * (max_len - len(e["tokens"])) for e in x])
    exemplar_tokens = torch.tensor(exemplar_tokens, dtype=torch.long)  # [b, s_doc_num, max_len]
    exemplar_mask = torch.tensor(exemplar_mask, dtype=torch.float)  # [b, s_doc_num, max_len]

    # queries
    # pad token_ids and create a mask to indicate padding
    query_tokens = []
    query_mask = []
    for x in [f["test_examples"] for f in batch]:
        query_tokens.append([e["tokens"] + [0] * (max_len - len(e["tokens"])) for e in x])
        query_mask.append([[1.0] * len(e["tokens"]) + [0.0] * (max_len - len(e["tokens"])) for e in x])
    query_tokens = torch.tensor(query_tokens, dtype=torch.long)  # [b, q_doc_num, max_len]
    query_mask = torch.tensor(query_mask, dtype=torch.float)  # [b, q_doc_num, max_len]

    # labels
    exemplar_labels = []  # [b, s_doc_num] dict
    for x in [f["exemplars"] for f in batch]:
        exemplar_labels.append([e["labels"] for e in x])
    query_labels = []  # [b, q_doc_num] dict
    for x in [f["test_examples"] for f in batch]:
        query_labels.append([e["labels"] for e in x])

    # entity positions
    exemplar_positions = []  # [b, s_doc_num, e, m] (start, end)
    for x in [f["exemplars"] for f in batch]:
        exemplar_positions.append([e["entity_positions"] for e in x])
    query_positions = []  # [b, q_doc_num, e, m] (start, end)
    for x in [f["test_examples"] for f in batch]:
        query_positions.append([e["entity_positions"] for e in x])

    label_types = [["NOTA"] for _ in batch]  # [b, r_types_in_an_episode]
    for i, x in enumerate([f["exemplars"] for f in batch]):
        # rts = []
        # for e in x:
        #     rts.extend(list(e["labels"].keys()))
        rts = [rel_type for rel_type in list(x[0]["labels"].keys()) if rel_type != "NOTA"]
        label_types[i].extend(rts)

    return exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types


def parse_relations(relation_info_file, tokenizer):
    with open(relation_info_file, "r", encoding="utf-8") as f:
        relation_info = json.load(f)

    relation_list = []
    relation_map = {}
    for i, (key, value) in enumerate(relation_info.items()):
        relation_map[key] = i
        relation = value['name'] + ": " + value['description']
        relation_list.append(relation)

    encoded_relations = tokenizer(relation_list, padding=True, return_tensors="pt")
    return encoded_relations['input_ids'], encoded_relations['attention_mask'], relation_map
