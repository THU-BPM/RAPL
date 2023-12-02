import torch
from opt_einsum import contract
from torch import nn
import torch.nn.functional as F

from losses import ClsLoss, RelWeightConLoss
from utils import process_long_input


class Encoder(nn.Module):
    def __init__(self,
                 config,
                 model,
                 relation_model,
                 cls_token_id=0,
                 sep_token_id=0,
                 markers=True,
                 k_percentage=10,
                 temperature=0.4,
                 lamda=0.1,
                 normalize_prototype=False,
                 normalize_query=False,
                 distance_metric="dot_product",
                 base_nota_num=20,
                 nota_rectification_factor=0.1,
                 device="cuda"):
        super().__init__()
        self.config = config
        self.model = model
        self.relation_model = relation_model
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.markers = markers
        self.k_percentage = k_percentage
        self.lamda = lamda
        self.normalize_prototype = normalize_prototype
        self.normalize_query = normalize_query
        self.distance_metric = distance_metric
        self.base_nota_num = base_nota_num
        self.nota_rectification_factor = nota_rectification_factor
        self.device = device

        self.hidden_size = config.hidden_size
        self.head_num = config.num_attention_heads

        self.head_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.relation_to_context_attention = nn.MultiheadAttention(config.hidden_size, 1, batch_first=True)

        self.nota_embeddings = nn.Parameter(torch.zeros(base_nota_num, 2 * config.hidden_size))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)
        self.first_run = True

        self.loss_fnt = ClsLoss()
        self.loss_sup = RelWeightConLoss(temperature=temperature)

    def encode(self, input_ids, attention_mask):
        # Source: https://github.com/wzhouad/ATLOP
        start_tokens = [self.cls_token_id]
        end_tokens = [self.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def forward(self,
                exemplar_input_ids=None,
                exemplar_masks=None,
                exemplar_entity_positions=None,
                exemplar_labels=None,
                query_input_ids=None,
                query_masks=None,
                query_entity_positions=None,
                query_labels=None,
                type_labels=None,
                relation_ids=None,
                relation_masks=None,
                relation_map=None,
                num_classes=1):
        # -------------- build relation representations from exemplars ------------

        prototypes = []
        batch_relation_embeddings = {}
        batch_size, num_exemplars, max_len = exemplar_input_ids.size()
        support_doc_length = exemplar_masks.sum(-1)  # [b, s_doc_num]

        sequence_output, attention = self.encode(exemplar_input_ids.view(-1, max_len), exemplar_masks.view(-1, max_len))
        sequence_output = sequence_output.view(batch_size, num_exemplars, max_len,
                                               -1)  # [b, s_doc_num, max_len, hidden]
        attention = attention.view(batch_size, num_exemplars, -1, max_len,
                                   max_len)  # [b, s_doc_num, head, max_len, max_len]

        relation_output = self.relation_model(input_ids=relation_ids, attention_mask=relation_masks)[
            0]  # [rel_num, rel_max_len, hidden]
        relation_cls = relation_output[:, 0, :]  # [rel_num, hidden]

        for batch_i in range(batch_size):
            episode_prototypes = [None for _ in type_labels[batch_i]]
            episode_relation_embeddings = {}
            for rel_type in type_labels[batch_i]:
                episode_relation_embeddings[rel_type] = []

            for support_i in range(num_exemplars):
                # entity_embs, entity_atts, masks = [], [], []
                # for entity in exemplar_entity_positions[batch_i][support_i]:
                #     e_emb, mask = [], []
                #     if len(entity) > 1:
                #         e_att = []
                #         for start, end in entity:
                #             if start < max_len:
                #                 e_emb.append(sequence_output[batch_i, support_i, start])
                #                 mask.append(1)
                #                 e_att.append(attention[batch_i, support_i, :, start])
                #         if len(e_att) > 0:
                #             e_att = torch.stack(e_att, dim=0).mean(0)
                #         else:
                #             e_att = torch.zeros(self.head_num, max_len).to(self.device)
                #     else:
                #         start, end = entity[0]
                #         if start < max_len:
                #             e_emb.append(sequence_output[batch_i, support_i, start])
                #             mask.append(1)
                #             e_att = attention[batch_i, support_i, :, start]
                #         else:
                #             e_att = torch.zeros(self.head_num, max_len).to(self.device)
                #     entity_embs.append(e_emb)
                #     masks.append(mask)
                #     entity_atts.append(e_att)
                #
                # max_mentions = max([len(e_emb) for e_emb in entity_embs])
                # for i, (e_emb, mask) in enumerate(zip(entity_embs, masks)):
                #     if len(e_emb) < max_mentions:
                #         for _ in range(max_mentions - len(e_emb)):
                #             e_emb.append(torch.zeros(self.hidden_size).to(self.device))
                #             mask.append(0)
                #     entity_embs[i] = torch.stack(e_emb, dim=0)
                #     masks[i] = torch.FloatTensor(mask).to(self.device)
                # entity_embs = torch.stack(entity_embs, dim=0)  # [e, max_mention, hidden]
                # masks = torch.stack(masks, dim=0)  # [e, max_mention]
                #
                # # newly added code
                # entity_embs = entity_embs.sum(1) / (masks.sum(1, keepdim=True) + 1e-10)  # [e, hidden]
                #
                # entity_atts = torch.stack(entity_atts, dim=0)  # [e, head, max_len]
                # n_e = entity_embs.size(0)
                entity_embs, entity_atts = [], []
                for entity in exemplar_entity_positions[batch_i][support_i]:
                    if len(entity) > 1:
                        e_emb, e_att = [], []
                        for start, end in entity:
                            if start < max_len:
                                e_emb.append(sequence_output[batch_i, support_i, start])
                                e_att.append(attention[batch_i, support_i, :, start])
                        if len(e_emb) > 0:
                            e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                            e_att = torch.stack(e_att, dim=0).mean(0)
                        else:
                            e_emb = torch.zeros(self.hidden_size).to(self.device)
                            e_att = torch.zeros(self.head_num, max_len).to(self.device)
                    else:
                        start, end = entity[0]
                        if start < max_len:
                            e_emb = sequence_output[batch_i, support_i, start]
                            e_att = attention[batch_i, support_i, :, start]
                        else:
                            e_emb = torch.zeros(self.hidden_size).to(self.device)
                            e_att = torch.zeros(self.head_num, max_len).to(self.device)
                    entity_embs.append(e_emb)
                    entity_atts.append(e_att)

                entity_embs = torch.stack(entity_embs, dim=0)  # [e, hidden]
                entity_atts = torch.stack(entity_atts, dim=0)  # [e, head, max_len]
                n_e = entity_embs.size(0)

                doc_length = support_doc_length[batch_i][support_i].item()
                for rel_type, hts in exemplar_labels[batch_i][support_i].items():
                    if rel_type != "NOTA" and len(hts) > 0:
                        ht_i = torch.LongTensor(hts).to(self.device)  # [pair_of_rel, 2]
                        hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # [pair_of_rel, hidden]
                        ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # [pair_of_rel, hidden]

                        h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])  # [pair_of_rel, head, max_len]
                        t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])  # [pair_of_rel, head, max_len]
                        ht_att = F.normalize((h_att * t_att).mean(1), p=1, dim=1)  # [pair_of_rel, max_len]
                        _, rel_att = self.relation_to_context_attention(
                            relation_cls[relation_map[rel_type]].expand(1, 1, -1),
                            sequence_output[batch_i, support_i].unsqueeze(0),
                            sequence_output[batch_i, support_i].unsqueeze(0), key_padding_mask=~(
                                exemplar_masks[batch_i, support_i].unsqueeze(0).bool()))  # [1, 1, max_len]
                        rel_att = torch.squeeze(rel_att, 1)  # [1, max_len]
                        att = ht_att * rel_att  # [pair_of_rel, max_len]
                        top_att_values = torch.topk(att, int(doc_length * self.k_percentage / 100), dim=-1)[
                            0]  # [pair_of_rel, doc_len*k%]
                        top_att_values = top_att_values[:, -1:]  # [pair_of_rel, 1]
                        top_att_mask = (att >= top_att_values)  # [pair_of_rel, max_len]
                        ht_att[top_att_mask] = ht_att[top_att_mask] + rel_att.expand(ht_att.shape[0], -1)[top_att_mask]
                        ht_att = F.normalize(ht_att, p=1, dim=1)  # [pair_of_rel, max_len]
                        rs = contract("ld,rl->rd", sequence_output[batch_i, support_i], ht_att)  # [pair_of_rel, hidden]

                        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # [pair_of_rel, hidden]
                        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # [pair_of_rel, hidden]
                        episode_relation_embeddings[rel_type].append(torch.cat([hs, ts], dim=1))

                hts = exemplar_labels[batch_i][support_i]["NOTA"]
                ht_i = torch.LongTensor(hts).to(self.device)  # [pair_of_nota, 2]

                hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # [pair_of_nota, hidden]
                ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # [pair_of_nota, hidden]

                h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])  # [pair_of_nota, head, max_len]
                t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])  # [pair_of_nota, head, max_len]
                ht_att = F.normalize((h_att * t_att).mean(1), p=1, dim=1)  # [pair_of_nota, max_len]
                rs = contract("ld,rl->rd", sequence_output[batch_i, support_i], ht_att)  # [pair_of_nota, hidden]

                hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # [pair_of_nota, hidden]
                ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # [pair_of_nota, hidden]
                episode_relation_embeddings["NOTA"].append(torch.cat([hs, ts], dim=1))

            for rel_i, rel_type in enumerate(type_labels[batch_i]):
                if rel_type != "NOTA":
                    episode_prototypes[rel_i] = torch.mean(torch.cat(episode_relation_embeddings[rel_type], dim=0),
                                                           dim=0, keepdim=True)  # [1, 2*hidden]

            episode_nota_embeddings = torch.cat(episode_relation_embeddings["NOTA"],
                                                dim=0)  # [nota_in_episode, 2*hidden]
            episode_prototype_embeddings = torch.cat(episode_prototypes[1:], dim=0)  # [rel_in_episode, 2*hidden]
            if self.first_run and self.training:
                self.nota_embeddings.data = torch.mean(episode_nota_embeddings, dim=0, keepdim=True)
                indexes = torch.randperm(episode_nota_embeddings.shape[0])
                self.nota_embeddings.data = episode_nota_embeddings[indexes[:self.base_nota_num], :]
                self.first_run = False
            class_scores_1 = episode_nota_embeddings.unsqueeze(0) * self.nota_embeddings.unsqueeze(
                1)  # [base_nota_num, nota_in_episode, 2*hidden]
            class_scores_1 = torch.sum(class_scores_1, dim=-1)  # [base_nota_num, nota_in_episode]
            class_scores_2 = episode_nota_embeddings.unsqueeze(0) * episode_prototype_embeddings.unsqueeze(
                1)  # [rel_in_episode, nota_in_episode, 2*hidden]
            class_scores_2 = torch.sum(class_scores_2, dim=-1)  # [rel_in_episode, nota_in_episode]
            class_scores_2 = torch.max(class_scores_2, dim=0, keepdim=True)[0]  # [1, nota_in_episode]
            class_scores = class_scores_1 - class_scores_2  # [base_nota_num, nota_in_episode]
            top_class_scores_indices = \
                torch.topk(class_scores, min(1, class_scores.shape[1]), dim=-1)[
                    1]  # [base_nota_num, 1]
            top_class_scores_indices = top_class_scores_indices.view(-1)  # [base_nota_num]
            correction_nota_embeddings = episode_nota_embeddings[top_class_scores_indices, :].view(self.base_nota_num,
                                                                                                   1,
                                                                                                   -1)  # [base_nota_num, 1, 2*hidden]
            correction_nota_embeddings = correction_nota_embeddings.mean(1)  # [base_nota_num, 2*hidden]
            episode_prototypes[0] = (
                                            1 - self.nota_rectification_factor) * self.nota_embeddings + self.nota_rectification_factor * correction_nota_embeddings  # [base_nota_num, 2*hidden]

            prototypes.append(episode_prototypes)

            if self.training:
                for rel_type, rel_embeddings in episode_relation_embeddings.items():
                    if rel_type != "NOTA":
                        if rel_type in batch_relation_embeddings:
                            batch_relation_embeddings[rel_type].append(torch.cat(rel_embeddings, dim=0))
                        else:
                            batch_relation_embeddings[rel_type] = [torch.cat(rel_embeddings, dim=0)]

        # -------------- build and match candidate representations from queries ------------

        batch_size, num_queries, max_len = query_input_ids.size()

        # -------------- build labels according to prototypes ------------

        sequence_output, attention = self.encode(query_input_ids.view(-1, max_len), query_masks.view(-1, max_len))
        sequence_output = sequence_output.view(batch_size, num_queries, max_len, -1)  # [b, q_doc_num, max_len, hidden]
        attention = attention.view(batch_size, num_queries, -1, max_len,
                                   max_len)  # [b, q_doc_num, head, max_len, max_len]

        all_matches = []
        loss = 0
        for batch_i in range(batch_size):
            matches = [[] for _ in range(num_queries)]
            for query_i in range(num_queries):
                entity_embs, entity_atts = [], []
                for entity in query_entity_positions[batch_i][query_i]:
                    if len(entity) > 1:
                        e_emb, e_att = [], []
                        for start, end in entity:
                            if start < max_len:
                                e_emb.append(sequence_output[batch_i, query_i, start])
                                e_att.append(attention[batch_i, query_i, :, start])
                        if len(e_emb) > 0:
                            e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                            e_att = torch.stack(e_att, dim=0).mean(0)
                        else:
                            e_emb = torch.zeros(self.hidden_size).to(self.device)
                            e_att = torch.zeros(self.head_num, max_len).to(self.device)
                    else:
                        start, end = entity[0]
                        if start < max_len:
                            e_emb = sequence_output[batch_i, query_i, start]
                            e_att = attention[batch_i, query_i, :, start]
                        else:
                            e_emb = torch.zeros(self.hidden_size).to(self.device)
                            e_att = torch.zeros(self.head_num, max_len).to(self.device)
                    entity_embs.append(e_emb)
                    entity_atts.append(e_att)

                entity_embs = torch.stack(entity_embs, dim=0)  # [e, hidden]
                entity_atts = torch.stack(entity_atts, dim=0)  # [e, head, max_len]
                n_e = entity_embs.size(0)

                hts = [[i, j] for i in range(n_e) for j in range(n_e) if i != j]
                ht_i = torch.LongTensor(hts).to(self.device)  # [pair, 2]
                hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # [pair, hidden]
                ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # [pair, hidden]

                h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])  # [pair, head, max_len]
                t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])  # [pair, head, max_len]
                ht_att = F.normalize((h_att * t_att).mean(1), p=1, dim=1)  # [pair, max_len]
                rs = contract("ld,rl->rd", sequence_output[batch_i, query_i], ht_att)  # [pair, hidden]

                hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # [pair, hidden]
                ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # [pair, hidden]
                candidates = torch.cat([hs, ts], dim=1)  # [pair, 2*hidden]

                if self.normalize_query:
                    candidates = F.normalize(candidates, p=2, dim=1)

                scores = []
                for class_prototypes in prototypes[batch_i]:
                    if self.normalize_prototype:
                        class_prototypes = F.normalize(class_prototypes, p=2, dim=1)
                    if self.distance_metric == "dot_product":
                        class_scores = candidates.unsqueeze(0) * class_prototypes.unsqueeze(
                            1)  # [1 or base_nota_num, pair, 2*hidden]
                        class_scores = torch.sum(class_scores, dim=-1)  # [1 or base_nota_num, pair]
                    else:
                        class_scores = torch.pow(candidates.unsqueeze(0) - class_prototypes.unsqueeze(1),
                                                 2)  # [1 or base_nota_num, pair, 2*hidden]
                        class_scores = -torch.sum(class_scores, dim=-1)  # [1 or base_nota_num, pair]
                    class_scores = class_scores.max(dim=0, keepdim=False)[0]  # [pair]
                    scores.append(class_scores)
                scores = torch.stack(scores).transpose(0, 1)  # [pair, r_types_in_an_episode]

                predictions_binary = self.loss_fnt.get_label(scores.detach(),
                                                             num_labels=num_classes)  # [pair, r_types_in_an_episode]
                for i in range(len(hts)):
                    for j in range(1, len(type_labels[batch_i])):
                        if predictions_binary[i, j] == 1.0:
                            matches[query_i].append([hts[i][0], hts[i][1], type_labels[batch_i][j]])

                # ------- LOSS CALCULATION --------
                if query_labels is not None:
                    hts_map = {}
                    for i, ht in enumerate(hts):
                        hts_map[(ht[0], ht[1])] = i
                    labels = torch.zeros(len(hts), len(type_labels[batch_i])).to(self.device)
                    for rel_type, rel_hts in query_labels[batch_i][query_i].items():
                        if rel_type != "NOTA":
                            rel_index = type_labels[batch_i].index(rel_type)
                            for ht in rel_hts:
                                labels[hts_map[(ht[0], ht[1])], rel_index] = 1
                        else:
                            for ht in rel_hts:
                                labels[hts_map[(ht[0], ht[1])], 0] = 1
                    loss += self.loss_fnt(scores.float(), labels.float())

            all_matches.append(matches)

        if query_labels is not None:
            features = []
            labels = []
            for rel_type, rel_embeddings in batch_relation_embeddings.items():
                rel_feature = torch.cat(rel_embeddings, dim=0)
                features.append(rel_feature)
                rel_label = torch.full((rel_feature.shape[0],), relation_map[rel_type]).to(self.device)
                labels.append(rel_label)
            features = torch.cat(features, dim=0)
            features = F.normalize(features, p=2, dim=1)
            labels = torch.cat(labels, dim=0)
            relation_features = relation_cls[labels]
            relation_features = F.normalize(relation_features, p=2, dim=1)
            relation_similarity = torch.matmul(relation_features, relation_features.T)
            relation_similarity = (relation_similarity + 1) / 2
            sup_loss = self.loss_sup(features.unsqueeze(1), labels=labels,
                                     relation_similarity=relation_similarity.detach())
            sup_loss_value = sup_loss.detach().item()
            if sup_loss_value > 0:
                loss_value = loss.detach().item()
                if sup_loss_value > loss_value:
                    sup_loss = loss_value / sup_loss_value * sup_loss
                loss += self.lamda * sup_loss

            loss = loss / batch_size
            return all_matches, loss
        return all_matches
