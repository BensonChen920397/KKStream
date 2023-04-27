import sys, logging
import json, pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as ss
import faiss
from pathlib import Path
from random import shuffle
from sklearn import metrics
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import IterableDataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import torch.optim.lr_scheduler as lr_scheduler

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        pretrained_model,
        pooling,
        channels: int = 1,
    ):
        super(TransformerEncoder, self).__init__()
        self.transformers = torch.nn.ModuleList()
        for c in range(channels):
            self.transformers.append(AutoModel.from_pretrained(pretrained_model))
        if pooling=='true':
            self.pooling = mean_pooling
        else:
            self.pooling = None

    def forward(
        self,
        tokens,
        channel,
    ):
        embeddings = self.transformers[channel](
            torch.squeeze(tokens['input_ids']),
            torch.squeeze(tokens['attention_mask'])
        )

        if self.pooling != None:
            embeddings = self.pooling(embeddings, torch.squeeze(tokens['attention_mask']))
        else:
            # Pick last_hidden_state 
            embeddings = embeddings[0]
            # Pick CLS embedding
            embeddings = embeddings[:, 0]

        return embeddings


class TwoTower(nn.Module):

    def __init__(
        self,
        pretrained_model,
        if_pooling,
        channels: int = 1,
    ):
        super(TwoTower, self).__init__()
        self.tower = TransformerEncoder(pretrained_model, if_pooling, channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, query_tokens, doc_tokens, labels):
        query_embeddings = self.tower(query_tokens, 0)
        doc_embeddings = self.tower(doc_tokens, 0)

        scores = torch.cosine_similarity(query_embeddings, doc_embeddings)
        loss = self.loss_fn(scores, labels).sum()
        return loss, scores


    def embeds(self, doc_tokens):
        return self.tower(doc_tokens, 0)


class FourTower(nn.Module):

    def __init__(
        self,
        pretrained_model,
        if_pooling,
        channels: int = 2,
    ):
        super(FourTower, self).__init__()
        self.towers = TransformerEncoder(pretrained_model, if_pooling, channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, query_tokens, q_meta_tokens, doc_tokens, d_meta_tokens, labels):
        query_embeddings = self.towers(query_tokens, 0)
        doc_embeddings = self.towers(doc_tokens, 0)
        q_meta_embeddings = self.towers(q_meta_tokens, 1)
        d_meta_embeddings = self.towers(d_meta_tokens, 1)

        scores = torch.cosine_similarity(
            (query_embeddings+q_meta_embeddings),
            (doc_embeddings+d_meta_embeddings)
        )
        loss = self.loss_fn(scores, labels).sum()
        return loss, scores


    def embeds(self, doc_tokens, meta_tokens):
        return self.towers(doc_tokens, 0)+self.towers(meta_tokens, 1)



def dump_code2tokens(
    kg_text_path: Path,
    kg_path: Path,
    pretrained_model,
    pickle_path: Path,
    with_meta,
    max_token_length = 64,
    device = 'cpu',
):
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = SentenceTransformer('bert-base-multilingual-cased', device=device)

    logger.info('Load from')
    logger.info(kg_path)
    e2m_list = defaultdict(list)
    with kg_path.open('r') as f:
        for line in f:
            ent, meta, _ = line.rstrip('\n').split(' ')
            e2m_list[ent].append(meta)

    code2text = {}
    logger.info('Load from')
    logger.info(kg_text_path)
    with kg_text_path.open('r') as f:
        for line in tqdm(f):
            text, code = line.rstrip('\n').split('\t')
            code2text[code] = text
    codes = [code for code in list(code2text.keys()) if code[0]=='e']

    code2tokens = {}
    # get embeddings
    logger.info('pre-compute embeddings')
    if with_meta=='true':
        entity_texts = [code2text[code] for code in codes]
        meta_texts = [
            ' '.join([code2text[m] for m in e2m_list[code]]) for code in codes]
        entity_embeds = np.array(model.encode(entity_texts), dtype='float32')
        meta_embeds = np.array(model.encode(meta_texts), dtype='float32')
        for code, e1, e2 in zip(codes, entity_embeds, meta_embeds):
            code2tokens[code] = {}
            code2tokens[code]['entity_embed'] = e1
            code2tokens[code]['meta_embed'] = e2
        # get tokens
        logger.info('pre-compute tokenizations')
        for code in tqdm(codes, total=len(codes)):
            tokens = tokenizer(
                f"{code2text[code]}",
                truncation=True,
                max_length=max_token_length,
                padding='max_length',
                return_tensors="pt",
            )
            code2tokens[code]['entity_tokens'] = tokens

            meta_text = ' '.join([code2text[m] for m in e2m_list[code]])
            tokens = tokenizer(
                f"{meta_text}",
                truncation=True,
                max_length=max_token_length,
                padding='max_length',
                return_tensors="pt",
            )
            code2tokens[code]['meta_tokens'] = tokens

        logger.info('Save to')
        logger.info(pickle_path)
        pickle.dump(code2tokens, pickle_path.open('wb'))
    else:
        entity_texts = [code2text[code] for code in codes]
        entity_embeds = np.array(model.encode(entity_texts), dtype='float32')
        for code, e1 in zip(codes, entity_embeds):
            code2tokens[code] = {}
            code2tokens[code]['entity_embed'] = e1

        # get tokens
        logger.info('pre-compute tokenizations')
        for code in tqdm(codes, total=len(codes)):
            tokens = tokenizer(
                f"{code2text[code]}",
                truncation=True,
                max_length=max_token_length,
                padding='max_length',
                return_tensors="pt",
            )
            code2tokens[code]['entity_tokens'] = tokens

        logger.info('Save to')
        logger.info(pickle_path)
        pickle.dump(code2tokens, pickle_path.open('wb'))


class RawDataset(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        code2tokens_path: Path,
        with_meta,
    ):
        self.code2tokens = pickle.load(code2tokens_path.open('rb'))
        self.with_meta = with_meta

        return self

    def __len__(self):
        return len(self.code2tokens)

    def __iter__(self):
        if self.with_meta=='true':
            for code in self.code2tokens:
                yield \
                    code, \
                    self.code2tokens[code]['entity_tokens'], \
                    self.code2tokens[code]['meta_tokens']
        else:
            for code in self.code2tokens:
                yield \
                    code, \
                    self.code2tokens[code]['entity_tokens']

class PairsDataset(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        train_pairs_path: Path,
        code2tokens_path: Path,
        with_meta,
        negative_type,
    ):
        logger.info('Load tokens/embeds')
        self.code2tokens = pickle.load(code2tokens_path.open('rb'))
        self.pairs = []
        with train_pairs_path.open('r') as f:
            for line in f:
                code1, code2 = line.rstrip('\n').split('\t')
                self.pairs.append((code1, code2))

        logger.info('build ANN')
        self.codes = [code for code in self.code2tokens]
        self.index = {code:e for e, code in enumerate(self.codes)}
        self.entity_embeds = np.array(
            [self.code2tokens[code]['entity_embed'] for code in self.codes], dtype='float32')
        self.meta_embeds = np.array(
            [self.code2tokens[code]['meta_embed'] for code in self.codes], dtype='float32')
        self.entity_nn = faiss.IndexFlatIP(len(self.entity_embeds[0]))
        self.meta_nn = faiss.IndexFlatIP(len(self.meta_embeds[0]))
        self.entity_nn.add(self.entity_embeds)
        self.meta_nn.add(self.meta_embeds)
        self.with_meta = with_meta
        self.negative_type = negative_type

        return self


    def __iter__(self):
        shuffle(self.pairs)
        rand_max = len(self.codes) - 1
        retrieval_max = 200
        if self.with_meta=='true':
            for pair in self.pairs:
                # positive
                yield \
                    self.code2tokens[pair[0]]['entity_tokens'], \
                    self.code2tokens[pair[0]]['meta_tokens'], \
                    self.code2tokens[pair[1]]['entity_tokens'], \
                    self.code2tokens[pair[1]]['meta_tokens'], \
                    1.0
                if self.negative_type == 'easy':
                    # entity-based negatives
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    # meta-based negatives
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(1, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                elif self.negative_type == 'hard':
                    query_idx0 = self.index[pair[0]]
                    query_idx1 = self.index[pair[1]]
                    # entity-based negatives
                    _, similars0 = self.entity_nn.search(
                        self.entity_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.entity_nn.search(
                        self.entity_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    # meta-based negatives
                    _, similars0 = self.meta_nn.search(
                        self.meta_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.meta_nn.search(
                        self.meta_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                elif self.negative_type == 'easy+hard':
                    query_idx0 = self.index[pair[0]]
                    query_idx1 = self.index[pair[1]]
                    # entity-based negatives
                    _, similars0 = self.entity_nn.search(
                        self.entity_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.entity_nn.search(
                        self.entity_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0

                    # meta-based negatives
                    _, similars0 = self.meta_nn.search(
                        self.meta_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.meta_nn.search(
                        self.meta_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[pair[0]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(1, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[pair[1]]['meta_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            self.code2tokens[rand_code]['meta_tokens'], \
                            0.0    
                else:
                    raise Exception("Out of type!!!")
        else:
            for pair in self.pairs:
                # positive
                yield \
                    self.code2tokens[pair[0]]['entity_tokens'], \
                    self.code2tokens[pair[1]]['entity_tokens'], \
                    1.0
                if self.negative_type == 'easy':
                    # entity-based negatives
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                    for _ in range(4): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                elif self.negative_type == 'hard':
                    query_idx0 = self.index[pair[0]]
                    query_idx1 = self.index[pair[1]]
                    # entity-based negatives
                    _, similars0 = self.entity_nn.search(
                        self.entity_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.entity_nn.search(
                        self.entity_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                    for _ in range(4): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                elif self.negative_type == 'easy+hard':
                    query_idx0 = self.index[pair[0]]
                    query_idx1 = self.index[pair[1]]
                    # entity-based negatives
                    _, similars0 = self.entity_nn.search(
                        self.entity_embeds[query_idx0:query_idx0+1], retrieval_max)
                    _, similars1 = self.entity_nn.search(
                        self.entity_embeds[query_idx1:query_idx1+1], retrieval_max)
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars0[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[0]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                    for _ in range(2): # hard negative
                        rand_code = self.codes[similars1[0][random.randint(10, retrieval_max-1)]]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                    for _ in range(2): # easy negative
                        rand_code = self.codes[random.randint(0, rand_max)]
                        yield \
                            self.code2tokens[pair[1]]['entity_tokens'], \
                            self.code2tokens[rand_code]['entity_tokens'], \
                            0.0
                else:
                    raise Exception("Out of type!!!")


if __name__ == '__main__':
    pairs_path = sys.argv[1]
    with_meta = sys.argv[2]
    if_pooling = sys.argv[3]
    fixed_sampling = sys.argv[4]
    negative_type = sys.argv[5]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pretrained_model = 'bert-base-multilingual-cased'
    pretrained_ckpt = ''
    model_path = Path('../model_status')

    if if_pooling == 'true':
        if_pooling_suffix = '1'
    else:
        if_pooling_suffix = '0'

    if fixed_sampling == 'true':
        fixed_sampling_suffix = '1'
    else:
        fixed_sampling_suffix = '0'

    # Print setting
    print("The setting of model is \n [")
    print("\tWith meta : ", with_meta)
    print("\tIf pooling : ", if_pooling)
    print("\tFixed_sampling : ", fixed_sampling)
    print("\tNegative type : ", negative_type)
    print("]")

    if with_meta=='true':
        embed_path = Path('../embedding_result/fourtower.embed')
        fourtower = FourTower(pretrained_model, if_pooling, 2)
        if pretrained_ckpt:
            fourtower.load_state_dict(torch.load(model_path/pretrained_ckpt))
        fourtower.to(device)
    else:
        embed_path = Path('../embedding_result/twotower.embed')
        twotower = TwoTower(pretrained_model, if_pooling)
        if pretrained_ckpt:
            twotower.load_state_dict(torch.load(model_path/pretrained_ckpt))
        twotower.to(device)

    dump_code2tokens(
        kg_text_path=Path('../../dataset/line_kg/line-kg.idx.txt'),
        kg_path=Path('../../dataset/line_kg/line-kg.txt'),
        pretrained_model=pretrained_model,
        pickle_path=Path(f'../code2tokens'),
        with_meta=with_meta,
        device=device,
    )

    if fixed_sampling:
        random.seed(10)

    pairs_dataset = PairsDataset()
    dataloader = torch.utils.data.DataLoader(
        pairs_dataset(
            train_pairs_path=Path(pairs_path),
            code2tokens_path=Path('../code2tokens'),
            with_meta=with_meta,
            negative_type=negative_type,
        ),
        batch_size=20
    )

    if with_meta=="true":
        optimizer = optim.AdamW(fourtower.parameters(), lr=0.00002)
    else:
        optimizer = optim.AdamW(twotower.parameters(), lr=0.00002)

    optimizer.zero_grad()
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    total_loss, total_cnt = 0., 0
    if with_meta=="true":
        for epoch in trange(3):
            for query_tokens, q_meta_tokens, doc_tokens, d_meta_tokens, labels in tqdm(dataloader):
                loss, scores = fourtower.forward(
                    query_tokens=query_tokens.to(device),
                    q_meta_tokens=q_meta_tokens.to(device),
                    doc_tokens=doc_tokens.to(device),
                    d_meta_tokens=d_meta_tokens.to(device),
                    labels=labels.type(torch.float).to(device),
                )
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
                total_cnt += 1

                if (total_cnt % 10)==0:
                    logger.info(f"Avg. Loss: {total_loss/10}")
                    total_cnt, total_loss = 0., 0.
        model_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nsave ckpt to\t{model_path}/{epoch}")
        torch.save(fourtower.state_dict(), model_path/f'{epoch}')
    else:
        for epoch in trange(3):
            for query_tokens, doc_tokens, labels in tqdm(dataloader):
                loss, scores = twotower.forward(
                    query_tokens=query_tokens.to(device),
                    doc_tokens=doc_tokens.to(device),
                    labels=labels.type(torch.float).to(device),
                )
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
                total_cnt += 1

                if (total_cnt % 10)==0:
                    logger.info(f"Avg. Loss: {total_loss/10}")
                    total_cnt, total_loss = 0., 0.

        model_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nsave ckpt to\t{model_path}/{epoch}")
        torch.save(twotower.state_dict(), model_path/f'{epoch}')

    raw_dataset = RawDataset()
    dataloader_raw = torch.utils.data.DataLoader(
        raw_dataset(
            code2tokens_path=Path('../code2tokens'),
            with_meta=with_meta,
        ),
        batch_size=20
    )
    embeddings, dim = [], 0
    if with_meta=='true':
        for codes, doc_tokens, meta_tokens in tqdm(dataloader_raw):
            embeds = fourtower.embeds(
                doc_tokens=doc_tokens.to(device),
                meta_tokens=meta_tokens.to(device),
            )
            for code, embed in zip(codes, embeds):
                embed = ' '.join(map(str, embed.tolist()))
                # embeddings.append(f"{code}\t{embed}")
                embeddings.append(f"{code} {embed}")
            dim = len(embeds[0]) 
    else:
        for codes, doc_tokens in tqdm(dataloader_raw):
            embeds = twotower.embeds(
                doc_tokens=doc_tokens.to(device),
            )
            for code, embed in zip(codes, embeds):
                embed = ' '.join(map(str, embed.tolist()))
                # embeddings.append(f"{code}\t{embed}")
                embeddings.append(f"{code} {embed}")
            dim = len(embeds[0]) 

    embeddings = [f"{len(embeddings)} {dim}"] + embeddings

    if embed_path.parent != '.':
        embed_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Save embeddings to')
    logger.info(embed_path)
    with embed_path.open('w') as f:
        f.write('\n'.join(embeddings))
        f.write('\n')


