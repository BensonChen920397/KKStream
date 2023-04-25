import sys, logging
import json, pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as ss
from random import shuffle
from sklearn import metrics
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import IterableDataset
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List

from utils import acc
from fastformer import Fastformer
from fastformer2 import FastformerEncoder

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class AdditiveAttention(nn.Module):
    def __init__(self, dim=100, r=2.):
        super().__init__()
        intermediate = int(dim * r)
        self.attn = nn.Sequential(
            nn.Linear(dim, intermediate),
            nn.Dropout(0.01),
            nn.LayerNorm(intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 1),
            nn.Softmax(1),
        )
        self.attn.apply(init_weights)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim]
        Returns:
            outputs, weights: [B, seq_len, dim], [B, seq_len]
        """
        w = self.attn(context).squeeze(-1)
        return torch.bmm(w.unsqueeze(1), context).squeeze(1), w


class AttentionPooling(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, emb_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, emb_dim
        """
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class NewsEncoder(nn.Module):

    def __init__(
        self,
    ):
        super(NewsEncoder, self).__init__()
        bert = AutoModel.from_pretrained('distilbert-base-cased')
        self.bert_att1 = bert.transformer.layer[0]
        self.bert_att2 = bert.transformer.layer[-1]
        self.dim = bert.config.hidden_size
        self.embeddings = bert.embeddings
        #self.embeddings = nn.Embedding.from_pretrained(bert.embeddings.word_embeddings.weight)
        #self.ff = Fastformer(dim=self.dim, decode_dim=self.dim)
        #self.ffe = FastformerEncoder()
        #self.mha_attention = torch.nn.MultiheadAttention(self.dim, 8, batch_first=True)
        #self.add_attentionT = AdditiveAttention(self.dim, 0.5)
        #self.add_attentionD = AdditiveAttention(self.dim, 0.5)
        #self.position_embeddings = nn.Embedding(128, self.dim)
        #self.layer_norm = nn.LayerNorm(self.dim)
        #self.dropout = nn.Dropout(0.2)

        self.attention_poolingT = AttentionPooling(self.dim, 128)
        self.attention_poolingD = AttentionPooling(self.dim, 128)
        self.poolerT = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.01),
            nn.LayerNorm(self.dim),
            #nn.SiLU(),
        )
        self.poolerD = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.01),
            nn.LayerNorm(self.dim),
            #nn.SiLU(),
        )

    def forward(
        self,
        tokens,
        pooling,
    ):
        # tokens -> embeddings
        if pooling:
            embeddings = self.embeddings(tokens['input_ids'][0])
            embeddings = self.bert_att1(embeddings, attn_mask=tokens['token_mask'][0])[0]
            embeddings = self.attention_poolingT(embeddings, tokens['token_mask'][0])
            embeddings.unsqueeze_(0)
            embeddings = self.bert_att2(embeddings, attn_mask=tokens['query_mask'])[0]
            embeddings = self.attention_poolingD(embeddings, tokens['query_mask'])
            #print(embeddings.shape)
            #positions = torch.arange(
            #    embeddings.shape[1], dtype=torch.long, device=embeddings.device)
            #embeddings = embeddings + self.position_embeddings(positions)
            #embeddings = self.bert_att(embeddings, attn_mask=tokens['query_mask'])[0]
            # tokens -> docs
            #embeddings = self.ffe(embeddings, tokens['token_mask'][0])
            # docs -> user
            #embeddings = self.ffe(embeddings, tokens['query_mask'])
            #embeddings = self.layer_norm(embeddings)
            #embeddings = self.poolerD(embeddings)
            return embeddings
        else:
            embeddings = self.embeddings(tokens['input_ids'][0])
            embeddings = self.bert_att1(embeddings, attn_mask=tokens['token_mask'])[0]
            #embeddings = self.poolerT(embeddings)
            #embeddings = self.ffe(embeddings, tokens['token_mask'][0])
            #embeddings = self.layer_norm(embeddings)
            embeddings = self.attention_poolingT(embeddings, tokens['token_mask'][0])
            return embeddings


class NRMS(nn.Module):

    def __init__(
        self,
    ):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder()
        #self.loss_fn = torch.nn.LogSigmoid()
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, user_tokens, item_tokens, labels):
        user_embedding = self.news_encoder(user_tokens, pooling=True)
        item_embedding = self.news_encoder(item_tokens, pooling=False)

        scores = torch.sum(user_embedding*item_embedding, dim=1)
        #scores = (user_embedding*item_embedding).sum(1)

        #loss = -self.loss_fn(scores*labels).sum() # negative log-likelihood
        loss = self.loss_fn(torch.flatten(scores), torch.flatten(labels)).sum()
        return loss, scores


def dump_news2tokens(
    news_paths: List[Path],
    pickle_path: Path,
    max_token_length,
):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    bert = AutoModel.from_pretrained('distilbert-base-cased')
    pooler = nn.Sequential(
        nn.Linear(self.dim, self.dim),
        nn.Dropout(0.01),
        nn.LayerNorm(self.dim),
        nn.SiLU(),
    )

    news2tokens = {}
    for path in news_paths:
        logger.info('Load from')
        logger.info(path)
        with path.open('r') as f:
            for line in tqdm(f):
                line = line.rstrip('\n').split('\t')
                news = line[0]
                cat1, cat2 = line[1], line[2]
                title = line[3]
                abstract = line[4]
                entities = eval(line[7])
                entities = [ent['Label'] for ent in entities]
                entities = ' '.join(entities)
                if news in news2tokens:
                    continue
                tokens = tokenizer(
                    f"{cat1} {cat2} {entities}",
                    truncation=True,
                    max_length=max_token_length,
                    padding='max_length',
                    return_tensors="pt",
                )
                news2tokens[f"{news}-meta"] = tokens
                tokens = tokenizer(
                    f"{title} {abstract}",
                    truncation=True,
                    max_length=max_token_length,
                    padding='max_length',
                    return_tensors="pt",
                )
                news2tokens[f"{news}-text"] = tokens
                tokens = tokenizer(
                    f"{cat1} {cat2} {entities} {title} {abstract}",
                    truncation=True,
                    max_length=max_token_length,
                    padding='max_length',
                    return_tensors="pt",
                )
                news2tokens[f"{news}"] = tokens
    # for empty news
    tokens = tokenizer(
        'OOO',
        truncation=True,
        max_length=max_token_length,
        padding='max_length',
        return_tensors="pt",
    )
    news2tokens[f'-meta'] = tokens
    news2tokens[f'-text'] = tokens
    news2tokens[f''] = tokens

    logger.info('Save to')
    logger.info(pickle_path)
    pickle.dump(news2tokens, pickle_path.open('wb'))


class MindDataset(IterableDataset):

    def __init__(self):
        pass

    def __call__(
        self,
        news2tokens,
        behavior_path,
        max_query_length,
        neg_ratio,
        pos_label,
        neg_label,
        has_label,
    ):
        self.news2tokens = news2tokens
        self.behavior_path = behavior_path
        self.max_query_length = max_query_length
        self.neg_ratio = neg_ratio
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.has_label = has_label
        return self


    def __iter__(self):
        epoch = 2 if self.neg_ratio else 1
        for _ in range(epoch):
            with self.behavior_path.open('r') as f:
                for line in f:
                    line = line.rstrip('\n').split('\t')
                    session = line[0]
                    user_news = line[3].split(' ')
                    item_news = line[4].split(' ')
                    labels = [neg_label for _ in range(len(item_news))]
                    if self.has_label:
                        item_news = [i_news.split('-') for i_news in item_news]
                        labels = [
                            self.pos_label if i_news[1]=='1' else self.neg_label
                                for i_news in item_news
                        ]
                        item_news = [i_news[0] for i_news in item_news]

                    # user
                    user_news = user_news[:self.max_query_length]
                    resize = self.max_query_length - len(user_news)
                    if resize:
                        user_news += ['' for _ in range(resize)]
                    user_tokens = BatchEncoding() # FIXME any better way to concat them?
                    user_tokens['input_ids'] = torch.concat(
                        [self.news2tokens[news]['input_ids'] for news in user_news]
                    )
                    user_tokens['query_mask'] = torch.tensor(
                        [1. if news else 0. for news in user_news]
                    )
                    user_tokens['token_mask'] = torch.concat(
                        [self.news2tokens[news]['attention_mask'] for news in user_news]
                    )

                    # item
                    item_tokens = BatchEncoding() # FIXME any better way to concat them?

                    if self.has_label and self.neg_ratio:
                        item_news_pos, item_news_neg = [], []
                        for news, label in zip(item_news, labels):
                            if label==self.pos_label:
                                item_news_pos.append(news)
                            else:
                                item_news_neg.append(news)

                        for pos in item_news_pos:
                            shuffle(item_news_neg)
                            negs = item_news_neg[:self.neg_ratio]
                            item_news = [pos] + negs
                            labels = [self.pos_label] + [self.neg_label for _ in range(len(negs))]
                            item_tokens['input_ids'] = torch.concat(
                                [self.news2tokens[news]['input_ids'] for news in item_news]
                            )
                            item_tokens['token_mask'] = torch.concat(
                                [self.news2tokens[news]['attention_mask'] for news in item_news]
                            )
                            yield user_tokens, item_tokens, torch.tensor(labels), session

                    else:
                        item_tokens['input_ids'] = torch.concat(
                            [self.news2tokens[news]['input_ids'] for news in item_news]
                        )
                        item_tokens['token_mask'] = torch.concat(
                            [self.news2tokens[news]['attention_mask'] for news in item_news]
                        )
                        yield user_tokens, item_tokens, torch.tensor(labels), session



if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    pos_label = 1.
    neg_label = 0.
    neg_ratio = 5
    batch_size = 1
    max_token_length = 30
    max_query_length = 16
    train_ckpt = ''
    dev_ckpt = '180000'
    test_ckpt = '30000'
    model_path = Path('model_states_bertl2')
    model_path.mkdir(parents=True, exist_ok=True)

    logger.info('prepare data')
    '''
    dump_news2tokens(
        news_paths=[
            Path('../data/train/news.tsv'),
            Path('../data/dev/news.tsv'),
            Path('../data/test/news.tsv'),
        ],
        max_token_length=max_token_length,
        pickle_path=Path('exp/news2tokens'),
    )
    '''

    logger.info('Load news2tokens ...')
    news2tokens = pickle.load(Path('exp/news2tokens').open('rb'))
    mind_dataset = MindDataset()

    if sys.argv[1] == 'train':
        dataloader = torch.utils.data.DataLoader(
            mind_dataset(
                news2tokens = news2tokens,
                behavior_path = Path('../data/train/behaviors.tsv'),
                max_query_length = max_query_length,
                pos_label = pos_label,
                neg_label = neg_label,
                neg_ratio = neg_ratio,
                has_label = True,
            ),
            batch_size=batch_size
        )

        logger.info('prepare model')
        nrms = NRMS()
        if train_ckpt:
            nrms.load_state_dict(torch.load(model_path/train_ckpt))
        nrms.to(device)
        optimizer = optim.Adam(nrms.parameters(), lr=0.001)

        total_loss, total_cnt = 0., 0
        for user_tokens, item_tokens, labels, session in tqdm(dataloader):
            user_tokens = user_tokens.to(device)
            item_tokens = item_tokens.to(device)
            labels = labels.to(device).type(torch.float32)

            loss, scores = nrms.forward(
                user_tokens=user_tokens,
                item_tokens=item_tokens,
                labels=labels[0],
            )
            total_loss += loss
            loss.backward()
            if random.randint(0, 9) == 9:
                optimizer.step()
                optimizer.zero_grad()
            total_cnt += 1

            if total_cnt and total_cnt % 10000 == 0:
                logger.info(f"\nsave ckpt to\t{model_path}/{total_cnt}")
                torch.save(nrms.state_dict(), model_path/f'{total_cnt}')
                logger.info(f"Avg. Loss: {total_loss/10000}")
                total_loss = 0.


    if sys.argv[1] == 'dev':
        dataloader = torch.utils.data.DataLoader(
            mind_dataset(
                news2tokens = news2tokens,
                behavior_path = Path('../data/dev/behaviors.tsv'),
                max_query_length = max_query_length,
                pos_label = pos_label,
                neg_label = neg_label,
                neg_ratio = 0,
                has_label = True,
            ),
            batch_size=batch_size,
            pin_memory=True,
        )

        with torch.no_grad():
            nrms = NRMS()
            nrms.load_state_dict(torch.load(model_path/dev_ckpt))
            nrms.to(device)
            preditions = defaultdict(list)
            auc, cnt = 0., 0.
            for user_tokens, item_tokens, labels, session in tqdm(dataloader):
                user_tokens = user_tokens.to(device)
                item_tokens = item_tokens.to(device)
                labels = labels.to(device).type(torch.float32)

                _, scores = nrms.forward(
                    user_tokens=user_tokens,
                    item_tokens=item_tokens,
                    labels=labels[0],
                )

                fpr, tpr, thresholds = metrics.roc_curve(
                    labels.cpu()[0], scores.cpu(), pos_label=pos_label)
                auc += metrics.auc(fpr, tpr)
                cnt += 1.
                if cnt and (cnt % 1000) == 0:
                    logger.info(auc/1000)
                    auc, cnt = 0., 0.
                #for session, score in zip(sessions, scores.cpu()):
                preditions[session[0]].append(-scores.cpu().numpy())
            for session in preditions:
                orders = ss.rankdata(preditions[session])
                orders = list(map(str, map(int, orders)))
                orders = ','.join(orders)
                print(f"{session} [{orders}]")

    if sys.argv[1] == 'test':
        dataloader = torch.utils.data.DataLoader(
            mind_dataset(
                news2tokens = news2tokens,
                behavior_path = Path('../data/test/behaviors.tsv'),
                max_query_length = max_query_length,
                pos_label = pos_label,
                neg_label = neg_label,
                neg_ratio = 0,
                has_label = False,
            ),
            batch_size=batch_size,
            pin_memory=True,
        )

        with torch.no_grad():
            nrms = NRMS()
            nrms.load_state_dict(torch.load(model_path/test_ckpt))
            nrms.to(device)
            preditions = defaultdict(list)
            auc, cnt = 0., 0.
            for user_tokens, item_tokens, labels, session in tqdm(dataloader):
                user_tokens = user_tokens.to(device)
                item_tokens = item_tokens.to(device)
                labels = labels.to(device).type(torch.float32)

                _, scores = nrms.forward(
                    user_tokens=user_tokens,
                    item_tokens=item_tokens,
                    labels=labels[0],
                )
                #for session, score in zip(sessions, scores.cpu()):
                preditions[session[0]].append(-scores.cpu().numpy())
            for session in preditions:
                orders = ss.rankdata(preditions[session])
                orders = list(map(str, map(int, orders)))
                orders = ','.join(orders)
                print(f"{session} [{orders}]")

