# #ETH-GBert
# import re

# import numpy as np
# import scipy.sparse as sp
# import torch
# from nltk.tokenize import TweetTokenizer
# from torch.utils import data
# from torch.utils.data import (
#     DataLoader,
#     Dataset,
#     RandomSampler,
#     SequentialSampler,
#     TensorDataset,
#     WeightedRandomSampler,
# )
# from torch.utils.data.distributed import DistributedSampler

# """
# General functions
# """


# def del_http_user_tokenize(tweet):
#     space_pattern = r"\s+"
#     url_regex = (
#         r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
#         r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
#     )
#     mention_regex = r"@[\w\-]+"
#     tweet = re.sub(space_pattern, " ", tweet)
#     tweet = re.sub(url_regex, "", tweet)
#     tweet = re.sub(mention_regex, "", tweet)
#     return tweet


# def clean_str(string):
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
#     string = re.sub(r"\'s", " 's", string)
#     string = re.sub(r"\'ve", " 've", string)
#     string = re.sub(r"n\'t", " n't", string)
#     string = re.sub(r"\'re", " 're", string)
#     string = re.sub(r"\'d", " 'd", string)
#     string = re.sub(r"\'ll", " 'll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


# def clean_tweet_tokenize(string):
#     tknzr = TweetTokenizer(
#         reduce_len=True, preserve_case=False, strip_handles=False
#     )
#     tokens = tknzr.tokenize(string.lower())
#     return " ".join(tokens).strip()


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     # adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))  # D-degree matrix
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


# def sparse_scipy2torch(coo_sparse):
#     # coo_sparse=coo_sparse.tocoo()
#     i = torch.LongTensor(np.vstack((coo_sparse.row, coo_sparse.col)))
#     v = torch.from_numpy(coo_sparse.data)
#     return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


# def get_class_count_and_weight(y, n_classes):
#     classes_count = []
#     weight = []
#     for i in range(n_classes):
#         count = np.sum(y == i)
#         classes_count.append(count)
#         weight.append(len(y) / (n_classes * count))
#     return classes_count, weight


# """
# Functions and Classes for read and organize data set
# """


# class InputExample(object):


#     def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
#         self.guid = guid
#         # string of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
#         self.text_a = text_a
#         self.text_b = text_b
#         # the label(class) for the sentence
#         self.confidence = confidence
#         self.label = label


# class InputFeatures(object):
#     def __init__(
#         self,
#         guid,
#         tokens,
#         input_ids,
#         gcn_vocab_ids,
#         input_mask,
#         segment_ids,
#         confidence,
#         label_id,
#     ):
#         self.guid = guid
#         self.tokens = tokens
#         self.input_ids = input_ids
#         self.gcn_vocab_ids = gcn_vocab_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.confidence = confidence
#         self.label_id = label_id


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b):
#             tokens_a.pop()
#         else:
#             tokens_b.pop()

# def example2feature(example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim):
#     tokens_a = example.text_a.split()  # Assume text_a contains account addresses or transaction-related fields
#     assert example.text_b == None
#     if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
#         tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

#     gcn_vocab_ids = []
#     for word in tokens_a:
#         if word in gcn_vocab_map:
#             gcn_vocab_ids.append(gcn_vocab_map[word])
#         else:
#             # If the word/address is not in gcn_vocab_map, use a default value (e.g., -1 means not found)
#             gcn_vocab_ids.append(gcn_vocab_map.get('UNK', -1))  # 'UNK' can be replaced with a suitable default value
#     # Build BERT input tokens, including [CLS] and [SEP]
#     tokens = ["[CLS]"] + tokens_a + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
#     segment_ids = [0] * len(tokens)

#     # Convert tokens into input_ids required by BERT
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)

#     # Create and return the InputFeatures object
#     feat = InputFeatures(
#         guid=example.guid,
#         tokens=tokens,
#         input_ids=input_ids,
#         gcn_vocab_ids=gcn_vocab_ids,
#         input_mask=input_mask,
#         segment_ids=segment_ids,
#         confidence=example.confidence,
#         label_id=example.label,
#     )
#     return feat

# class CorpusDataset(Dataset):
#     def __init__(
#         self,
#         examples,
#         tokenizer,
#         gcn_vocab_map,
#         max_seq_len,
#         gcn_embedding_dim,
#     ):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
#         self.gcn_embedding_dim = gcn_embedding_dim
#         self.gcn_vocab_map = gcn_vocab_map

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         feat = example2feature(
#             self.examples[idx],
#             self.tokenizer,
#             self.gcn_vocab_map,
#             self.max_seq_len,
#             self.gcn_embedding_dim,
#         )
#         return (
#             feat.input_ids,
#             feat.input_mask,
#             feat.segment_ids,
#             feat.confidence,
#             feat.label_id,
#             feat.gcn_vocab_ids,
#         )

#     # @classmethod
#     def pad(self, batch):
#         gcn_vocab_size = len(self.gcn_vocab_map)
#         seqlen_list = [len(sample[0]) for sample in batch]
#         maxlen = np.array(seqlen_list).max()

#         f_collect = lambda x: [sample[x] for sample in batch]
#         f_pad = lambda x, seqlen: [
#             sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
#         ]
#         f_pad2 = lambda x, seqlen: [
#             [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
#             for sample in batch
#         ]

#         batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
#         batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
#         batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
#         batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
#         batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)
#         batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)
#         batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
#             batch_gcn_vocab_ids_paded
#         ][:, :-1]
#         batch_gcn_swop_eye = batch_gcn_swop_eye.view(
#             len(batch), -1, gcn_vocab_size
#         ).transpose(1, 2)

#         return (
#             batch_input_ids,
#             batch_input_mask,
#             batch_segment_ids,
#             batch_confidences,
#             batch_label_ids,
#             batch_gcn_swop_eye,
#         )


#ETH-GSetBert
import re

import numpy as np
import scipy.sparse as sp
import torch
from nltk.tokenize import TweetTokenizer
from torch.utils import data
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

"""
General functions
"""


def del_http_user_tokenize(tweet):
    space_pattern = r"\s+"
    url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    tweet = re.sub(space_pattern, " ", tweet)
    tweet = re.sub(url_regex, "", tweet)
    tweet = re.sub(mention_regex, "", tweet)
    return tweet


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(
        reduce_len=True, preserve_case=False, strip_handles=False
    )
    tokens = tknzr.tokenize(string.lower())
    return " ".join(tokens).strip()


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    rowsum[rowsum == 0] = 1.0      # tránh chia 0
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    coo = coo_sparse.tocoo()
    i = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, size=coo.shape)


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


"""
Functions and Classes for read and organize data set
"""


class InputExample(object):


    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        self.guid = guid
        # string of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.text_a = text_a
        self.text_b = text_b
        # the label(class) for the sentence
        self.confidence = confidence
        self.label = label


class InputFeatures(object):
    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def example2feature(example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim):
    tokens_a = example.text_a.split()  # Assume text_a contains account addresses or transaction-related fields
    assert example.text_b == None
    if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
        tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

    gcn_vocab_ids = []
    for word in tokens_a:
        if word in gcn_vocab_map:
            gcn_vocab_ids.append(gcn_vocab_map[word])
        else:
            # If the word/address is not in gcn_vocab_map, use a default value (e.g., -1 means not found)
            gcn_vocab_ids.append(gcn_vocab_map.get('UNK', -1))  # 'UNK' can be replaced with a suitable default value
    # Build BERT input tokens, including [CLS] and [SEP]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
    segment_ids = [0] * len(tokens)

    # Convert tokens into input_ids required by BERT
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Create and return the InputFeatures object
    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )
    return feat

class CorpusDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        gcn_vocab_map,
        max_seq_len,
        gcn_embedding_dim,
        set_bank=None,        # np.ndarray [num_accounts, Lmax, d_in] hoặc None
        set_mask_bank=None,   # np.ndarray [num_accounts, Lmax] hoặc None
        set_meta=None,        # dict: {"Lmax": int, "d_in": int} hoặc None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

        # === NEW: cache cho nhánh SetTransformer ===
        self.set_bank = set_bank              # có thể là None (chạy y như cũ)
        self.set_mask_bank = set_mask_bank    # có thể là None
        self.set_Lmax = set_meta["Lmax"] if (set_meta and "Lmax" in set_meta) else None
        self.set_d_in = set_meta["d_in"] if (set_meta and "d_in" in set_meta) else None


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        # === NEW: xác định account_idx và rút set_feats/mask nếu có ===
        set_feats_i = None
        set_mask_i = None

        if (self.set_bank is not None) and (self.set_mask_bank is not None):
            # heuristic: dùng token đầu tiên xuất hiện trong gcn_vocab_map làm đại diện account
            tokens_a = self.examples[idx].text_a.split()
            account_idx = None
            for w in tokens_a:
                if w in self.gcn_vocab_map:
                    account_idx = self.gcn_vocab_map[w]
                    break

            if account_idx is not None and account_idx >= 0 and account_idx < self.set_bank.shape[0]:
                # lấy trực tiếp từ bank đã pad/truncate sẵn
                set_feats_i = self.set_bank[account_idx]      # [Lmax, d_in]
                set_mask_i  = self.set_mask_bank[account_idx] # [Lmax]
            else:
                # nếu không tìm được, trả zero-pad để giữ shape đúng
                if (self.set_Lmax is not None) and (self.set_d_in is not None):
                    set_feats_i = np.zeros((self.set_Lmax, self.set_d_in), dtype=np.float32)
                    set_mask_i  = np.zeros((self.set_Lmax,), dtype=bool)
                else:
                    set_feats_i, set_mask_i = None, None

        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
            set_feats_i,   # NEW: có thể None
            set_mask_i,    # NEW: có thể None
        )


    # @classmethod
    def pad(self, batch):
        gcn_vocab_size = len(self.gcn_vocab_map)
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch
        ]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)
        batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
            batch_gcn_vocab_ids_paded
        ][:, :-1]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        # ==== phần mới: ghép set_feats / set_mask nếu có ====
        # Lưu ý: __getitem__ đã trả thêm 2 phần tử ở vị trí 6,7 (có thể None)
        set_feats_list = [sample[6] if len(sample) > 6 else None for sample in batch]
        set_mask_list  = [sample[7] if len(sample) > 7 else None for sample in batch]

        # Nếu toàn batch đều None -> trả về 6 phần tử như cũ
        if all(x is None for x in set_feats_list) or all(x is None for x in set_mask_list):
            return (
                batch_input_ids,
                batch_input_mask,
                batch_segment_ids,
                batch_confidences,
                batch_label_ids,
                batch_gcn_swop_eye,
            )

        # Xác định Lmax, d_in để pad những phần tử None (ưu tiên lấy từ self.set_meta)
        Lmax = self.set_Lmax
        d_in = self.set_d_in
        if (Lmax is None) or (d_in is None):
            # Suy ra từ phần tử đầu tiên khác None
            for sf in set_feats_list:
                if sf is not None:
                    Lmax = sf.shape[0]
                    d_in = sf.shape[1]
                    break
            assert (Lmax is not None) and (d_in is not None), "Không xác định được Lmax/d_in cho set_feats"

        feats_np = []
        mask_np  = []
        for sf, sm in zip(set_feats_list, set_mask_list):
            if sf is None or sm is None:
                feats_np.append(np.zeros((Lmax, d_in), dtype=np.float32))
                mask_np.append(np.zeros((Lmax,), dtype=bool))
            else:
                # Bảo đảm dtype/shape đúng
                if sf.shape[0] != Lmax or sf.shape[1] != d_in:
                    # nếu vì lý do nào đó chưa pad chuẩn từ trước, ta sẽ truncate/pad ở đây
                    if sf.shape[0] >= Lmax:
                        sf = sf[:Lmax]
                        sm = sm[:Lmax]
                    else:
                        pad_len = Lmax - sf.shape[0]
                        sf = np.concatenate([sf, np.zeros((pad_len, d_in), dtype=sf.dtype)], axis=0)
                        sm = np.concatenate([sm, np.zeros((pad_len,), dtype=sm.dtype)], axis=0)
                feats_np.append(sf.astype(np.float32, copy=False))
                mask_np.append(sm.astype(bool, copy=False))

        batch_set_feats = torch.from_numpy(np.stack(feats_np, axis=0)).float()  # [B, Lmax, d_in]
        batch_set_mask  = torch.from_numpy(np.stack(mask_np, axis=0)).bool()   # [B, Lmax]

        # Trả về 8 phần tử khi có set_feats/set_mask
        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
            batch_set_feats,   # NEW
            batch_set_mask,    # NEW
        )
