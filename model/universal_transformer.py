import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import math
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalTimestepEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, time_step: int) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        x = x + self.pe[time_step]
        return self.dropout(x)


class UniversalTransformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head,
                 dimfeedforward, max_len, max_time_step,
                 halting_thresh, transition_function=None):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model, n_head,
                                                     dimfeedforward,
                                                     activation=nn.GELU())
        # if transition_function:
        #     self.encoder_layer._ff_block = nn.Sequential(nn.Conv2d())

        self.decoder_layer = TransformerDecoderLayer(d_model, n_head,
                                                     dimfeedforward,
                                                     activation=nn.GELU())
        self.halting_layer = nn.Sequential(nn.Linear(dimfeedforward, 1),
                                           nn.Sigmoid())
        self.coordinate_embedding = PositionalTimestepEncoding(d_model, max_len=max_len)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model) #TODO (Cati) aufräumen
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model) #TODO (Cati) aufräumen
        self.generator = nn.Linear(dimfeedforward, tgt_vocab_size) #generate output
        self.max_time_step = max_time_step
        self.halting_thresh = halting_thresh

    def forward(self, src: Tensor, target, src_padding_mask=None,
                target_padding_mask=None):
        src = self.src_tok_emb(src) #TODO (Cati) aufräumen
        target = self.tgt_tok_emb(target) #TODO (Cati) aufräumen
        memory = self.forward_encoder(src, src_padding_mask)
        target_mask = self.generate_target_mask(target) #TODO (Cati) aufräumen
        output = self.forward_decoder(memory, target, target_mask,
                                      src_padding_mask, target_padding_mask)
        output = torch.softmax(output, dim=-1)
        output = self.generator(output) #TODO (Cati) aufräumen
        return output

    def forward_encoder(self, src: Tensor, src_padding_mask=None):
        halting_probability = torch.zeros(src.shape[:-1], device=src.device)
        remainders = torch.zeros(src.shape[:-1], device=src.device)
        n_updates = torch.zeros(src.shape[:-1], device=src.device)
        new_src = src.clone()
        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh
            p = self.halting_layer(new_src).squeeze(dim=-1)
            new_halted = (halting_probability + p * still_running) > self.halting_thresh
            still_running = (halting_probability + p * still_running) <= self.halting_thresh
            halting_probability += p * still_running
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders
            n_updates += still_running + new_halted
            update_weights = p * still_running + new_halted * remainders
            new_src = self.coordinate_embedding(src, time_step)
            new_src = self.encoder_layer(new_src, src_key_padding_mask=src_padding_mask)
            src = ((new_src * update_weights) + (src * (1 - update_weights)))
        return src

    def forward_decoder(self, memory: Tensor, target, target_mask=None,
                        memory_padding_mask=None, target_padding_mask=None):
        halting_probability = torch.zeros(target.shape[:-1], device=target.device)
        remainders = torch.zeros(target.shape[:-1], device=target.device)
        n_updates = torch.zeros(target.shape[:-1], device=target.device)
        new_target = target.clone()
        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh
            p = self.halting_layer(new_target).squeeze(dim=-1)
            new_halted = (halting_probability + p * still_running) > self.halting_thresh
            still_running = (halting_probability + p * still_running) <= self.halting_thresh
            halting_probability += p * still_running
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders
            n_updates += still_running + new_halted
            update_weights = p * still_running + new_halted * remainders
            new_target = self.coordinate_embedding(target, time_step)
            new_target = self.decoder_layer(new_target, memory,
                                             tgt_mask=target_mask,
                                             tgt_key_padding_mask=target_padding_mask,
                                             memory_key_padding_mask=memory_padding_mask)
            target = ((new_target * update_weights) + (target * (1 - update_weights)))
        return target


    @staticmethod
    def generate_target_mask(target):
        sz = len(target)
        target_mask = (torch.triu(
            torch.ones((sz, sz), device=target.device)) == 1).transpose(
            0, 1)
        target_mask = target_mask.float().masked_fill(target_mask == 0, float(
            '-inf')).masked_fill(
            target_mask == 1, float(0.0))
        return target_mask
