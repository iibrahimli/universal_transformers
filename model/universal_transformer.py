import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, embedding_method=None):
        super().__init__()
        if embedding_method == "linear":
            self.embedding = nn.Linear(vocab_size, emb_size)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        if isinstance(self.embedding, nn.Linear):
            tokens = tokens.unsqueeze(dim=-1).float()
            return self.embedding(tokens)
        else:
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalTimestepEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        # generate position encoding as in [Vaswani et al., 2017]
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, time_step: int) -> Tensor:
        """
        Args:
            x: Has shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        x = x + self.pe[:, time_step]
        return self.dropout(x)


class UniversalTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_feedforward: int,
        max_seq_len: int,
        max_time_step: int,
        halting_thresh: float,
        embedding_method: str = None,
        target_input_size: int = None
    ):
        """
        Universal Transformer for sequence to sequence tasks.

        Args:
            source_vocab_size: Number of tokens in the source vocabulary.
            target_vocab_size: Number of tokens in the target vocabulary.
            d_model: Dimensionality of the model.
            n_heads: Number of attention heads.
            d_feedforward: Dimensionality of the feedforward layer.
            max_seq_len: Maximum length of the sequence.
            max_time_step: Maximum time step of the model.
            halting_thresh: Threshold for halting.
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_feedforward, activation=nn.GELU(), batch_first=True
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_feedforward, activation=nn.GELU(), batch_first=True
        )
        self.halting_layer = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.pos_encoder = PositionalTimestepEncoding(d_model, max_len=max_seq_len)
        target_input_size = target_input_size or target_vocab_size
        # token embeddings
        self.source_tok_emb = TokenEmbedding(source_vocab_size, d_model, embedding_method)
        self.target_tok_emb = TokenEmbedding(target_input_size, d_model, embedding_method)

        # final output generating layer
        self.generator = nn.Linear(d_model, target_vocab_size)

        # constants
        self.max_seq_len = max_seq_len
        self.max_time_step = max_time_step
        self.halting_thresh = halting_thresh
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_padding_mask: Tensor = None,
        target_padding_mask: Tensor = None,
    ) -> Tensor:
        """
        Perform forward pass.

        Args:
            source: Source tokens (integers) of shape [batch_size, src_seq_len]
            target: Target tokens of shape [batch_size, tgt_seq_len]
            source_padding_mask: Mask of shape [batch_size, src_seq_len]
            target_padding_mask: Mask of shape [batch_size, tgt_seq_len]

        Returns:
            Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # embed source and target tokens
        source = self.source_tok_emb(source)
        target = self.target_tok_emb(target)

        # get autoregressive mask (no future lookahead)
        target_mask = self.generate_subsequent_mask(target)

        # run encoder and decoder
        memory, ponder_time = self.forward_encoder(source, source_padding_mask)
        output = self.forward_decoder(
            memory, target, target_mask, source_padding_mask, target_padding_mask
        )
        output = self.generator(output)
        return output, ponder_time

    def forward_encoder(self, src: Tensor, src_padding_mask: Tensor = None):
        """
        Perform forward pass of the encoder.

        Args:
            source: Tensor of shape [batch_size, src_seq_len, embedding_dim]
            source_padding_mask: Mask of shape [batch_size, src_seq_len]

        Returns:
            Has shape [batch_size, src_seq_len, embedding_dim]
        """
        halting_probability = torch.zeros((*src.shape[:-1], 1), device=src.device)
        remainders = torch.zeros_like(halting_probability)
        n_updates = torch.zeros_like(halting_probability)
        ponder_time = torch.zeros_like(halting_probability)
        new_src = src.clone()
        """
        Perform forward pass of the encoder.
        Args:

            source_padding_mask: Mask of shape [batch_size, src_seq_len]
            source: Tensor of shape [batch_size, src_seq_len, embedding_dim]

        Returns:
            Has shape [batch_size, src_seq_len, embedding_dim]
        """
        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh
            p = self.halting_layer(new_src)
            new_halted = (halting_probability + p * still_running) > self.halting_thresh
            ponder_time[~new_halted] += 1
            still_running = (
                halting_probability + p * still_running
            ) <= self.halting_thresh
            halting_probability += p * still_running
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders
            n_updates += still_running + new_halted
            update_weights = p * still_running + new_halted * remainders
            new_src = self.pos_encoder(src, time_step)
            new_src = self.encoder_layer(new_src, src_key_padding_mask=src_padding_mask)
            src = (new_src * update_weights) + (src * (1 - update_weights))
        return src, ponder_time

    def forward_decoder(
        self,
        memory: Tensor,
        target: Tensor,
        target_mask: Tensor = None,
        memory_padding_mask: Tensor = None,
        target_padding_mask: Tensor = None,
    ):
        """
        Perform forward pass of the decoder.

        Args:
            memory: Has shape [batch_size, src_seq_len, embedding_dim]
            target: Has shape [batch_size, tgt_seq_len]
            target_mask: Has shape [tgt_seq_len, tgt_seq_len]
            memory_padding_mask: Has shape [batch_size, src_seq_len, embedding_dim]
            target_padding_mask: Has shape [batch_size, tgt_seq_len]

        Returns:
            Has shape [batch_size, tgt_seq_len, embedding_dim]
        """
        halting_probability = torch.zeros((*target.shape[:-1], 1), device=target.device)
        remainders = torch.zeros_like(halting_probability)
        n_updates = torch.zeros_like(halting_probability)
        new_target = target.clone()
        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh
            p = self.halting_layer(new_target)
            new_halted = (halting_probability + p * still_running) > self.halting_thresh
            still_running = (
                halting_probability + p * still_running
            ) <= self.halting_thresh
            halting_probability += p * still_running
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders
            n_updates += still_running + new_halted
            update_weights = p * still_running + new_halted * remainders
            new_target = self.pos_encoder(target, time_step)
            new_target = self.decoder_layer(
                new_target,
                memory,
                tgt_mask=target_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )
            target = (new_target * update_weights) + (target * (1 - update_weights))
        return target

    def generate(
        self,
        source: Tensor,
        eos_token_id: int,
        min_length: int = 2,
        max_length: int = 100,
    ):
        """
        Autoregressively generate output sequence.

        TODO check correctness (huggingface)
        TODO add generation methods: beam search, top-k sampling, etc.

        Returns:
            Sequence of generated tokens of shape [batch_size, seq_len]
        """

        max_length = min(max_length, self.max_seq_len)

        # embed source tokens
        source = self.source_tok_emb(source)

        # run encoder
        memory, ponder_time = self.forward_encoder(source)

        # start from EOS token, append last generated token to the input to
        # the decoder and generate until EOS token
        generated = torch.tensor(
            [[eos_token_id]], device=memory.device, dtype=torch.long
        )
        cur_length = 1
        while cur_length < max_length:
            target = self.target_tok_emb(generated)
            target_mask = self.generate_subsequent_mask(target)
            output = self.forward_decoder(memory, target, target_mask)
            output = self.generator(output)
            output = output.argmax(-1)
            new_token = output[0, -1:].unsqueeze(0)
            generated = torch.cat([generated, new_token], dim=1)
            cur_length += 1
            if (
                generated.squeeze()[-1].item() == eos_token_id
                and cur_length >= min_length
            ):
                break
        return generated

    def generate_algorithmic(
        self,
        source: Tensor,
        source_pad_mask: Tensor
    ):
        """
        Autoregressively generate output sequence for algorithmic tasks.

        TODO check correctness (huggingface)
        TODO add generation methods: beam search, top-k sampling, etc.

        Returns:
            Sequence of generated tokens of shape [batch_size, seq_len]
        """

        # pad zero to fit training padding shape
        out_shape = (source.shape[0], source.shape[1]+1, self.target_vocab_size)
        out = torch.zeros(out_shape, device=source.device)

        # embed source and output
        out_emb = self.target_tok_emb(torch.zeros(out_shape[:2], device=source.device))
        source = self.source_tok_emb(source)

        out_mask = self.generate_subsequent_mask_algo(out_emb)

        # run encoder
        source, ponder_time = self.forward_encoder(source, source_pad_mask)

        # start from pad token, append last generated token to the input to
        # the decoder and generate until EOS token
        for i in range(out.shape[1]):
            output = self.forward_decoder(source, out_emb[:, :i+1],
                                          target_padding_mask=out_mask[i, :, :i+1])
            # i + 1 to skip padding token
            output = self.generator(output[:, i])
            out[:, i] = output
            out_emb[:, i] = self.source_tok_emb(output.argmax(-1))
        #return unpadded output
        return out[:, 1:], ponder_time

    @staticmethod
    def generate_subsequent_mask(target):
        """
        Generate autoregression mask that prevents attending to future positions.

        Args:
            target: Has shape [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        Returns:
            Tensor with shape [seq_len, seq_len]
        """
        sz = target.size(1) if target.dim() == 3 else target.size(0)
        target_mask = (
            torch.triu(torch.ones((sz, sz), device=target.device)) == 1
        ).transpose(0, 1)
        target_mask = (
            target_mask.float()
            .masked_fill(target_mask == 0, float("-inf"))
            .masked_fill(target_mask == 1, float(0.0))
        )
        return target_mask

    @staticmethod
    def generate_subsequent_mask_algo(target):
        """
        Generate autoregression mask that prevents attending to future positions.

        Args:
            target: Has shape [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        Returns:
            Tensor with shape [seq_len, seq_len]
        """
        sz = target.size(1)
        target_mask = (
                torch.triu(torch.ones((sz, sz), device=target.device)) == 1
        ).transpose(0, 1)
        target_mask = (
            target_mask.float()
                .masked_fill(target_mask == 0, float("-inf"))
                .masked_fill(target_mask == 1, float(0.0))
        )
        target_mask = target_mask.unsqueeze(dim=1)
        target_mask = target_mask.repeat(1,target.size(0),1)
        return target_mask
