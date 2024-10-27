import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF
from torch import Tensor
from typing import List, Tuple


class BiRnnCrf(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_rnn_layers: int = 1,
        rnn: str = "lstm",
    ) -> None:
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.crf = CRF(hidden_dim, self.tagset_size)

    def build_features(self, sentences: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param
            sentences: Tensor[batch, max_sequence_length]
        Output:
            (lstm_out, masks) : Tensor[batch, sequence_length, hidden_dim],
                                Tensor[batch, max_sequence_length]
        """
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(
            embeds, lengths=sorted_seq_length, batch_first=True
        )
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs: Tensor, tags: Tensor) -> Tensor:
        """
        :param xs: Tensor[batch, max_sequence_length]
        :param tags: Tensor[batch, max_sequence_length]
        :return loss: Tensor[0]
        """
        features, masks = self.build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs: Tensor) -> Tuple[Tensor, List[List[int]]]:
        """
        :param xs: Tensor[batch, sequence_length]
        :return tuple[scores, tag_seq]: Tuple[Tensor[batch], List[List[int]]]
        """
        # Get the emission scores from the BiLSTM
        features, masks = self.build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
