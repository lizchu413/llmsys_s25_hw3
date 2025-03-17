import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool = False, p_dropout: float = 0.1, bias: bool = True,
                 backend: TensorBackend = None, use_fused_kernel: bool = False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers

        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        # COPY FROM ASSIGN2_4
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        self.use_fused_kernel = use_fused_kernel

    def create_causal_mask(self, bs, nh, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        # mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        # This should be ok, but may be problematic -> the loss will be NaN in Assignment 3 because the mask
        # will not broadcast correctly in the kernel.
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype),
                                                 1)  # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention

        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM
        x_flat = x.view(batch_size * seq_len, n_embd)
        q = self.q_projection(x_flat).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        v = self.v_projection(x_flat).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        kT = self.k_projection(x_flat).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        kT = kT.permute(0, 1, 3, 2)

        return q, kT, v

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p),
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None

        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            if self.causal:
                scores = q @ kT / np.sqrt(self.attn_hidden_dim) + self.create_causal_mask(batch_size, num_head, queries_len)
            else:
                scores = q @ kT / np.sqrt(self.attn_hidden_dim)
            scores = softmax(scores, dim=3)
        else:
            # BEGIN ASSIGN3_3
            scores = q @ kT / np.sqrt(self.attn_hidden_dim)
            if self.causal:
                mask = self.create_causal_mask(batch_size, num_head, queries_len)
            else:
                mask = tensor_from_numpy(np.zeros(batch_size, num_head, queries_len, queries_len), backend=self.backend)
            scores = scores.attn_softmax(mask)
            # END ASSIGN3_3
        result = scores @ v
        result = result.permute(0, 2, 1, 3).contiguous().view(batch_size, queries_len, num_head * q_dim)
        return result

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed.

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # COPY FROM ASSIGN2_4
        q, kT, v = self.project_to_query_key_value(x)
        attn_output = self.self_attention(q, kT, v)
        attn_flat = attn_output.contiguous().view(batch_size * seq_len, n_embd)
        output = self.out_projection(attn_flat)
        output = output.view(batch_size, seq_len, n_embd)
        output = self.dropout(output)
        return output


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int = 256, p_dropout: float = 0.1, bias: bool = True,
                 backend: TensorBackend = None):
        super().__init__()
        """The Feed Forward Module.

        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers

        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        # COPY FROM ASSIGN2_4
        self.linear_in = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM ASSIGN2_4
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x


class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1, ln_eps: float = 1e-8, bias: bool = True,
                 backend: TensorBackend = None, use_fused_kernel: bool = False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers

        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """

        # COPY FROM ASSIGN2_4
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout=p_dropout, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)

        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend, use_fused_kernel=use_fused_kernel)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend, use_fused_kernel=use_fused_kernel)

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape

        # if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
        x_norm1 = self.ln_1(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim)
        attn_output = self.attention(x_norm1)
        x += attn_output
        x_norm2 = self.ln_2(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim)
        ff_output = self.ff(x_norm2)
        x += ff_output
        # else:
        #     # BEGIN ASSIGN3_3
        #     raise NotImplementedError
        #     # END ASSIGN3_3
        return x


class DecoderLM(Module):
    def __init__(
            self,
            n_vocab: int,
            n_embd: int,
            n_head: int,
            n_positions: int,
            p_dropout: float = 0.1,
            ln_eps: float = 1e-5,
            bias: bool = True,
            backend: TensorBackend = None,
            use_fused_kernel: bool = False,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.

        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab

        # COPY FROM ASSIGN2_4
        self.token_embeddings    = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1           = TransformerLayer(n_embd, n_head, p_dropout=p_dropout,
                                                    ln_eps=ln_eps, bias=bias, backend=backend,
                                                    use_fused_kernel=use_fused_kernel)
        self.t_layer_2           = TransformerLayer(n_embd, n_head, p_dropout=p_dropout,
                                                    ln_eps=ln_eps, bias=bias, backend=backend,
                                                    use_fused_kernel=use_fused_kernel)
        self.t_layer_3           = TransformerLayer(n_embd, n_head, p_dropout=p_dropout,
                                                    ln_eps=ln_eps, bias=bias, backend=backend,
                                                    use_fused_kernel=use_fused_kernel)
        self.t_layer_4           = TransformerLayer(n_embd, n_head, p_dropout=p_dropout,
                                                    ln_eps=ln_eps, bias=bias, backend=backend,
                                                    use_fused_kernel=use_fused_kernel)
        self.dropout             = Dropout(p_dropout)
        self.lm_head             = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        self.ln                  = LayerNorm1d(n_embd, eps=ln_eps, backend=backend, use_fused_kernel=use_fused_kernel)

        # self.use_fused_kernel = use_fused_kernel
        # if not self.use_fused_kernel:
        #     # COPY FROM ASSIGN2_4
        #
        # else:
        #     # BEGIN ASSIGN3_3
        #     raise NotImplementedError
        #     # END ASSIGN3_3

    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args:
            idx: input of shape (batch_size, seq_len)

        Returns:
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """

        batch_size, seq_len = idx.shape
        pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        # if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
        token_embeddings = self.token_embeddings(idx)
        position_ids = tensor_from_numpy(np.arange(seq_len).reshape((1, -1)), backend=self.backend)
        position_embeddings = self.position_embeddings(position_ids)
        assert position_embeddings.shape == (1, seq_len, self.n_embd)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        # Pass through each transformer Layer
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        # Final LayerNorm
        x = self.ln(x.view(batch_size * seq_len, self.n_embd))
        x = self.lm_head(x)
        # Get correct shape
        x = x.view(batch_size, seq_len, self.n_vocab)
        # else:
        #     # BEGIN ASSIGN3_3
        #     raise NotImplementedError
        #     # END ASSIGN3_3

        return x
