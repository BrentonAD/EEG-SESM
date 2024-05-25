import torch
import torch.nn as nn

from sesm import (
    MultiHeadSelectiveAttention,
    ConvNormPool,
    Swish,
)

class Embedder(nn.Module):
    def __init__(self, d_input, d_embed, d_kernel):
        super().__init__()
        self.embed = nn.Sequential(
            # Make padding same so sequence length not effected
            nn.Conv1d(d_input, d_embed, d_kernel, padding="same"),
            nn.BatchNorm1d(num_features=d_embed),
            Swish(),
        )

    def forward(self, x):
        x = self.embed(x)
        return x

class Conceptizer(nn.Module):
    def __init__( self, n_heads, d_embed, d_hidden, d_kernel, dropout):
            super().__init__()
            ## sequential selector
            self.multi_head_selective_attention = MultiHeadSelectiveAttention(
                n_heads, d_embed, dropout
            )
            self.encoder = nn.Sequential(
                ConvNormPool(d_embed, d_hidden, d_kernel),
                ConvNormPool(d_hidden, d_hidden, d_kernel),
                nn.AdaptiveMaxPool1d((1)),
            )
            # (Batch, Heads, Hidden)

    def forward(self, x, sent_mask):
        
        ## multi-head selective attention
        selective_actions = self.multi_head_selective_attention(x, sent_mask)
        (Batch, Heads, Seqlen) = selective_actions.shape
        # (Batch, n_heads, Seqlen)
        
        # Apply selective actions to embedded sequence
        x_selective_actions = x.unsqueeze(1) * selective_actions.unsqueeze(-1)
        x_selective_actions = x_selective_actions.reshape(
            Batch * Heads, Seqlen, -1
        ).transpose(-2, -1)
        # (Batch * Heads, Hidden, Seqlen)

        ## Encode resulting tensor
        x_concepts_encoded = self.encoder(
            x_selective_actions
        ).reshape(Batch, Heads, -1)
        # (Batch, Heads, Hidden)
        
        return x_concepts_encoded, selective_actions

class Parameterizer(nn.Module):
    def __init__(self, n_heads, d_embed, d_hidden, d_kernel, dropout):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvNormPool(d_embed, d_hidden, d_kernel),
            ConvNormPool(d_hidden, d_hidden, d_kernel),
            nn.AdaptiveMaxPool1d((1)),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_heads),
            nn.Softplus(),
        )
        # (Batch, Heads)
    
    def forward(self, x):
        
        Batch = x.shape[0]
        
        encoded_sequence = self.encoder(x.transpose(-2, -1)).reshape(Batch, -1)
        relevance_weights = self.proj(encoded_sequence)

        return relevance_weights

class Aggregator(nn.Module):
    def __init__(self, d_hidden, d_out, dropout):
        super().__init__()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            Swish(),
            # Dropout right before this final layer causing consistent classification of class 0 when used in the conceptizer
            # nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
    
    def forward(self, x_concepts_encoded, relevance_weights, ignore_relevance_weights=False):

        (Batch, Heads) = relevance_weights.shape

        out = self.fc(x_concepts_encoded.reshape(Batch * Heads, -1))
        if not ignore_relevance_weights:
            out = (out.reshape(Batch, Heads, -1) * relevance_weights.unsqueeze(-1)).sum(
                    1
                )
        return out

class Model(nn.Module):
    def __init__(
        self,
        n_heads,
        d_input,
        d_embed,
        d_hidden,
        d_kernel,
        d_out,
        dropout=0.2,
        dmin=1,
        **kwargs
    ):
        super().__init__()
        self.d_kernel = d_kernel
        self.dmin = dmin

        # Define the components
        self.embedder = Embedder(d_input, d_embed, d_kernel)
        self.conceptizer = Conceptizer(n_heads, d_embed, d_hidden, d_kernel, dropout)
        self.parameterizer = Parameterizer(n_heads, d_embed, d_hidden, d_kernel, dropout)
        self.aggregator = Aggregator(d_hidden, d_out, dropout)

    def classifier(self, sequence):
        x = self.embedder(sequence)
        # (Batch, Hidden, Seqlen)
        x = self.conceptizer.encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.aggregator.fc(x)

    # @torchsnooper.snoop()
    def forward(self, sequence, sent_mask=None, ignore_relevance_weights=False):
        """
        sequence: (Batch, Seqlen)
        """

        ## Embed sequence
        x = self.embedder(sequence).transpose(-2, -1)
        # (Batch, Seqlen, Embed)

        # Get encoded concepts
        x_concepts_encoded, selective_actions = self.conceptizer(x, sent_mask)
        # (Batch, Heads, Hidden)

        # Get relevance weights
        relevance_weights = self.parameterizer(x)
        # (Batch, Heads)

        ## Aggregate scores
        out = self.aggregator(x_concepts_encoded, relevance_weights, ignore_relevance_weights)
        # (Batch, d_out)

        ## Calculate regularisation constraints
        L_diversity = self._diversity_term(selective_actions, dmin=self.dmin)
        L_stability = self._stability_term(
            x_concepts_encoded, relevance_weights
        )
        L_locality = self._locality_term(selective_actions, sent_mask)
        L_simplicity = self._simplicity_term(relevance_weights)

        return (
            out,
            [L_diversity, L_stability, L_locality, L_simplicity],
            selective_actions,
            relevance_weights,
        )

    def _diversity_term(self, x, d="euclidean", eps=1e-9, dmin=1.0):

        if d == "euclidean":
            # euclidean distance
            D = torch.cdist(x, x, 2)
            Rd = torch.relu(-D + dmin)

            zero_diag = torch.ones_like(Rd, device=Rd.device) - torch.eye(
                x.shape[-2], device=Rd.device
            )
            return ((Rd * zero_diag)).sum() / 2.0

        elif d == "cosine":
            # cosine distance
            x_n = x.norm(dim=-1, keepdim=True)
            x_norm = x / torch.clamp(x_n, min=eps)
            D = 1 - torch.matmul(x_norm, x_norm.transpose(-1, -2))
            zero_diag = torch.ones_like(D, device=D.device) - torch.eye(
                x.shape[-2], device=D.device
            )
            return (D * zero_diag).sum() / 2.0
        
        else:
            raise NotImplementedError

    def _stability_term(self, x_sequential_selected_encoded, concept_selector):

        x = x_sequential_selected_encoded.transpose(0, 1)
        x_n = x.norm(dim=-1, keepdim=True)
        x_norm = x / torch.clamp(x_n, min=1e-9)
        D = 1 - torch.matmul(x_norm, x_norm.transpose(-1, -2))

        mask = concept_selector.transpose(0, 1)  # (Head, Batch)
        mask = mask.unsqueeze(-2) * mask.unsqueeze(-1)  # (Head, Batch, Batch)

        return (D * mask).sum() / 2.0

    def _locality_term(self, multi_head_selective_attention, sent_mask):
        if sent_mask is not None:
            return (multi_head_selective_attention.sum(2) / sent_mask.sum(2)).sum()
        else:
            return (multi_head_selective_attention.mean(2)).sum()

    def _simplicity_term(self, concept_selector):
        # (Batch, n_heads)
        return (concept_selector.sum(1) / concept_selector.shape[1]).sum()
