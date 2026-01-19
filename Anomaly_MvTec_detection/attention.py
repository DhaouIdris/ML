import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module): #Base class for every NN in pytorch

    def __init__(self, d_model=2,
                 row_dim=0,
                 col_dim=1):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_model,
                                bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_model,
                                bias=False)
        def forward(self, token_encodings):
            q = self.W_q(token_encodings)
            k = self.W_k(token_encodings)
            v = self.W_v(token_encodings)

            sims = torch.matmul(q, k.transpose(dim0=self.row_dim,
                                               dim1=self.col_dim))
            scaled_sims = sims / torch.sqrt(torch.tensor(q.size(self.col_dim)**0.5))
            attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
            attention_output = torch.matmul(attention_percents, v)


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(in_features=self.d_model,
                             out_features=d_model,
                             bias = False)
        self.W_k = nn.Linear(in_features=self.d_model,
                             out_features=d_model,
                             bias = False)
        self.W_v = nn.Linear(in_features=self.d_model,
                             out_features=d_model,
                             bias = False)
    def forward(self, tokens, mask=None):
        q = self.W_q(tokens)
        k = self.W_k(tokens)
        v = self.W_v(tokens)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim,
                                           dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask,
                                                  value =1e9)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention = torch.matmul(attention_percents, v)

        return attention
    

encoding_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.42, -2.16]])

torch.manual_seed(42)

maskedSelfAttention = MaskedSelfAttention(d_model=2)

mask = torch.tril(torch.ones(3,3))
mask = mask == 0

maskedSelfAttention(encoding_matrix, mask=mask)


class Attention(nn.Module):
    def __init__(self, d_model=2,
                 row_dim=0,
                 col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_model,
                                bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_model,
                                bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_model, 
                                bias=False)
        
    def forward(self, encodings_for_q,
                      encodings_for_k,
                      encodings_for_v,
                      mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim,
                                           dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask,
                                                  value=1e9)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_output = torch.matmul(attention_percents, v)

        return attention_output
    

encoding_matrix_q = torch.tensor([[1.16, 0.23],
                                  [0.57, 1.36],
                                    [4.41, -2.16]])

encoding_matrix_k = encoding_matrix
encoding_matrix_v = encoding_matrix

attention = Attention(d_model=2, row_dim=0, col_dim=1)

attention(encoding_matrix_q,
          encoding_matrix_k,
        encoding_matrix_v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2,
                 num_heads=2,
                 row_dim=0,
                 col_dim=1):
        super().__init__()

        self.heads = nn.ModuleList([Attention(d_model=d_model,
                                              row_dim=row_dim, col_dim=col_dim)
                                              for _ in range(num_heads)])
        self.col_dim = col_dim

    def forward(self, encodings_for_q,
                      encodings_for_k,
                      encodings_for_v,
                      mask=None):


        return torch.cat([head(encodings_for_q,
                              encodings_for_k,
                              encodings_for_v,
                              mask=mask) for head in self.heads], dim=self.col_dim)
    

multihead_attention = MultiHeadAttention(d_model=2, num_heads=1, row_dim=0, col_dim=1)

multihead_attention(encoding_matrix_q,
                    encoding_matrix_k,
                    encoding_matrix_v)
