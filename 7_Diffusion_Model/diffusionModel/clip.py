import torch
import torch.nn as nn
from torch.nn import functional as Fn
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    '''
        Generally positional embedding are sinosidal (Sine and Cosine) but 
        as we are using pretrained weights the positional embedding are encoded 
        in it so we will directly use them.
    '''
    def __init__(self, vocabSize:int, n_embed:int, n_tokens:int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabSize, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(vocabSize, n_embed))

    
    def forward(self, tokens) :
        '''
            Normally we add positional input embed to positional embed, 
            similarily we will add the embed to positional embed
        '''
        # batch_size, seq_Len -> batch_size, seq_len, Dim
        x = self.token_embedding(tokens)

        x += self.positional_embedding

        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head:int, n_embed:int):
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(heads=n_head, d_embed=n_embed)
        self.layerNorm2 = nn.LayerNorm(n_embed)
        self.linear1 = nn.Linear(in_features=n_embed, out_features=4 * n_embed)
        self.linear2 = nn.Linear(in_features=4 * n_embed, out_features=n_embed)

    def forward(self, x):
        '''
            Attention Mechanism (of Transformer)
        '''
        residue = x

        # layer Norm
        x = self.layerNorm1(x)
        # self attention
        x = self.attention(x)

        # residue
        x += residue

        residue = x # store again for next connection

        x = self.layerNorm2(x)

        x = self.linear1(x)

        # Quick GELU Activation
        x = x * torch.sigmoid(1.702 * x )

        x = self.linear2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


        


class CLIP(nn.Module):

    def __init__(self, ):
        super().__init__()
                                    # vocabSize, embeddingDim, sequenceLength
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layerNorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # batchSIze, seqLen -> Batch_size, seq_len, Dim
        state = self.embedding

        for layer in self.layers:
            state = layer(state)

        # Batch_size, seq_len, Dim
        output = self.layerNorm(state)

        return output


