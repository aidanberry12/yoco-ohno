import math
import torch
from torch import nn, Tensor

# Code adopted from pytorch docs: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class TransformerModel(nn.Module):

    def __init__(self, ingr_size,instr_size,d_model,nhead,
                         num_encoder_layers,num_decoder_layers,
                         dim_feedforward,dropout,max_ingr,max_instr,device):
        super().__init__()
        self.device=device
        print('Defining Model Params')
        self.model_type = 'Transformer'
        self.d_model=d_model
        print('Defining Pos Encodings')
        self.ingr_pos_embedding = PositionalEncoding(d_model, dropout,max_ingr,device)
        self.instr_pos_embedding = PositionalEncoding(d_model, dropout,max_instr,device)
        print('Defining Ingredient Embeddings')
        self.ingr_embedding = nn.Embedding(ingr_size, d_model).to(device)
        print('Defining Instruction Embeddings')
        self.instr_embedding = nn.Embedding(instr_size, d_model).to(device)
        print('Creating Transformer Module')
        self.transformer=nn.Transformer(d_model=d_model,nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dropout=dropout).to(device)
        print('Creating Final Linear Layer')
        self.final_linear=nn.Linear(self.d_model,instr_size).to(device)
        self.init_weights()
        self.max_instr=max_instr

    def init_weights(self) -> None:
        initrange = 0.1
        print('Initializing Weights')
        self.ingr_embedding.weight.data.uniform_(-initrange, initrange)
        self.instr_embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, tgt) -> Tensor:
        src = self.ingr_embedding(src) * math.sqrt(self.d_model)
        src = self.ingr_pos_embedding(src)
        tgt = self.instr_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.instr_pos_embedding(tgt)
        
        # The target (instructions) are not allowed to attend to the works ahead of it. So we add a tgt_mask)
        # The source (ingredients) don't require a mask
        output = self.transformer(src,tgt ,tgt_mask=generate_square_subsequent_mask(tgt.size()[0],self.device))
        output=self.final_linear(output)
        return output
    
    def get_inference(self,src,start_token,end_token):
        tgt=torch.ones(1,src.size()[1])*start_token
        tgt=tgt.to(torch.int64)
        next_item=start_token
        for i in range(self.max_instr):
            pred = self.forward(src, tgt)
            next_item = torch.argmax(pred,dim=2)[-1]
            tgt = torch.cat((tgt, next_item[None,:]), dim=0)
    

        return tgt
    
class TransformerDecoderModel(nn.Module):

    def __init__(self, ingr_size,instr_size,d_model,nhead,
                         num_encoder_layers,num_decoder_layers,
                         dim_feedforward,dropout,max_ingr,max_instr,device):
        super().__init__()
        self.device=device
        print('using decoder')
        print('Defining Model Params')
        self.model_type = 'Transformer'
        self.d_model=d_model
        print('Defining Pos Encodings')
        # self.ingr_pos_embedding = PositionalEncoding(d_model, dropout,max_ingr,device)
        self.instr_pos_embedding = PositionalEncoding(d_model, dropout,max_instr,device)
        print('Defining Ingredient Embeddings')
        self.ingr_embedding = nn.Embedding(ingr_size, d_model).to(device)
        print('Defining Instruction Embeddings')
        self.instr_embedding = nn.Embedding(instr_size, d_model).to(device)
        print('Creating Transformer Module')
        decoder_layer=nn.TransformerDecoderLayer(d_model=d_model,nhead=nhead,dropout=dropout).to(device)
        self.transformer=nn.TransformerDecoder(decoder_layer,num_decoder_layers).to(device)
        # self.transformer=nn.Transformer(d_model=d_model,nhead=nhead,
        #                                 num_encoder_layers=num_encoder_layers,
        #                                 num_decoder_layers=num_decoder_layers,
        #                                 dropout=dropout).to(device)
        print('Creating Final Linear Layer')
        self.final_linear=nn.Linear(self.d_model,instr_size).to(device)
        self.init_weights()
        self.max_instr=max_instr

    def init_weights(self) -> None:
        initrange = 0.1
        print('Initializing Weights')
        self.ingr_embedding.weight.data.uniform_(-initrange, initrange)
        self.instr_embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, tgt) -> Tensor:
        src = self.ingr_embedding(src)
        
        tgt = self.instr_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.instr_pos_embedding(tgt)
        
        # The target (instructions) are not allowed to attend to the works ahead of it. So we add a tgt_mask)
        # The source (ingredients) don't require a mask
        output = self.transformer(tgt,src,tgt_mask=generate_square_subsequent_mask(tgt.size()[0],self.device))
        output=self.final_linear(output)
        return output
    
    def get_inference(self,src,start_token,end_token):
        tgt=torch.ones(1,src.size()[1])*start_token
        tgt=tgt.to(torch.int64)
        next_item=start_token
        for i in range(self.max_instr):
            pred = self.forward(src, tgt)
            next_item = torch.argmax(pred,dim=2)[-1]
            tgt = torch.cat((tgt, next_item[None,:]), dim=0)
    

        return tgt


def generate_square_subsequent_mask(sz,device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)




class PositionalEncoding(nn.Module):

    def __init__(self, d_model,dropout,max_len,device):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe=pe.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)