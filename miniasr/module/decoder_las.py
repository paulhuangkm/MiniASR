import torch
from torch import batch_norm, nn
from torch.autograd import Variable

def CreateOnehotVariable(input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    # print(onehot_x)
    
    return onehot_x

class Attention(nn.Module):
    '''
        Implementation of https://aclanthology.org/D15-1166.pdf
    '''
    def __init__(self, input_dim, n_head, mode='dot'):
        super().__init__()
        self.input_dim = input_dim
        self.n_head = n_head
        self.mode = mode
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, encoder_out, decoder_out):
        if self.mode == 'dot':
            if self.n_head == 1:
                align_vec = torch.bmm(decoder_out, encoder_out.transpose(1, 2)).squeeze(dim=1)
                attention_score = self.softmax(align_vec)
                contextualized_vec = torch.sum(encoder_out * attention_score.unsqueeze(2).repeat(1, 1, encoder_out.size(2)), dim=1)
                del align_vec, attention_score
            else:
                # multihead attention
                raise NotImplementedError
        else:
            # TODO: other type of attention
            raise NotImplementedError
        
        return contextualized_vec
        

class RNNDecoder(nn.Module):
    '''
        For LAS.
        RNN-based decoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        module [str]: RNN model type
        dropout [float]: dropout rate
        bidirectional [bool]: bidirectional encoding
    '''

    def __init__(self, in_dim, num_class, hid_dim, n_layers, module='LSTM',
                 dropout=0, bidirectional=True):
        super().__init__()

        # RNN model
        self.rnn = getattr(nn, module)(
            input_size=in_dim + num_class,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # RNN Output dimension
        # Bidirectional makes output size * 2
        self.rnn_out_dim = hid_dim * (2 if bidirectional else 1)
        
        self.attention = Attention(
            input_dim=self.rnn_out_dim,
            n_head=1,
        )
        
        self.output_dim = num_class
        
        self.project_layer = nn.Sequential(
            # Since input is concated by rnn_out & contextualized vector obtained from attention
            nn.Linear(2 * self.rnn_out_dim, self.output_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, feat, text, text_len):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''
        # print(f"text.shape: {text.shape}")
        batch_size, n_class = feat.shape[0], self.output_dim
        output_word = CreateOnehotVariable(torch.zeros(batch_size, 1), n_class).cuda()
        rnn_input = torch.cat([output_word, feat[:, 0:1, :]], dim=-1)
        del output_word
        # iterate with teacher forcing
        preds = []
        for step in range(text.shape[1]):
            rnn_out, _ = self.rnn(rnn_input)
            # Attention of encoder output features & current rnn output
            context = self.attention(feat, rnn_out)
            # Project to dim=num_class, than pass softmax
            pred = self.project_layer(torch.cat([rnn_out.squeeze(dim=1), context], dim=-1))
            preds.append(pred.unsqueeze(-1)) # each prediction is be of shape (batch_size, num_class, 1)
            del rnn_out, pred
            # ground_truth = text[:, step : step + 1] # shape: (batch_size, 1)
            ground_truth = CreateOnehotVariable(text[:, step : step + 1], n_class).cuda()
            rnn_input = torch.cat([ground_truth, context.unsqueeze(1)], dim=-1)
            del ground_truth, context
            
        preds = torch.cat(preds, dim=-1)
        
        return preds