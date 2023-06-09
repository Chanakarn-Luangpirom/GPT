import torch
import torch.nn as nn
from torch.nn import functional as F


### Params
batch_size = 32
context_length = 8
train_iters = 3000
emb_dim = 32
estimate_loss_interval = 1000
estimate_loss_iterations = 200
lr = 1e-3
num_heads = 6
num_blocks = 3
dropout_pct = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'




with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
print('Length of Text:',len(text))


characters = sorted(list(set(text)))
vocab_size = len(characters)
print('Total Characters:', len(characters))

encoder = {}
decoder = {}
for idx,char in enumerate(characters):
    encoder[char] = idx
    decoder[idx] = char

def encode(text):
    output = []
    for char in text:
        output.append(encoder[char])
    return output
def decode(indices):
    output = []
    for idx in indices:
        output.append(decoder[idx])
    return ''.join(output)

print('The encoded text is :', encode('test encode'))
print('Decoding back:', decode(encode('test encode')))


## Encode the whole dataset
data = torch.tensor(encode(text),dtype = torch.int64)
print('Length after encoding:',len(data)) ## Should be equal to length of text

## Train/Validation Split
split_size = 0.9
n = int(split_size*len(data))
train_data = data[0:n]
val_data = data[n:]

print('size of train:',len(train_data))
print('size of val:',len(val_data))


### Single Data Loader
# x = train_data[:context_length]
# y = train_data[1:context_length+1]

# for i in range(context_length):
#     context = x[:i+1]
#     target = y[i]
#     print('input:', context)
#     print('target:', target)
#     print('--------')


### Batch Data Loader
torch.manual_seed(123)
def generate_batch(split = 'train'):
    if split == 'train':
        data = train_data
    else:
        data = val_data

    start_idxs = torch.randint(len(data)-context_length, (batch_size,))  # Random int with shape

    x_batch = []
    y_batch = []
    for start_idx in start_idxs:
        x = data[start_idx:start_idx+context_length]
        y = data[start_idx+1:start_idx+context_length+1]

        x_batch.append(x)
        y_batch.append(y)
    x_batch, y_batch = torch.stack(x_batch).to(device),torch.stack(y_batch).to(device)

    return x_batch,y_batch

x_batch,y_batch = generate_batch() ## batch_size x context_length
print('Sample of x_batch:', x_batch[0]) 
print('Sample of y_batch:', y_batch[0]) 

print('------------------------')
print('Example of 1 batch input')
for b in range(batch_size):
    for i in range(context_length):
        context = x_batch[b,:i+1]
        target = y_batch[b,i]
        if b==0:
            print('input:', context, ' target:',target)
print('------------------------')

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias = False)
        self.query = nn.Linear(emb_dim, head_size, bias = False)
        self.value = nn.Linear(emb_dim,head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))  ## Register Buffer is not trained by the optimizer
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self,x):
        #(B, context_length, head_size)
        B, T, hs = x.shape
        k = self.key(x)    
        q = self.query(x)
        v = self.value(x)

        head_size = k.shape[-1]

        weights = q @ k.transpose(-2,-1)
        weights = weights * head_size**-0.5
        weights = weights.masked_fill(self.tril[:T,:T]==0, float('-inf'))  # Using :T because input context_size is smaller than the context size used in the model (T <= context_length)
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        output = weights @ v # (B,context,context) @ (B, context, head) --> (B, context, head)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        heads = []
        for _ in range(num_heads):
            head = Head(head_size)
            heads.append(head)

        self.heads = nn.ModuleList(heads) ## Stores multiple head
        self.proj = nn.Linear(head_size*num_heads, emb_dim)
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self, x):
        output = []
        for h in self.heads:
            head_output = h(x)
            output.append(head_output)
        
        output = torch.cat(output,dim = -1)
        output = self.proj(output) # (B,context, emb_dim)
        output = self.dropout(output)
        return output ##
    

class FeedForward(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout_pct)
            )

    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_head):
        super().__init__()
        head_size = emb_dim//n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(emb_dim)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

    def forward(self,x):

        ## Skip connection
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))  
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,emb_dim) ## vocab_size x embedding dimension
        self.position_embedding_table = nn.Embedding(context_length, emb_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(emb_dim, n_head = num_heads) for _ in range(num_blocks)])  ## num_blocks = Number of Transformer Block
        self.layer_norm_output = nn.LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim,vocab_size) #emb_dim x vocab_size


    def forward(self,contexts,targets = None):
        B, L  = contexts.shape 
        token_emb = self.embedding_table(contexts) ## batch_size x context_length x embedding dimension
        position_emb = self.position_embedding_table(torch.arange(L,device = device)) ## context_length x embbedding dimension
        x = token_emb + position_emb
        x = self.transformer_blocks(x) ## (B,context,emb_dim)
        x = self.layer_norm_output(x)
        logits = self.output_head(x) ## batch_size x context_length x vocab_size
        if targets is None:  ## Use for generating
            loss = None
        else:
            B,N,D = logits.shape
            logits = logits.view(B*N,D)
            targets = targets.view(B*N)  
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,contexts,max_tokens):
        ## Input contexts: B x (Input context size)
        for i in range(max_tokens):
            contexts_cropped = contexts[:, -context_length:]  # Crop the context to context_length size
            logits, loss = self.forward(contexts_cropped)
            logits = logits[:,-1,:] # Take only the last element to generate new text --> Now logits is batch_size x embedding_dimision
            probas = F.softmax(logits,dim = 1)
            next_char = torch.multinomial(probas,num_samples = 1) ## sample next char from a probability distribution
            contexts = torch.cat((contexts,next_char),dim = 1)  ## Append the generated text to the sequence
        return contexts


    

model = GPTLanguageModel(vocab_size)
logits, loss = model(x_batch,y_batch)
print('logits shape:',logits.shape)  #(Batch_size x context_length, Total Characters)
print('The current loss: ',loss)

# print(decoder)
# test_context = torch.zeros((1,1),dtype = torch.int64) ## Recall that zero is new line. --> Giving a new line as context
# print('Generate Text using test context: ')
# print(decode(model.generate(test_context, max_tokens = 100)[0].tolist()))

@torch.no_grad()
def estimate_loss():
    ### To do: Estimate Loss of Train and Validation Set Over estimate_loss_iterations
    model.eval()
    training_loss = 0
    validation_loss = 0

    #Estimate Train Loss
    for i in range(estimate_loss_iterations):
        x_batch,y_batch = generate_batch('train')
        logits, loss = model.forward(x_batch,y_batch)
        training_loss += loss

    #Estimate Val Loss
    for i in range(estimate_loss_iterations):
        x_batch,y_batch = generate_batch('val')
        logits, loss = model.forward(x_batch,y_batch)
        validation_loss += loss

    model.train() #Back to training mode
    return training_loss/estimate_loss_iterations, validation_loss/estimate_loss_iterations


## Retest the Model After Training
optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
for step in range(train_iters):
    x_batch, y_batch = generate_batch('train')

    if step%estimate_loss_interval==0:
        print('Currently at step:',step)
        training_loss,validation_loss = estimate_loss()
        print('Training Loss: ',training_loss.item())
        print('Validation Loss:',validation_loss.item())

    #Evaluate Loss and Optimize
    logits, loss = model(x_batch,y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

test_context = torch.zeros((1,1),dtype = torch.int64) 
print('Generate Text using test context: ')
print(decode(model.generate(test_context, max_tokens = 300)[0].tolist()))





    