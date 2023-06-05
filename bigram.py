import torch

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
print('Length of Text:',len(text))


characters = sorted(list(set(text)))
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

print(encode('test encode'))
print(decode(encode('test encode')))


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
context_length = 8
batch_size = 4

x = train_data[:context_length]
y = train_data[1:context_length+1]

for i in range(context_length):
    context = x[:i+1]
    target = y[i]
    print('input:', context)
    print('target:', target)
    print('--------')



## Batch Data Loader
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
    return torch.stack(x_batch),torch.stack(y_batch)

x_batch,y_batch = generate_batch()
print(x_batch)
print(y_batch)

for b in range(batch_size):
    for i in range(context_length):
        context = x_batch[b,:i+1]
        target = y_batch[b,i]
        print('input:', context)
        print('target:', target)
        print('--------')


    