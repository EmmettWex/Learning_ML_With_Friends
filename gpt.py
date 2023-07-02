import torch
import torch.nn as nn
from torch.nn import functional as F

# flash attention
# replace the attention model in karpathy's work with the flash attention model
# 384/6

# hyperparameters
# meta-variables you use to define aspects of your overall architecture, learning rate, batch size, block size, etc.
# Use these to tweak your model to experiment with the best inputs for performance (loss)
batch_size = 10 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?

max_iters = 1000 # training loop, determine how many iterations you want to run here (forward + back passes)
eval_interval = 50
eval_iters = 200

learning_rate = 3e-4 # this is plugged into the optimizer; how big of a gradient descent step do we take?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# higher dimensionality greater increases the ability to clarify which character is which (character being the operative word here)
n_embd = 384 # this is the vocab embedding dimensionality embedding dimensionality going into the "Feedfoward"

n_head = 6 # declares the number of attention layers (heads) we are using prior to probability normalization

n_layer = 6

dropout = 0.2
# ------------


...
#
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, )) 
    x = torch.stack([data[i:i+block_size] for i in ix]) # results in x.shape = tensor(10,64)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # same here, this is the target tensor for training
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # token emits to communcate about itself to other tokens
        self.key = nn.Linear(n_embd, head_size, bias=False)        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)


        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        # note: k, q, and v are all effectively the same at this step, it is how you use these layers that informs the effect they have on the data.
        # it informs the effect back propegation has on THEM
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) # tensor = tensor.permute(0, 2, 1)
        ### This will zero out anything that happened to evaluate to zero in either tensor
        ### if one layer does not care about a specific node, then it will inform the other layer of that.
        # [
        #     [1,2,3],
        #     [4,5,6]
        # ]
        # .transpose @ =
        # [
        #     [1,4],
        #     [2,5],
        #     [3,6]
        # ]
        
        # [
        #     [14, 32],
        #     [32, 77]
        # ] * 1/10
        # =
        # [
        #     [1.4, 3.2],
        #     [3.2, 7.7]
        # ]
        # torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) 
        ###### out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # also works, and is faster :)
        
        # k.shape = tensor(10,32,256)
        # q.shape = tensor(10,32,256)
        # k.q.transpose.shape = tensor(10,32,32) ?
        wei = wei * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # 1/sqrt(60) * every got dang embedding by this, reducing it to around 1/7.75 of it's value
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # projects to mroe complex space?
            nn.ReLU(), # zero's out negative numbers along with applying non-linear computations (tanh, other ones too?)
            nn.Linear(4 * n_embd, n_embd), # compresses back down to n_embd
            nn.Dropout(dropout), # allows us to train the system to recognize an input from multiple vantage points. Zeroes out random numbers within each embedding.
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        
        super().__init__()
        head_size = n_embd // n_head 
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)       
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        #NOTE: Ask algo
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # definitional modality
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # positional modality
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # word_1_embedding = def_mod_1 + pos_mod_1 (2 separate modalities information sums to total of both modalities)
        x = tok_emb + pos_emb # (B,T,C)
        # x = multi-modal embeddings for each individual word, currently exists as the sum of these modalities prior to probability normalization
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C) # final layer normalization
        ## re-shapes our dimensionality to be the vocab size :)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)                                                                                                                                        
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
