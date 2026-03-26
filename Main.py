import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
import random
import math
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# ======================
# 1. DATA PREPARATION
# ======================
class TextPreprocessor:
    def __init__(self):
        self.first_words = set()
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()
    
    def process_csv(self, file_path):
        df = pd.read_csv(file_path)
        df['cleaned'] = df['Research Topic'].apply(self.clean_text)
        topics = df['cleaned'].tolist()
        
        # Extract first words
        for topic in topics:
            if topic:
                first_word = topic.split()[0]
                self.first_words.add(first_word)
        
        return topics

# ======================
# 2. CUSTOM TOKENIZATION
# ======================
class Tokenizer:
    def __init__(self, topics, min_freq=1):  # Lower min_freq to include more words
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.word2idx = {}
        self.idx2word = {}
        self.max_len = 0
        self.first_words = set()
        self.build_vocab(topics, min_freq)
    
    def build_vocab(self, topics, min_freq):
        # Collect first words and calculate max length
        for topic in topics:
            words = topic.split()
            if words:
                self.first_words.add(words[0])
                if len(words) > self.max_len:
                    self.max_len = len(words)
        
        # Count words
        word_counts = Counter()
        for topic in topics:
            words = topic.split()
            word_counts.update(words)
        
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
        
        # Add words meeting frequency threshold
        idx = len(self.special_tokens)
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        # Add all first words (even if infrequent)
        for word in self.first_words:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, text, max_length):
        words = text.split()
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Add start and end tokens
        tokens = [self.word2idx['<SOS>']] + tokens + [self.word2idx['<EOS>']]
        
        # Pad sequence
        padded = tokens + [self.word2idx['<PAD>']] * (max_length - len(tokens))
        return padded[:max_length]
    
    def decode(self, token_seq):
        words = []
        for token in token_seq:
            if token in [self.word2idx['<SOS>'], self.word2idx['<PAD>']]:
                continue
            if token == self.word2idx['<EOS>']:
                break
            words.append(self.idx2word.get(token, '<UNK>'))
        return ' '.join(words)

# ======================
# 3. GPT-2 STYLE TRANSFORMER
# ======================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, 1e-5
        )

class GPT2Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Project to query, key, value
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(attn_output)

class GPT2Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model, bias=True)
        self.attn = GPT2Attention(d_model, nhead, dropout)
        self.ln2 = LayerNorm(d_model, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ResearchTopicGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8, max_length=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, nhead) for _ in range(num_layers)
        ])
        
        self.ln_f = LayerNorm(d_model, bias=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Apply GPT-2 style initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        B, T = x.size()
        token_emb = self.embedding(x)
        pos_emb = self.pos_embed[:, :T, :]
        x = self.dropout(token_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return self.fc_out(x)
    
    def generate(self, tokenizer, start_word, max_len=30, temperature=0.8, top_p=0.9):
        self.eval()
        start_idx = tokenizer.word2idx.get(start_word, tokenizer.word2idx['<UNK>'])
        tokens = [tokenizer.word2idx['<SOS>'], start_idx]
        
        with torch.no_grad():
            for _ in range(max_len):
                seq = torch.tensor(tokens[-50:]).unsqueeze(0)  # Use last 50 tokens
                output = self(seq)
                logits = output[0, -1, :] / temperature
                
                # Apply nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                
                tokens.append(next_idx)
                if next_idx == tokenizer.word2idx['<EOS>'] or len(tokens) >= max_len + 2:
                    break
        
        return tokenizer.decode(tokens)

# ======================
# 4. DATA LOADER
# ======================
class TopicDataset(Dataset):
    def __init__(self, topics, tokenizer, max_length):
        self.sequences = []
        self.max_length = max_length
        
        for topic in topics:
            encoded = tokenizer.encode(topic, max_length)
            self.sequences.append(encoded)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = torch.tensor(seq[:-1])
        target_seq = torch.tensor(seq[1:])
        return input_seq, target_seq

# ======================
# 5. TRAINING PROCESS WITH OPTIMIZED HYPERPARAMETERS
# ======================
def get_optimizer(model, lr=5e-4, weight_decay=0.01):
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or name.endswith('norm.weight'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def train_model(model, dataloader, tokenizer, epochs=50, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Total training steps
    total_steps = len(dataloader) * epochs
    warmup_steps = total_steps * 0.1  # 10% warmup
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, lr=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
    
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss only on non-padding tokens
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_tokens = (targets != tokenizer.word2idx['<PAD>']).sum().item()
            
            # Print progress
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Tokens: {total_tokens}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        
        # Generate sample topics
        print("Generated Topics:")
        sample_words = random.sample(list(tokenizer.first_words), min(5, len(tokenizer.first_words)))
        for word in sample_words:
            generated = model.generate(tokenizer, word, temperature=0.7, top_p=0.9)
            print(f"- {word}: {generated}")
        print()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }
        
        torch.save(checkpoint, f'checkpoints/epoch_{epoch+1}.pth')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def adaptive_fine_tune(model, generated_topics, tokenizer, epochs=1, lr=1e-5):
    if not generated_topics:
        return model
        
    device = next(model.parameters()).device
    dataset = TopicDataset(generated_topics, tokenizer, tokenizer.max_len+2)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
    optimizer = get_optimizer(model, lr=lr, weight_decay=0.0)  # No weight decay for fine-tuning
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Adaptive fine-tuning epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# ======================
# 6. BODEN CREATIVITY ENGINE WITH NOVELTY
# ======================
class CreativityEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.novelty_threshold = 3.5  # Higher threshold for innovation
        self.first_word_embeddings = self._extract_first_word_embeddings()
    
    def _extract_first_word_embeddings(self):
        embeddings = {}
        for word in self.tokenizer.first_words:
            word_idx = self.tokenizer.word2idx.get(word, self.tokenizer.word2idx['<UNK>'])
            with torch.no_grad():
                embedding = self.model.embedding(torch.tensor([word_idx]))
                embeddings[word] = embedding
        return embeddings
    
    def combinatorial_creativity(self, k=3, temp=0.7, top_p=0.85):
        # Select diverse words
        selected_words = random.sample(list(self.tokenizer.first_words), k)
        selected_embeddings = [self.first_word_embeddings[word] for word in selected_words]
        
        # Weighted interpolation
        weights = torch.softmax(torch.randn(k), dim=0)
        interpolated = sum(w * emb for w, emb in zip(weights, selected_embeddings))
        
        # Find closest actual first word
        closest_word = self._find_closest_word(interpolated)
        return self.model.generate(self.tokenizer, closest_word, temperature=temp, top_p=top_p)
    
    def exploratory_creativity(self, word, noise_scale=0.3, temp=0.8, top_p=0.9):
        if word not in self.first_word_embeddings:
            word = random.choice(list(self.tokenizer.first_words))
        
        # Add directional noise
        vector = self.first_word_embeddings[word]
        noise = torch.randn_like(vector) * noise_scale
        noisy_vector = vector + noise
        
        # Find closest actual first word
        closest_word = self._find_closest_word(noisy_vector)
        return self.model.generate(self.tokenizer, closest_word, temperature=temp, top_p=top_p)
    
    def _find_closest_word(self, vector):
        min_dist = float('inf')
        closest_word = None
        
        for word, emb in self.first_word_embeddings.items():
            dist = torch.norm(emb - vector)
            if dist < min_dist:
                min_dist = dist
                closest_word = word
        
        return closest_word
    
    def transformative_creativity(self, novel_topics, epochs=10):
        print("Performing transformative creativity with novel topics...")
        # Fine-tune model with novel topics
        dataset = TopicDataset(novel_topics, self.tokenizer, self.tokenizer.max_len+2)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Use a lower learning rate for transformative creativity
        optimizer = get_optimizer(self.model, lr=1e-4, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word2idx['<PAD>'])
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            print(f"Transformative epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
        # Update embeddings after training
        self.first_word_embeddings = self._extract_first_word_embeddings()
        torch.save(self.model.state_dict(), f'transformative_model.pth')

# ======================
# 7. HUMAN FEEDBACK WITH NOVELTY SCORING
# ======================
def get_human_feedback_with_novelty(generated_topics):
    valid_topics = []
    novelty_scores = []
    
    print("\nGenerated Topics for Feedback (rate novelty 1-5):")
    for i, topic in enumerate(generated_topics, 1):
        print(f"{i}. {topic}")
        valid = input(f"Is this topic valid? (y/n): ").strip().lower()
        if valid == 'y':
            novelty = 0
            while novelty < 1 or novelty > 5:
                try:
                    novelty = int(input("  Rate novelty (1-5): "))
                except ValueError:
                    print("Please enter a number between 1-5")
            novelty_scores.append(novelty)
            valid_topics.append(topic)
        else:
            novelty_scores.append(0)
        print()
    
    # Calculate average novelty
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
    print(f"Average novelty score: {avg_novelty:.2f}/5.0")
    
    return valid_topics, novelty_scores, avg_novelty

# ======================
# MAIN EXECUTION
# ======================
def main():
    # 1. Data preparation
    preprocessor = TextPreprocessor()
    topics = preprocessor.process_csv('preexisting_research_topics_cleaned.csv')
    
    # 2. Tokenization
    tokenizer = Tokenizer(topics, min_freq=1)  # Lower frequency threshold
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Max sequence length: {tokenizer.max_len}")
    print(f"First words count: {len(tokenizer.first_words)}")
    
    # 3. Prepare dataset
    dataset = TopicDataset(topics, tokenizer, max_length=tokenizer.max_len+3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. Initialize model with optimized hyperparameters
    model = ResearchTopicGPT(
        vocab_size=len(tokenizer.word2idx),
        d_model=384,         # Larger model capacity
        nhead=8,             # 8 attention heads
        num_layers=8,         # 8 transformer blocks
        max_length=100        # Increased max length
    )
    
    # 5. Initial training
    print("Starting GPT-2 style training...")
    model = train_model(model, dataloader, tokenizer, epochs=60)  # More epochs
    
    # 6. Initialize creativity engine
    engine = CreativityEngine(model, tokenizer)
    
    # 7. Creativity cycles
    for cycle in range(5):  # Run 5 creativity cycles
        print(f"\n=== CREATIVITY CYCLE {cycle+1} ===")
        
        # Combinatorial creativity
        print("\nCombinatorial Creativity:")
        combo_topics = [engine.combinatorial_creativity() for _ in range(7)]  # More variations
        for i, topic in enumerate(combo_topics, 1):
            print(f"{i}. {topic}")
        
        # Exploratory creativity
        print("\nExploratory Creativity:")
        exploratory_topics = []
        sample_words = random.sample(list(tokenizer.first_words), min(5, len(tokenizer.first_words)))
        for word in sample_words:
            topic = engine.exploratory_creativity(word, noise_scale=0.4)  # More exploration
            exploratory_topics.append(topic)
            print(f"- {word}: {topic}")
        
        # Combine all generated topics
        all_generated = combo_topics + exploratory_topics
        
        # Adaptive fine-tuning after combinatorial/exploratory
        print("\nPerforming adaptive fine-tuning...")
        model = adaptive_fine_tune(model, all_generated, tokenizer, epochs=2)  # More fine-tuning
        
        # Get human feedback with novelty scoring
        valid_topics, novelty_scores, avg_novelty = get_human_feedback_with_novelty(all_generated)
        
        # Adjust novelty threshold based on average
        if avg_novelty < 3.0:
            engine.novelty_threshold = max(3.0, engine.novelty_threshold - 0.3)
            print(f"Lowering novelty threshold to {engine.novelty_threshold:.1f}")
        elif avg_novelty > 4.0:
            engine.novelty_threshold = min(4.5, engine.novelty_threshold + 0.3)
            print(f"Raising novelty threshold to {engine.novelty_threshold:.1f}")
            
        # Filter topics based on novelty threshold
        novel_topics = [topic for topic, score in zip(valid_topics, novelty_scores) 
                        if score >= engine.novelty_threshold]
        
        print(f"Selected {len(novel_topics)} novel topics for transformative creativity")
        
        # Transformative creativity
        if novel_topics:
            print("\nPerforming transformative creativity...")
            engine.transformative_creativity(novel_topics, epochs=12)
            torch.save(model.state_dict(), f'transformed_cycle_{cycle+1}.pth')
        
        # Sample generation with increased creativity
        sample_word = random.choice(list(tokenizer.first_words))
        sample_topic = model.generate(tokenizer, sample_word, temperature=0.9, top_p=0.95)
        print(f"\nSample Generation ({sample_word}): {sample_topic}")
        
        # Save novelty scores for analysis
        with open(f'novelty_scores_cycle_{cycle+1}.txt', 'w') as f:
            f.write(f"Novelty Threshold: {engine.novelty_threshold:.1f}\n")
            f.write(f"Average Novelty: {avg_novelty:.2f}\n\n")
            for i, (topic, score) in enumerate(zip(all_generated, novelty_scores), 1):
                f.write(f"Topic {i}: {score} - {topic}\n")
    
    print("\nTraining and creativity cycles completed!")
    torch.save(model.state_dict(), 'final_creative_model.pth')

if __name__ == "__main__":
    main()