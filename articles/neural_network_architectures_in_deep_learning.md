# A Professional Overview of Neural Network Architectures in Deep Learning (with PyTorch Examples)

Neural networks are the foundation of modern deep learning systems. Understanding the various architectures is essential for choosing the right model for your data—whether it’s images, sequences, tabular data, or even text.

In this post, I present a concise yet comprehensive overview of key neural network architectures, their use cases, and PyTorch code examples for each.

---

## 1. Feedforward Neural Networks (Multi-Layer Perceptrons)

**Overview:**
Feedforward networks (also called MLPs) are composed of fully connected layers where data flows only in one direction. They are commonly applied to structured/tabular data, regression tasks, and as a baseline in many classification problems.

**PyTorch Implementation:**

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Example instantiation
model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
```

---

## 2. Convolutional Neural Networks (CNNs)

**Overview:**
CNNs are specialized for spatial data such as images. They apply convolutional filters to capture local patterns (edges, textures, shapes), making them the backbone of modern computer vision.

**PyTorch Implementation:**

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()
```

---

## 3. Recurrent Neural Networks (RNNs), LSTM, and GRU

**Overview:**
These architectures are designed for sequential data such as time series, language, and speech. LSTMs and GRUs address the vanishing gradient problem, allowing the model to learn long-term dependencies.

**PyTorch Implementation (LSTM Example):**

```python
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)       # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1])  # last timestep
        return out

model = LSTMNet(input_dim=50, hidden_dim=128, num_layers=2, output_dim=2)
```

---

## 4. Transformer-Based Models (BERT, GPT)

**Overview:**
Transformers use attention mechanisms to model long-range dependencies in sequences without relying on recurrence. BERT and GPT are pre-trained transformer models that achieve state-of-the-art results in natural language processing (NLP).

**Example using Hugging Face Transformers (BERT):**

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Deep learning has transformed AI."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# outputs.last_hidden_state.shape = [batch_size, seq_length, hidden_size]
```

Use `GPT2Model` or `GPT2LMHeadModel` for GPT-style generative models.

---

## 5. Autoencoders

**Overview:**
Autoencoders are unsupervised models that learn to encode inputs into a latent representation and reconstruct them. They are useful for dimensionality reduction, anomaly detection, and data compression.

**PyTorch Implementation:**

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
```

---

## 6. Generative Adversarial Networks (GANs)

**Overview:**
GANs consist of a Generator and a Discriminator competing in a minimax game. The Generator tries to create realistic data, while the Discriminator tries to distinguish fake from real data. GANs are widely used in image synthesis, super-resolution, and domain adaptation.

**PyTorch GAN (Minimal Version):**

```python
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

G = Generator(z_dim=100)
D = Discriminator()
```

---

## Conclusion

Each neural network architecture serves a distinct purpose:

| Architecture           | Best For                         | Strengths                             |
| ---------------------- | -------------------------------- | ------------------------------------- |
| **MLP**                | Tabular/classification           | Simplicity, speed                     |
| **CNN**                | Images, spatial data             | Local pattern recognition             |
| **LSTM/GRU**           | Sequential, time-series, NLP     | Long-term memory, temporal modeling   |
| **Transformer (BERT)** | NLP, long sequences              | Contextual understanding, scalability |
| **Autoencoder**        | Anomaly detection, compression   | Latent representation learning        |
| **GAN**                | Image synthesis, data generation | High-quality sample generation        |

---

*Moustafa Mohamed | AI & ML Developer*

