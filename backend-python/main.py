import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

# --- 1. ARQUITECTURA DEL MODELO (Inspirado en Raschka) ---

# Hiperparámetros (reducidos para este ejemplo)
vocab_size = 256  # Nivel de caracteres básico para el ejemplo
n_embd = 64
n_head = 4
n_layer = 4
block_size = 128

class Head(nn.Module):
    """Una cabeza de self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Calculamos la atención (Q * K^T / sqrt(d))
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        # Bucle autorregresivo de inferencia
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Tomar el último paso
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Instanciar modelo
model = SimpleLLM()

# --- AÑADE ESTO AQUÍ ---
import os
if os.path.exists('modelo_entrenado.pt'):
    # Cargamos el cerebro que acabamos de entrenar
    model.load_state_dict(torch.load('modelo_entrenado.pt', weights_only=True))
    print("¡Modelo entrenado cargado con éxito!")
else:
    print("Usando pesos aleatorios. No se encontró modelo_entrenado.pt")
# -----------------------

model.eval()

# --- 2. API FASTAPI ---

app = FastAPI(title="Local LLM API")

class ChatRequest(BaseModel):
    message: str

@app.post("/generate")
async def generate_text(request: ChatRequest):
    # Simulamos un pequeño retraso para el efecto de pensar
    await asyncio.sleep(1.5)
    
    # Aquí iría tu tokenizador real. Para el ejemplo, codificamos en ASCII.
    input_text = request.message
    
    try:
        context = torch.tensor([[ord(c) if ord(c) < 256 else 0 for c in input_text[-block_size:]]], dtype=torch.long)
        
        # Le pedimos que genere 100 caracteres nuevos (tokens)
        generated_indices = model.generate(context, max_new_tokens=100)
        
        # --- CAMBIA LA RESPUESTA POR ESTO ---
        # Decodificamos los números de vuelta a letras
        response_text = "".join([chr(i.item()) for i in generated_indices[0] if i.item() > 0])
        
        # Para que no repita tu mensaje inicial, se lo cortamos
        response_text = response_text[len(input_text):].strip()
        
        if not response_text:
            response_text = "..."
        # ------------------------------------
        
        return {"response": response_text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)