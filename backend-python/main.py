import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os

# --- 1. ARQUITECTURA DEL MODELO ---

# Hiperparámetros mejorados
vocab_size = 256
n_embd = 128       # Aumentado de 64 → mayor capacidad de representación
n_head = 4
n_layer = 6        # Aumentado de 4 → más profundidad
block_size = 256   # Aumentado de 128 → más contexto
dropout = 0.1      # Regularización para evitar sobreajuste

class Head(nn.Module):
    """Una cabeza de self-attention con dropout"""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),              # GELU en vez de ReLU → mejor para lenguaje
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """
        Generación mejorada con temperatura y top-k sampling.
        - temperature < 1.0 → más coherente/predecible
        - temperature > 1.0 → más creativo/aleatorio
        - top_k → solo muestrea entre los k tokens más probables
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Aplicar temperatura

            # Top-k filtering: ponemos -inf a todo lo que no esté en el top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Instanciar y cargar pesos ---
model = SimpleLLM()

if os.path.exists('modelo_entrenado.pt'):
    model.load_state_dict(torch.load('modelo_entrenado.pt', weights_only=True))
    print("¡Modelo entrenado cargado con éxito!")
else:
    print("Usando pesos aleatorios. Ejecuta train.py primero.")

model.eval()

# --- 2. API FASTAPI ---

app = FastAPI(title="Local LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.8   # Nuevo parámetro controlable desde el frontend
    max_tokens: int = 150       # Más tokens por defecto

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    parameters: dict

@app.get("/status")
async def get_status():
    """Endpoint para verificar el estado del modelo"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "status": "online",
        "model_loaded": os.path.exists('modelo_entrenado.pt'),
        "parameters": {
            "total": total_params,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "block_size": block_size,
        }
    }

@app.post("/generate")
async def generate_text(request: ChatRequest):
    input_text = request.message

    try:
        # Codificar el input en tokens ASCII
        tokens = [ord(c) if ord(c) < 256 else 0 for c in input_text[-block_size:]]
        context = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            generated_indices = model.generate(
                context,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=40,
            )

        # Decodificar solo los tokens nuevos (después del input)
        new_tokens = generated_indices[0][len(tokens):]
        chars = []
        for idx in new_tokens:
            val = idx.item()
            # Filtrar solo caracteres imprimibles y comunes
            if 32 <= val < 127 or val in (10, 13):  # ASCII imprimible + saltos de línea
                chars.append(chr(val))

        response_text = "".join(chars).strip()

        if not response_text:
            response_text = "No pude generar una respuesta coherente. Intenta entrenar el modelo más iteraciones."

        return {"response": response_text, "tokens_generated": len(new_tokens)}

    except Exception as e:
        return {"error": str(e), "response": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)