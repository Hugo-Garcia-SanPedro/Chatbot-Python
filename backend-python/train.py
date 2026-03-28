import torch
import torch.nn.functional as F
from main import model, block_size, vocab_size
import time

# 1. Leer los datos de entrenamiento
print("Cargando el texto de input.txt...")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Texto cargado: {len(text)} caracteres")

# Tokenización a nivel de caracteres (ASCII)
data = torch.tensor(
    [ord(c) if ord(c) < 256 else 0 for c in text],
    dtype=torch.long
)

# División 90/10 entrenamiento/validación
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

batch_size = 32   # Lotes de 32 secuencias en paralelo

def get_batch(split):
    """Genera un lote aleatorio de (entrada, objetivo)"""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size]     for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(eval_iters=50):
    """Estima la pérdida promediando varios lotes para mayor estabilidad"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits = model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 2. Optimizador con tasa de aprendizaje más baja para mayor estabilidad
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Scheduler: reduce el LR a la mitad a los 2/3 del entrenamiento
max_iters     = 5000    # Subido de 1000 → más aprendizaje
eval_interval = 250     # Evaluar cada 250 iteraciones
best_val_loss = float('inf')

# Warmup + cosine decay del learning rate
def get_lr(it):
    warmup_iters = 200
    lr_min = 1e-5
    lr_max = 3e-4
    if it < warmup_iters:
        return lr_max * it / warmup_iters
    # Cosine decay
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

print(f"\n{'='*55}")
print(f"  Iniciando entrenamiento — {max_iters} iteraciones")
print(f"  Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*55}\n")

start_time = time.time()

for iter in range(max_iters):

    # Actualizar learning rate (warmup + cosine decay)
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluación periódica
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(
            f"  Iter {iter:5d} | "
            f"train: {losses['train']:.4f} | "
            f"val: {losses['val']:.4f} | "
            f"lr: {lr:.2e} | "
            f"tiempo: {elapsed:.1f}s"
        )

        # Guardar el mejor modelo automáticamente
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'modelo_entrenado.pt')
            print(f"  ✓ Mejor modelo guardado (val loss: {best_val_loss:.4f})")

    # Paso de entrenamiento
    xb, yb = get_batch('train')
    logits  = model(xb)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient clipping: evita que los gradientes exploten
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

total_time = time.time() - start_time
print(f"\n{'='*55}")
print(f"  Entrenamiento completado en {total_time:.1f}s")
print(f"  Mejor pérdida de validación: {best_val_loss:.4f}")
print(f"  Pesos guardados en 'modelo_entrenado.pt'")
print(f"{'='*55}")