import torch
import torch.nn.functional as F
# Importamos el modelo y variables desde nuestro main.py
from main import model, block_size, vocab_size

# 1. Leer los datos de entrenamiento
print("Cargando el texto de input.txt...")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenización a nivel de caracteres (ASCII)
# Si hay caracteres raros, los convertimos a 0 para no romper el vocabulario de 256
data = torch.tensor([ord(c) if ord(c) < 256 else 0 for c in text], dtype=torch.long)

# Dividir en datos de entrenamiento (90%) y validación (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 32 # Cuántos bloques procesamos a la vez

def get_batch(split):
    # Genera un lote de entradas (x) y objetivos a predecir (y)
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

# 2. Configurar el Optimizador (El encargado de actualizar las neuronas)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 3. Bucle de Entrenamiento
max_iters = 1000  # Puedes subir esto a 3000 o 5000 si tienes tiempo
eval_interval = 100

print(f"Iniciando entrenamiento por {max_iters} iteraciones...")

for iter in range(max_iters):
    # Evaluar la pérdida (loss) periódicamente
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            logits = model(xb)
            B, T, C = logits.shape
            # Calculamos la Entropía Cruzada (qué tan equivocada está la IA)
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            print(f"Iteración {iter}: Pérdida de validación = {loss.item():.4f}")
        model.train()

    # Obtener lote, calcular predicciones y pérdida
    xb, yb = get_batch('train')
    logits = model(xb)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
    
    # Retropropagación: ajustar los pesos matemáticos
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Entrenamiento finalizado. Pérdida final: {loss.item():.4f}")

# 4. Guardar el "Cerebro" de la IA
torch.save(model.state_dict(), 'modelo_entrenado.pt')
print("¡Pesos guardados exitosamente en 'modelo_entrenado.pt'!")