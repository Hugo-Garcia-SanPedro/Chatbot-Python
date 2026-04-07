# NeuralChat: LLM Local y Chatbot 🤖

NeuralChat es un proyecto educativo que implementa un Modelo de Lenguaje (LLM) basado en la arquitectura Transformer desde cero. El sistema se compone de una IA alojada en Python (usando FastAPI y PyTorch) y una interfaz web construida sobre Node.js y Vanilla JS.

## ✨ Características Principales

* **Arquitectura Propia:** Un modelo Transformer con 6 capas, 4 cabezas de atención y embedding de 128, diseñado directamente en PyTorch.
* **API Backend Dinámica:** Servidor FastAPI preparado para la inferencia, que soporta ajustes en vivo de "creatividad" (temperatura) y muestreo (top-k).
* **Frontend y Proxy:** Un servidor Express.js ligero que entrega la interfaz y actúa como proxy inverso para comunicar el chat de forma segura con la IA local.

---

## 🚀 Instalación y Uso

### 1. Desplegar el Backend de IA (Python)
Asegúrate de contar con Python 3.8+.
```bash
cd backend-python
pip install -r requirements.txt
# ¡IMPORTANTE!: Crea un archivo 'input.txt' en esta carpeta con el texto para entrenar tu modelo.
python train.py  # Ejecuta el script de entrenamiento
python main.py   # Arranca la API en [http://127.0.0.1:8000](http://127.0.0.1:8000)
