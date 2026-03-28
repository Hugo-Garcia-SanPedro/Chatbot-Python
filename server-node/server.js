import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = 3000;
const PYTHON_API = 'http://127.0.0.1:8000';

app.use(express.json());

// Servir el frontend estático
app.use(express.static(path.join(__dirname)));

// Proxy de estado del modelo
app.get('/status-proxy', async (req, res) => {
    try {
        const r    = await fetch(`${PYTHON_API}/status`);
        const data = await r.json();
        res.json(data);
    } catch {
        res.status(503).json({ status: 'offline' });
    }
});

// Proxy del chat hacia la IA en Python
app.post('/chat', async (req, res) => {
    try {
        const r = await fetch(`${PYTHON_API}/generate`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(req.body),
        });
        const data = await r.json();
        res.json(data);
    } catch (err) {
        console.error('Error conectando con la IA:', err.message);
        res.status(503).json({ error: 'La IA local no responde. ¿Está corriendo main.py?' });
    }
});

app.listen(PORT, () => {
    console.log(`\n✓ Servidor Node corriendo en http://localhost:${PORT}`);
    console.log(`  → Asegúrate de tener main.py corriendo en ${PYTHON_API}\n`);
});