import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json());

// Servir el frontend estático
app.use(express.static(path.join(__dirname, '../frontend')));

// Ruta puente hacia la API de Python
app.post('/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;
        
        // Hacemos fetch a nuestra IA local en Python (FastAPI)
        const pythonResponse = await fetch('http://127.0.0.1:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMessage })
        });

        const data = await pythonResponse.json();
        res.json(data);
    } catch (error) {
        console.error("Error conectando con la IA:", error);
        res.status(500).json({ error: "La IA local no está encendida o no responde." });
    }
});

app.listen(PORT, () => {
    console.log(`Servidor Node corriendo en http://localhost:${PORT}`);
});