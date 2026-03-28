// ── DOM refs ───────────────────────────────────────────────
const chatBox      = document.getElementById('chat-box');
const userInput    = document.getElementById('user-input');
const sendBtn      = document.getElementById('send-btn');
const clearBtn     = document.getElementById('clear-btn');
const tempSlider   = document.getElementById('temperature');
const tempValue    = document.getElementById('temp-value');
const tokensSlider = document.getElementById('max-tokens');
const tokensValue  = document.getElementById('tokens-value');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const tokenCounter = document.getElementById('token-counter');
const lastTokens   = document.getElementById('last-tokens');
const sessionTime  = document.getElementById('session-time');

// ── Session timer ──────────────────────────────────────────
const sessionStart = Date.now();
setInterval(() => {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    const m = Math.floor(s / 60);
    const ss = String(s % 60).padStart(2, '0');
    sessionTime.textContent = `${m}:${ss}`;
}, 1000);

// ── Slider bindings ────────────────────────────────────────
tempSlider.addEventListener('input', () => {
    tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
});
tokensSlider.addEventListener('input', () => {
    tokensValue.textContent = tokensSlider.value;
});

// ── Auto-resize textarea ───────────────────────────────────
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
});

// ── Status check ───────────────────────────────────────────
async function checkStatus() {
    try {
        const res  = await fetch('/status-proxy');
        const data = await res.json();
        statusDot.className  = 'status-dot online';
        statusText.textContent = 'Modelo en línea';

        if (data.parameters) {
            document.getElementById('stat-params').textContent  =
                (data.parameters.total / 1000).toFixed(0) + 'K';
            document.getElementById('stat-layers').textContent  = data.parameters.n_layer;
            document.getElementById('stat-context').textContent = data.parameters.block_size + ' tok';
        }
    } catch {
        statusDot.className    = 'status-dot offline';
        statusText.textContent = 'Desconectado';
    }
}

checkStatus();
setInterval(checkStatus, 15000); // Re-check cada 15s

// ── Helpers ────────────────────────────────────────────────
function getTime() {
    return new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
}

function createTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.id = 'typing-indicator';
    el.innerHTML = `
        <div class="msg-avatar">◈</div>
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    return el;
}

function appendMessage(text, sender, isError = false) {
    const wrap = document.createElement('div');
    wrap.className = `message ${sender}-message`;

    const avatar = sender === 'ai' ? '◈' : '›';

    wrap.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-body">
            <div class="msg-meta">
                ${sender === 'ai' ? 'NeuralChat' : 'Tú'}
                <span class="msg-time">${getTime()}</span>
            </div>
            <div class="msg-text${isError ? ' error-text' : ''}"></div>
        </div>
    `;

    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrap.querySelector('.msg-text');
}

// Efecto máquina de escribir
async function streamText(el, text, speed = 18) {
    el.textContent = '';
    for (let i = 0; i < text.length; i++) {
        el.textContent += text[i];
        chatBox.scrollTop = chatBox.scrollHeight;
        await new Promise(r => setTimeout(r, speed));
    }
}

// ── Send message ───────────────────────────────────────────
let isLoading = false;

async function sendMessage() {
    if (isLoading) return;
    const text = userInput.value.trim();
    if (!text) return;

    // Mostrar mensaje del usuario
    appendMessage(text, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    // Typing indicator
    const typing = createTypingIndicator();
    chatBox.appendChild(typing);
    chatBox.scrollTop = chatBox.scrollHeight;
    isLoading = true;
    sendBtn.disabled = true;

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                temperature: parseFloat(tempSlider.value),
                max_tokens: parseInt(tokensSlider.value),
            })
        });

        const data = await res.json();
        typing.remove();

        if (data.error) {
            appendMessage(`Error: ${data.error}`, 'ai', true);
        } else {
            const aiEl = appendMessage('', 'ai');
            await streamText(aiEl, data.response);

            // Mostrar contador de tokens
            if (data.tokens_generated) {
                tokenCounter.style.display = 'block';
                lastTokens.textContent = data.tokens_generated;
            }
        }
    } catch {
        typing.remove();
        appendMessage(
            'Sin conexión con los servidores. ¿Están corriendo main.py y server.js?',
            'ai', true
        );
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// ── Clear chat ─────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
    // Mantener solo el mensaje de bienvenida
    chatBox.innerHTML = `
        <div class="message ai-message">
            <div class="msg-avatar">◈</div>
            <div class="msg-body">
                <div class="msg-meta">NeuralChat <span class="msg-time">${getTime()}</span></div>
                <div class="msg-text">Chat limpiado. Escribe un nuevo prompt para comenzar.</div>
            </div>
        </div>
    `;
    tokenCounter.style.display = 'none';
});

// ── Events ─────────────────────────────────────────────────
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});