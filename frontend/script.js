const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Crear y añadir el indicador de "escribiendo..."
const typingIndicator = document.createElement('div');
typingIndicator.className = 'typing-indicator';
typingIndicator.textContent = 'La IA está procesando los tensores...';
chatBox.appendChild(typingIndicator);

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    
    // Insertamos el mensaje antes del indicador de escribiendo
    chatBox.insertBefore(msgDiv, typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

// Efecto de streaming visual
async function streamText(element, text, speed = 30) {
    element.textContent = "";
    for (let i = 0; i < text.length; i++) {
        element.textContent += text.charAt(i);
        chatBox.scrollTop = chatBox.scrollHeight;
        await new Promise(resolve => setTimeout(resolve, speed));
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Mostrar mensaje del usuario
    appendMessage(text, 'user');
    userInput.value = '';
    
    // Mostrar "escribiendo..."
    typingIndicator.style.display = 'block';
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        
        // Ocultar "escribiendo..."
        typingIndicator.style.display = 'none';

        if (data.error) {
            appendMessage(`Error: ${data.error}`, 'ai');
        } else {
            // Creamos el div de la IA vacío y lanzamos el efecto máquina de escribir
            const aiMsgDiv = appendMessage('', 'ai');
            await streamText(aiMsgDiv, data.response);
        }
    } catch (err) {
        typingIndicator.style.display = 'none';
        appendMessage('Error de red. Asegúrate de que los servidores estén corriendo.', 'ai');
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});