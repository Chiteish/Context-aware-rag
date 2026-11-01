document.addEventListener('DOMContentLoaded', () => {
    // Session Helper Functions
    function generateUUID() {
        if (self.crypto && self.crypto.randomUUID) {
            return self.crypto.randomUUID();
        }
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Initialize or retrieve the session ID
    let currentSessionId = localStorage.getItem('current_session_id');
    if (!currentSessionId) {
        currentSessionId = generateUUID();
        localStorage.setItem('current_session_id', currentSessionId);
    }

    // DOM Elements
    const chatForm = document.getElementById('chatForm');
    const queryInput = document.getElementById('queryInput');
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const sendBtn = document.getElementById('sendBtn');
    
    // Upload & Prompt States
    const uploadPromptState = document.getElementById('uploadPromptState');
    const readyPromptState = document.getElementById('readyPromptState');
    const dragDropZone = document.getElementById('dragDropZone');
    const mainFileInput = document.getElementById('mainFileInput');
    const sidebarFileInput = document.getElementById('sidebarFileInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressText = document.getElementById('uploadProgressText');
    
    // Sidebar elements
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const metaDoc = document.getElementById('metaDoc');
    const metaChunks = document.getElementById('metaChunks');
    const metaEmbed = document.getElementById('metaEmbed');
    const metaLLM = document.getElementById('metaLLM');
    
    // Sources elements
    const sourcesBadge = document.getElementById('sourcesBadge');
    const sourcesIntro = document.getElementById('sourcesIntro');
    const sourcesList = document.getElementById('sourcesList');
    
    // Control Buttons
    const newChatBtn = document.getElementById('newChatBtn');
    const clearBtn = document.getElementById('clearBtn');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileCloseBtn = document.getElementById('mobileCloseBtn');
    const sidebar = document.getElementById('sidebar');

    // Configure Marked.js options
    marked.setOptions({
        gfm: true,
        breaks: true,
        headerIds: false,
        mangle: false
    });

    // Auto-grow input text area
    queryInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Suggestion chips
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            if (queryInput.disabled) return;
            const query = card.getAttribute('data-query');
            queryInput.value = query;
            queryInput.dispatchEvent(new Event('input')); // trigger auto-grow
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Mobile Sidebar Toggles
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.add('active');
    });

    mobileCloseBtn.addEventListener('click', () => {
        sidebar.classList.remove('active');
    });

    let isProcessing = false;

    // Fetch pipeline info on startup
    async function checkSystemStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${currentSessionId}`);
            if (!response.ok) throw new Error('API status failure');
            
            const data = await response.json();
            
            // Render details
            metaDoc.textContent = data.document_name || 'None';
            metaDoc.title = data.document_name || 'None';
            metaChunks.textContent = data.chunks_count === -1 ? 'Unknown' : (data.chunks_count === 0 ? '-' : data.chunks_count);
            metaEmbed.textContent = data.embedding_model;
            metaLLM.textContent = data.llm_model;

            if (data.status === 'ready') {
                statusDot.className = 'status-dot ready pulsing';
                statusDot.style.backgroundColor = ''; // Use CSS default
                statusText.textContent = 'Ready';
                
                // Show chat view
                uploadPromptState.style.display = 'none';
                readyPromptState.style.display = 'block';
                
                // Enable inputs
                queryInput.removeAttribute('disabled');
                queryInput.placeholder = 'Ask a question about the document...';
                sendBtn.removeAttribute('disabled');
                isProcessing = false;
            } else if (data.status === 'awaiting_document') {
                statusDot.className = 'status-dot pulsing';
                statusDot.style.backgroundColor = '#6366f1'; // Indigo for awaiting
                statusText.textContent = 'Awaiting Upload';
                
                // Show upload view
                uploadPromptState.style.display = 'block';
                readyPromptState.style.display = 'none';
                dragDropZone.style.display = 'flex';
                uploadProgress.style.display = 'none';
                
                // Disable inputs
                queryInput.setAttribute('disabled', 'true');
                queryInput.placeholder = 'Please upload a PDF document to begin...';
                sendBtn.setAttribute('disabled', 'true');
                isProcessing = false;
            } else if (data.status === 'processing') {
                statusDot.className = 'status-dot pulsing';
                statusDot.style.backgroundColor = '#06b6d4'; // Cyan for processing
                statusText.textContent = 'Processing...';
                
                // Show progress state
                uploadPromptState.style.display = 'block';
                readyPromptState.style.display = 'none';
                dragDropZone.style.display = 'none';
                uploadProgress.style.display = 'flex';
                uploadProgressText.textContent = 'Processing & Embedding PDF...';
                
                // Disable inputs
                queryInput.setAttribute('disabled', 'true');
                queryInput.placeholder = 'Processing document, please wait...';
                sendBtn.setAttribute('disabled', 'true');
                isProcessing = true;
            } else if (data.status.startsWith('error')) {
                statusDot.className = 'status-dot';
                statusDot.style.backgroundColor = '#ef4444';
                statusText.textContent = 'Error';
                console.error(data.status);
                
                // Show upload view with error details
                uploadPromptState.style.display = 'block';
                readyPromptState.style.display = 'none';
                dragDropZone.style.display = 'flex';
                uploadProgress.style.display = 'none';
                
                // Disable inputs
                queryInput.setAttribute('disabled', 'true');
                queryInput.placeholder = 'System error occurred. Check server logs.';
                sendBtn.setAttribute('disabled', 'true');
                isProcessing = false;
            }

        } catch (error) {
            statusDot.className = 'status-dot';
            statusDot.style.backgroundColor = '#ef4444';
            statusText.textContent = 'Offline';
            console.error('Error fetching system status:', error);
        }
    }

    // File Upload Handler
    async function handleFileUpload(file) {
        if (!file) return;
        if (!file.name.endsWith('.pdf')) {
            alert('Only PDF files are supported.');
            return;
        }

        // Show progress state immediately in UI
        dragDropZone.style.display = 'none';
        uploadProgress.style.display = 'flex';
        uploadProgressText.textContent = 'Uploading PDF...';
        
        statusDot.className = 'status-dot pulsing';
        statusDot.style.backgroundColor = '#06b6d4';
        statusText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`/api/upload?session_id=${currentSessionId}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Upload failed');
            }

            // Immediately check status
            checkSystemStatus();
        } catch (err) {
            console.error('Upload error:', err);
            statusDot.className = 'status-dot';
            statusDot.style.backgroundColor = '#ef4444';
            statusText.textContent = 'Upload Failed';
            
            // Show error in the progress area and reset back to drag-drop after 4s
            uploadProgressText.innerHTML = `<span style="color: #ef4444;">Upload Failed: ${err.message}</span>`;
            setTimeout(() => {
                checkSystemStatus();
            }, 4000);
        }
    }

    // Drag & Drop event listeners
    dragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragDropZone.classList.add('dragover');
    });

    dragDropZone.addEventListener('dragleave', () => {
        dragDropZone.classList.remove('dragover');
    });

    dragDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dragDropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    dragDropZone.addEventListener('click', () => {
        mainFileInput.click();
    });

    mainFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    sidebarFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Initialize status check and poll every 5 seconds to track upload processing state
    checkSystemStatus();
    setInterval(checkSystemStatus, 5000);

    // Append a message to the chat display
    function appendMessage(sender, content, isHtml = false) {
        // Hide welcome screen if visible
        if (welcomeScreen.style.display !== 'none') {
            welcomeScreen.style.display = 'none';
        }

        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${sender}`;

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        if (isHtml) {
            messageBubble.innerHTML = content;
        } else {
            messageBubble.textContent = content;
        }

        messageWrapper.appendChild(messageBubble);
        messagesContainer.appendChild(messageWrapper);
        
        // Smooth scroll to bottom
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Show/Remove loading typing indicators
    let typingIndicatorElement = null;
    
    function showTypingIndicator() {
        if (typingIndicatorElement) return;

        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper assistant';
        messageWrapper.id = 'typingIndicator';

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.className = 'typing-dot';
            indicator.appendChild(dot);
        }

        messageBubble.appendChild(indicator);
        messageWrapper.appendChild(messageBubble);
        messagesContainer.appendChild(messageWrapper);
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
        
        typingIndicatorElement = messageWrapper;
    }

    function removeTypingIndicator() {
        if (typingIndicatorElement) {
            typingIndicatorElement.remove();
            typingIndicatorElement = null;
        }
    }

    // Render retrieved sources to sidebar
    function renderSources(sources) {
        sourcesList.innerHTML = '';
        sourcesBadge.textContent = sources.length;

        if (sources.length === 0) {
            sourcesIntro.style.display = 'block';
            return;
        }

        sourcesIntro.style.display = 'none';

        sources.forEach((src, idx) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            // Header showing index and metadata
            const header = document.createElement('div');
            header.className = 'source-item-header';
            header.innerHTML = `
                <span>[${idx + 1}] Page ${src.page}</span>
                <span>${src.source}</span>
            `;

            // Content snippet
            const content = document.createElement('div');
            content.className = 'source-content';
            content.textContent = src.page_content;

            sourceItem.appendChild(header);
            sourceItem.appendChild(content);

            // Toggle expansion state on click to view full document text context
            sourceItem.addEventListener('click', () => {
                sourceItem.classList.toggle('expanded');
            });

            sourcesList.appendChild(sourceItem);
        });
    }

    // Handle Form Submit (Sending message)
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query || queryInput.disabled) return;

        // Clear input and reset auto-grown height
        queryInput.value = '';
        queryInput.style.height = 'auto';
        
        // Add user message to UI
        appendMessage('user', query);
        
        // Display loading typing state
        showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: query, session_id: currentSessionId })
            });

            removeTypingIndicator();

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch RAG response');
            }

            // Create message bubble placeholders dynamically
            if (welcomeScreen.style.display !== 'none') {
                welcomeScreen.style.display = 'none';
            }

            const messageWrapper = document.createElement('div');
            messageWrapper.className = 'message-wrapper assistant';
            const messageBubble = document.createElement('div');
            messageBubble.className = 'message-bubble';
            messageWrapper.appendChild(messageBubble);
            messagesContainer.appendChild(messageWrapper);

            // Read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let assistantAnswer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep partial line in buffer

                for (const line of lines) {
                    const cleanLine = line.trim();
                    if (!cleanLine.startsWith('data: ')) continue;
                    
                    try {
                        const parsed = JSON.parse(cleanLine.substring(6));
                        if (parsed.type === 'sources') {
                            renderSources(parsed.sources);
                        } else if (parsed.type === 'token') {
                            assistantAnswer += parsed.token;
                            messageBubble.innerHTML = marked.parse(assistantAnswer);
                            
                            // Scroll down as new tokens arrive
                            messagesContainer.scrollTo({
                                top: messagesContainer.scrollHeight,
                                behavior: 'auto'
                            });
                        }
                    } catch (err) {
                        console.error('Error parsing line data:', err);
                    }
                }
            }

        } catch (error) {
            removeTypingIndicator();
            appendMessage(
                'assistant', 
                `<p style="color: #ef4444; font-weight: 500;">Connection Error:</p><p>${error.message}</p><p>Please make sure Ollama is running local LLM services and that the FastAPI backend is running.</p>`, 
                true
            );
            console.error('Chat error:', error);
        }
    });

    // Clear History Action
    clearBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear the conversation history? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            if (!response.ok) throw new Error('Failed to reset conversation history');
            
            // Reset message interface
            messagesContainer.innerHTML = '';
            welcomeScreen.style.display = 'flex';
            messagesContainer.appendChild(welcomeScreen);
            
            // Reset sources list
            renderSources([]);
            
            alert('Chat memory has been reset.');
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error('Clear chat error:', error);
        }
    });

    // New Chat Action
    newChatBtn.addEventListener('click', async () => {
        if (isProcessing) {
            alert('Please wait, a document is currently being processed.');
            return;
        }

        if (!confirm('Are you sure you want to start a new chat? This will clear the current session.')) {
            return;
        }

        try {
            // Call POST /api/reset-session to delete old files and data
            const response = await fetch('/api/reset-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            
            if (!response.ok) {
                throw new Error('Failed to reset the session on the server.');
            }
            
            // Generate new session ID
            currentSessionId = generateUUID();
            localStorage.setItem('current_session_id', currentSessionId);
            
            // Reset UI state (WITHOUT page reload)
            messagesContainer.innerHTML = '';
            welcomeScreen.style.display = 'flex';
            messagesContainer.appendChild(welcomeScreen);
            
            // Reset metadata sidebar values
            metaDoc.textContent = 'None';
            metaDoc.title = 'None';
            metaChunks.textContent = '-';
            
            // Reset prompts visibility
            readyPromptState.style.display = 'none';
            uploadPromptState.style.display = 'block';
            dragDropZone.style.display = 'flex';
            uploadProgress.style.display = 'none';
            
            // Disable inputs
            queryInput.setAttribute('disabled', 'true');
            queryInput.value = '';
            queryInput.style.height = 'auto';
            queryInput.placeholder = 'Please upload a PDF document to begin...';
            sendBtn.setAttribute('disabled', 'true');
            
            // Reset sources list
            renderSources([]);
            
            // Refresh status for the new session (will show Awaiting Upload)
            checkSystemStatus();
            
        } catch (err) {
            alert(`Error resetting session: ${err.message}`);
            console.error(err);
        }
    });
});
