document.addEventListener('DOMContentLoaded', function() {
    // Mode switching
    const modeButtons = document.querySelectorAll('.mode-btn');
    const modeContents = document.querySelectorAll('.mode-content');
    
    modeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const mode = this.getAttribute('data-mode');
            
            // Update active button
            modeButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding content
            modeContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${mode}-mode`) {
                    content.classList.add('active');
                }
            });
        });
    });
    
    // Update slider values
    const speedSlider = document.getElementById('speed');
    const pitchSlider = document.getElementById('pitch');
    const speedValue = document.getElementById('speed-value');
    const pitchValue = document.getElementById('pitch-value');
    
    speedSlider.addEventListener('input', function() {
        speedValue.textContent = this.value;
    });
    
    pitchSlider.addEventListener('input', function() {
        pitchValue.textContent = this.value;
    });
    
    // Audio elements
    const audioPlayer = document.getElementById('audio-player');
    const qaAudioPlayer = document.getElementById('qa-audio-player');
    const multiAudioPlayer = document.getElementById('multi-audio-player');
    
    // Status elements
    const statusElement = document.getElementById('status');
    const qaStatusElement = document.getElementById('qa-status');
    const multiStatusElement = document.getElementById('multi-status');
    
    // Download button
    const downloadBtn = document.getElementById('download-btn');
    let currentAudioBlob = null;
    
    // Generate buttons
    const generateBtn = document.getElementById('generate-btn');
    const generateQaBtn = document.getElementById('generate-qa-btn');
    const generateMultiBtn = document.getElementById('generate-multi-btn');
    
    // Simulate TTS generation (demo mode)
    function simulateTTS(text, voice, speed, pitch) {
        return new Promise((resolve) => {
            // In demo mode, we create a dummy audio blob
            // In real implementation, this would call your Python backend API
            const backendUrl = document.getElementById('backend-url').value;
            
            if (!backendUrl) {
                // Demo mode - simulate processing
                setTimeout(() => {
                    statusElement.textContent = `Generated audio for: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"\nVoice: ${voice}\nSpeed: ${speed}x\nPitch: ${pitch}`;
                    
                    // Create a silent audio file for demo
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const duration = Math.min(text.length * 0.1, 30); // Max 30 seconds
                    const sampleRate = 24000;
                    const numSamples = duration * sampleRate;
                    
                    const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
                    const channelData = audioBuffer.getChannelData(0);
                    
                    // Add a simple beep for demo
                    for (let i = 0; i < numSamples; i++) {
                        const t = i / sampleRate;
                        // Generate a tone
                        channelData[i] = 0.3 * Math.sin(2 * Math.PI * 440 * t) * Math.exp(-0.001 * t);
                    }
                    
                    // Convert to WAV
                    const wavBytes = audioBufferToWav(audioBuffer);
                    const blob = new Blob([wavBytes], { type: 'audio/wav' });
                    
                    resolve({
                        success: true,
                        audioUrl: URL.createObjectURL(blob),
                        stats: `Duration: ${duration.toFixed(1)}s | Sample rate: ${sampleRate}Hz`,
                        blob: blob
                    });
                }, 2000);
            } else {
                // Real API call to backend
                fetchBackendAPI(text, voice, speed, pitch, backendUrl).then(resolve);
            }
        });
    }
    
    // Function to call real backend API
    async function fetchBackendAPI(text, voice, speed, pitch, backendUrl) {
        statusElement.textContent = 'Connecting to backend API...';
        
        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    voice: voice,
                    speed: parseFloat(speed),
                    pitch: parseFloat(pitch),
                    mode: 'standard'
                })
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                // Assuming the API returns base64 audio data
                const audioBlob = base64ToBlob(data.audio_data, 'audio/mp3');
                const audioUrl = URL.createObjectURL(audioBlob);
                
                return {
                    success: true,
                    audioUrl: audioUrl,
                    stats: data.stats || 'Audio generated successfully',
                    blob: audioBlob
                };
            } else {
                throw new Error(data.error || 'Unknown error from backend');
            }
        } catch (error) {
            console.error('API call failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    // Helper function to convert AudioBuffer to WAV
    function audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;
        
        const bufferLength = buffer.length;
        const dataSize = bufferLength * blockAlign;
        
        // Create WAV header
        const header = new ArrayBuffer(44);
        const view = new DataView(header);
        
        // RIFF identifier
        writeString(view, 0, 'RIFF');
        // RIFF chunk length
        view.setUint32(4, 36 + dataSize, true);
        // RIFF type
        writeString(view, 8, 'WAVE');
        // Format chunk identifier
        writeString(view, 12, 'fmt ');
        // Format chunk length
        view.setUint32(16, 16, true);
        // Sample format (raw)
        view.setUint16(20, format, true);
        // Channel count
        view.setUint16(22, numChannels, true);
        // Sample rate
        view.setUint32(24, sampleRate, true);
        // Byte rate (sample rate * block align)
        view.setUint32(28, sampleRate * blockAlign, true);
        // Block align (channel count * bytes per sample)
        view.setUint16(32, blockAlign, true);
        // Bits per sample
        view.setUint16(34, bitDepth, true);
        // Data chunk identifier
        writeString(view, 36, 'data');
        // Data chunk length
        view.setUint32(40, dataSize, true);
        
        // Write PCM samples
        const data = new Uint8Array(header.byteLength + dataSize);
        data.set(new Uint8Array(header), 0);
        
        // Interleave the audio data
        const offset = 44;
        const channels = [];
        for (let i = 0; i < numChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }
        
        // Write interleaved data
        for (let i = 0; i < bufferLength; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, channels[channel][i]));
                const intValue = sample < 0 ? sample * 32768 : sample * 32767;
                const index = offset + (i * blockAlign) + (channel * bytesPerSample);
                view.setInt16(index, intValue, true);
            }
        }
        
        return data.buffer;
    }
    
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    function base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteArrays = [];
        
        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
            const slice = byteCharacters.slice(offset, offset + 512);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }
        
        return new Blob(byteArrays, { type: mimeType });
    }
    
    // Standard mode generation
    generateBtn.addEventListener('click', async function() {
        const text = document.getElementById('text-input').value;
        const voice = document.getElementById('voice-select').value;
        const speed = speedSlider.value;
        const pitch = pitchSlider.value;
        
        if (!text.trim()) {
            statusElement.textContent = 'Please enter some text to convert to speech.';
            return;
        }
        
        // Disable button during processing
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        statusElement.textContent = 'Starting TTS generation...';
        
        try {
            const result = await simulateTTS(text, voice, speed, pitch);
            
            if (result.success) {
                // Play audio
                audioPlayer.src = result.audioUrl;
                audioPlayer.load();
                
                // Enable download
                currentAudioBlob = result.blob;
                downloadBtn.disabled = false;
                
                // Update status
                statusElement.textContent = `✓ Audio generated successfully!\n${result.stats}\n\nText processed: ${text.length} characters`;
                
                // Auto-play (optional)
                // audioPlayer.play().catch(e => console.log('Auto-play prevented:', e));
            } else {
                statusElement.textContent = `✗ Error: ${result.error}`;
            }
        } catch (error) {
            console.error('Generation failed:', error);
            statusElement.textContent = `✗ Error: ${error.message}`;
        } finally {
            // Re-enable button
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-play"></i> Generate Speech';
        }
    });
    
    // Q&A mode generation (simplified)
    generateQaBtn.addEventListener('click', async function() {
        const text = document.getElementById('qa-text-input').value;
        const voiceQ = document.getElementById('voice-q').value;
        const voiceA = document.getElementById('voice-a').value;
        
        if (!text.trim()) {
            qaStatusElement.textContent = 'Please enter Q&A text.';
            return;
        }
        
        generateQaBtn.disabled = true;
        generateQaBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        qaStatusElement.textContent = 'Generating Q&A audio...';
        
        // Simulate processing
        setTimeout(() => {
            qaStatusElement.textContent = 'Q&A audio generated (demo mode).\nIn production, this would call the backend API.';
            
            // Create demo audio
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const duration = 5; // 5 second demo
            const sampleRate = 24000;
            const numSamples = duration * sampleRate;
            
            const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
            const channelData = audioBuffer.getChannelData(0);
            
            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                channelData[i] = 0.2 * Math.sin(2 * Math.PI * 523.25 * t) * Math.exp(-0.002 * t);
            }
            
            const wavBytes = audioBufferToWav(audioBuffer);
            const blob = new Blob([wavBytes], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            
            qaAudioPlayer.src = audioUrl;
            qaAudioPlayer.load();
            
            generateQaBtn.disabled = false;
            generateQaBtn.innerHTML = '<i class="fas fa-play"></i> Generate Q&A Audio';
        }, 2000);
    });
    
    // Multi-character mode generation (simplified)
    generateMultiBtn.addEventListener('click', async function() {
        const text = document.getElementById('multi-text-input').value;
        const char1Voice = document.getElementById('char1-voice').value;
        const char2Voice = document.getElementById('char2-voice').value;
        const char3Voice = document.getElementById('char3-voice').value;
        
        if (!text.trim()) {
            multiStatusElement.textContent = 'Please enter multi-character dialogue.';
            return;
        }
        
        generateMultiBtn.disabled = true;
        generateMultiBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        multiStatusElement.textContent = 'Generating multi-character audio...';
        
        // Simulate processing
        setTimeout(() => {
            multiStatusElement.textContent = 'Multi-character audio generated (demo mode).\nIn production, this would call the backend API.';
            
            // Create demo audio
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const duration = 7; // 7 second demo
            const sampleRate = 24000;
            const numSamples = duration * sampleRate;
            
            const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
            const channelData = audioBuffer.getChannelData(0);
            
            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                // Multiple tones for multiple characters
                const freq = 440 + 100 * Math.sin(t * 2);
                channelData[i] = 0.15 * Math.sin(2 * Math.PI * freq * t) * Math.exp(-0.0015 * t);
            }
            
            const wavBytes = audioBufferToWav(audioBuffer);
            const blob = new Blob([wavBytes], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            
            multiAudioPlayer.src = audioUrl;
            multiAudioPlayer.load();
            
            generateMultiBtn.disabled = false;
            generateMultiBtn.innerHTML = '<i class="fas fa-play"></i> Generate Multi-Character Audio';
        }, 3000);
    });
    
    // Download button handler
    downloadBtn.addEventListener('click', function() {
        if (currentAudioBlob) {
            const url = URL.createObjectURL(currentAudioBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'tts-output.mp3';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    });
    
    // Stop button
    document.getElementById('stop-btn').addEventListener('click', function() {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
        qaAudioPlayer.pause();
        qaAudioPlayer.currentTime = 0;
        multiAudioPlayer.pause();
        multiAudioPlayer.currentTime = 0;
    });
    
    // Initialize
    statusElement.textContent = 'Ready to generate speech. Enter text and click "Generate Speech".';
});
