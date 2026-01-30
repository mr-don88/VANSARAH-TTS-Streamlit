// netlify/functions/tts.js
const axios = require('axios');
const { Readable } = require('stream');

exports.handler = async function(event, context) {
    // Handle CORS
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            body: ''
        };
    }

    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ error: 'Method not allowed' })
        };
    }

    try {
        const { text, voice, rate = '+0%', volume = '100%' } = JSON.parse(event.body);

        if (!text || !voice) {
            return {
                statusCode: 400,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ error: 'Text and voice are required' })
            };
        }

        // Using a public TTS service as a fallback
        // You can replace this with your preferred TTS service
        const ttsResponse = await generateTTS(text, voice, rate, volume);

        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'audio/mpeg',
                'Content-Disposition': 'attachment; filename="tts_audio.mp3"'
            },
            body: ttsResponse,
            isBase64Encoded: true
        };

    } catch (error) {
        console.error('TTS Error:', error);
        return {
            statusCode: 500,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                error: 'Failed to generate TTS audio',
                details: error.message 
            })
        };
    }
};

async function generateTTS(text, voice, rate, volume) {
    try {
        // Method 1: Try using Google Translate TTS (free, no API key needed)
        const googleTTS = `https://translate.google.com/translate_tts?ie=UTF-8&tl=${voice.split('-')[0]}&client=tw-ob&q=${encodeURIComponent(text)}`;
        
        const response = await axios.get(googleTTS, {
            responseType: 'arraybuffer'
        });

        // Convert to base64 for Netlify Functions
        return Buffer.from(response.data).toString('base64');

    } catch (error) {
        console.error('Google TTS failed:', error);
        
        // Method 2: Fallback to ResponsiveVoice (requires API key in production)
        // You can sign up for a free API key at https://responsivevoice.org/
        throw new Error('TTS service unavailable');
    }
}
