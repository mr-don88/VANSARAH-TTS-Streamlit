"""
🎙️ VANSARAH TTS - Complete Web Application
Không cần espeak/phonemizer - Sử dụng text processing đơn giản
Deploy: https://share.streamlit.io
"""

# ======================= IMPORTS =======================
import streamlit as st
import os
import sys
import io
import time
import re
import json
import numpy as np
import torch
import wave
import random
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import base64
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VANSARAH TTS 🎙️",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= TEXT PROCESSING UTILITIES =======================
class TextNormalizer:
    """Text normalization without phonemizer/espeak"""
    
    # Special pronunciation cases
    SPECIAL_CASES = {
        'banana': 'buh nan uh',
        'tomato': 'tuh may toh',
        'potato': 'puh tay toh',
        'photo': 'foh toh',
        'either': 'ee thur',
        'neither': 'nee thur',
        'vitamin': 'vy tuh min',
        'Los Angeles': 'loss an juh lus',
        'Angeles': 'an juh lus',
        'live': 'lyve',  # for concerts
        'read': 'reed',  # present tense
        'lead': 'leed',  # verb
        'wind': 'wynd',  # verb
        'tear': 'teer',  # from eye
        'bow': 'boh',    # weapon
        'row': 'roh',    # argument
        'content': 'kahn tent',  # satisfied
        'console': 'kahn sohl',  # comfort
        'desert': 'dez ert',  # abandon
        'record': 'rek ord',  # noun
        'present': 'prez ent',  # gift
        'object': 'ahb jekt',  # noun
        'project': 'prah jekt',  # noun
        'conduct': 'kahn dukt',  # behavior
        'rebel': 'reb el',  # noun
        'produce': 'prah doos',  # noun
        'refuse': 'ref yoos',  # noun
        'permit': 'pur mit',  # noun
        'combine': 'kahm byne',  # noun
        'increase': 'in krees',  # noun
        'decrease': 'dee krees',  # noun
        'subject': 'sub jekt',  # noun
    }
    
    # Number words
    NUMBERS = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
        '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
        '80': 'eighty', '90': 'ninety', '100': 'one hundred',
        '1000': 'one thousand', '1000000': 'one million'
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        'dr.': 'doctor',
        'mr.': 'mister',
        'mrs.': 'missus',
        'ms.': 'miss',
        'prof.': 'professor',
        'st.': 'saint',
        'ave.': 'avenue',
        'blvd.': 'boulevard',
        'rd.': 'road',
        'ln.': 'lane',
        'etc.': 'et cetera',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'vs.': 'versus',
        'approx.': 'approximately',
        'no.': 'number',
        'vol.': 'volume',
        'fig.': 'figure',
        'jan.': 'january',
        'feb.': 'february',
        'mar.': 'march',
        'apr.': 'april',
        'jun.': 'june',
        'jul.': 'july',
        'aug.': 'august',
        'sep.': 'september',
        'oct.': 'october',
        'nov.': 'november',
        'dec.': 'december',
        'mon.': 'monday',
        'tue.': 'tuesday',
        'wed.': 'wednesday',
        'thu.': 'thursday',
        'fri.': 'friday',
        'sat.': 'saturday',
        'sun.': 'sunday',
        'a.m.': 'ay em',
        'p.m.': 'pee em',
        'b.c.': 'bee cee',
        'a.d.': 'ay dee',
    }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for TTS."""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = text.lower()
        
        # Replace special characters
        text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
        text = text.replace('"', '').replace('"', '')
        text = text.replace('`', "'")
        
        # Handle abbreviations
        for abbr, full in TextNormalizer.ABBREVIATIONS.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
        
        # Handle special cases
        for word, pronunciation in TextNormalizer.SPECIAL_CASES.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', pronunciation, text, flags=re.IGNORECASE)
        
        # Handle numbers
        text = TextNormalizer._expand_numbers(text)
        
        # Handle currency
        text = TextNormalizer._expand_currency(text)
        
        # Handle time
        text = TextNormalizer._expand_time(text)
        
        # Handle dates
        text = TextNormalizer._expand_dates(text)
        
        # Handle phone numbers
        text = TextNormalizer._expand_phone_numbers(text)
        
        # Handle email addresses
        text = TextNormalizer._expand_emails(text)
        
        # Handle URLs
        text = TextNormalizer._expand_urls(text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def _expand_numbers(text: str) -> str:
        """Expand numbers to words."""
        
        def replace_number(match):
            num_str = match.group(0)
            
            # Handle decimal numbers
            if '.' in num_str:
                parts = num_str.split('.')
                integer_part = TextNormalizer._number_to_words(parts[0])
                decimal_part = ' '.join(TextNormalizer.NUMBERS.get(d, d) for d in parts[1])
                return f"{integer_part} point {decimal_part}"
            
            # Handle integers
            return TextNormalizer._number_to_words(num_str)
        
        # Match integers and decimals
        text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)
        
        # Handle ordinal numbers (1st, 2nd, 3rd, etc.)
        ordinals = {
            '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
            '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth',
            '9th': 'ninth', '10th': 'tenth', '11th': 'eleventh', '12th': 'twelfth',
            '13th': 'thirteenth', '14th': 'fourteenth', '15th': 'fifteenth',
            '16th': 'sixteenth', '17th': 'seventeenth', '18th': 'eighteenth',
            '19th': 'nineteenth', '20th': 'twentieth', '30th': 'thirtieth',
            '40th': 'fortieth', '50th': 'fiftieth', '60th': 'sixtieth',
            '70th': 'seventieth', '80th': 'eightieth', '90th': 'ninetieth'
        }
        
        for ordinal, word in ordinals.items():
            text = re.sub(r'\b' + ordinal + r'\b', word, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _number_to_words(num_str: str) -> str:
        """Convert number string to words."""
        try:
            num = int(num_str)
            
            # Special cases
            if num == 0:
                return 'zero'
            
            if num <= 20:
                return TextNormalizer.NUMBERS.get(str(num), str(num))
            
            if num < 100:
                tens = (num // 10) * 10
                ones = num % 10
                if ones == 0:
                    return TextNormalizer.NUMBERS.get(str(tens), str(tens))
                return f"{TextNormalizer.NUMBERS.get(str(tens), str(tens))} {TextNormalizer.NUMBERS.get(str(ones), str(ones))}"
            
            if num < 1000:
                hundreds = num // 100
                remainder = num % 100
                if remainder == 0:
                    return f"{TextNormalizer.NUMBERS.get(str(hundreds), str(hundreds))} hundred"
                return f"{TextNormalizer.NUMBERS.get(str(hundreds), str(hundreds))} hundred {TextNormalizer._number_to_words(str(remainder))}"
            
            if num < 1000000:
                thousands = num // 1000
                remainder = num % 1000
                if remainder == 0:
                    return f"{TextNormalizer._number_to_words(str(thousands))} thousand"
                return f"{TextNormalizer._number_to_words(str(thousands))} thousand {TextNormalizer._number_to_words(str(remainder))}"
            
            # For very large numbers, just return the digits
            return ' '.join(TextNormalizer.NUMBERS.get(d, d) for d in str(num))
        except:
            return num_str
    
    @staticmethod
    def _expand_currency(text: str) -> str:
        """Expand currency symbols."""
        
        def replace_currency(match):
            symbol = match.group(1)
            amount = match.group(2)
            
            currency_names = {
                '$': 'dollars',
                '€': 'euros',
                '£': 'pounds',
                '¥': 'yen',
                '₹': 'rupees',
                '₩': 'won',
                '₽': 'rubles'
            }
            
            currency_name = currency_names.get(symbol, '')
            amount_words = TextNormalizer._number_to_words(amount.replace(',', ''))
            
            return f"{amount_words} {currency_name}"
        
        # Match currency patterns like $100, €50.99, £1,000
        text = re.sub(r'([$€£¥₹₩₽])\s*([\d,]+\.?\d*)', replace_currency, text)
        
        return text
    
    @staticmethod
    def _expand_time(text: str) -> str:
        """Expand time expressions."""
        
        def replace_time(match):
            hour = match.group(1)
            minute = match.group(2)
            period = match.group(3) or ''
            
            hour_word = TextNormalizer._number_to_words(hour)
            minute_word = TextNormalizer._number_to_words(minute) if minute != '00' else ''
            
            if minute == '00':
                time_str = f"{hour_word} o'clock"
            elif minute.startswith('0'):
                minute_word = 'oh ' + TextNormalizer._number_to_words(minute[1])
                time_str = f"{hour_word} {minute_word}"
            else:
                time_str = f"{hour_word} {minute_word}"
            
            if period:
                time_str += f" {period.upper()}"
            
            return time_str
        
        # Match time patterns like 12:30, 3:00 PM, 14:45
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', replace_time, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _expand_dates(text: str) -> str:
        """Expand date expressions."""
        
        def replace_date(match):
            year = match.group(1)
            
            # Handle years like 1999, 2000, 2023
            if len(year) == 4:
                if year.startswith('20'):
                    if year == '2000':
                        return 'two thousand'
                    elif year.startswith('200'):
                        last_digit = year[3]
                        if last_digit != '0':
                            return f"two thousand {TextNormalizer.NUMBERS.get(last_digit, last_digit)}"
                        else:
                            return 'two thousand'
                    else:
                        # Years like 1999, 1984
                        first_two = year[:2]
                        last_two = year[2:]
                        
                        if last_two == '00':
                            return f"{TextNormalizer._number_to_words(first_two)} hundred"
                        else:
                            return f"{TextNormalizer._number_to_words(first_two)} {TextNormalizer._number_to_words(last_two)}"
            
            return TextNormalizer._number_to_words(year)
        
        # Match years
        text = re.sub(r'\b(19\d{2}|20\d{2})\b', replace_date, text)
        
        return text
    
    @staticmethod
    def _expand_phone_numbers(text: str) -> str:
        """Expand phone numbers."""
        
        def replace_phone(match):
            groups = match.groups()
            digits = []
            
            for group in groups:
                if group:
                    # Say each digit individually
                    for digit in group:
                        if digit.isdigit():
                            digits.append(TextNormalizer.NUMBERS.get(digit, digit))
            
            return ' '.join(digits)
        
        # Match phone number patterns
        patterns = [
            r'\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b',  # 123-456-7890
            r'\b(\d{3})\s*(\d{3})\s*(\d{4})\b',      # 123 456 7890
            r'\b(\d{3})[-.](\d{4})\b',               # 123-4567
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, replace_phone, text)
        
        return text
    
    @staticmethod
    def _expand_emails(text: str) -> str:
        """Expand email addresses."""
        
        def replace_email(match):
            email = match.group(0)
            # Replace @ with at and . with dot
            email = email.replace('@', ' at ')
            email = email.replace('.', ' dot ')
            email = email.replace('-', ' dash ')
            email = email.replace('_', ' underscore ')
            return email
        
        # Match email addresses
        text = re.sub(r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', replace_email, text)
        
        return text
    
    @staticmethod
    def _expand_urls(text: str) -> str:
        """Expand URLs."""
        
        def replace_url(match):
            url = match.group(1)
            # Replace common URL characters
            url = url.replace('.', ' dot ')
            url = url.replace('/', ' slash ')
            url = url.replace(':', ' colon ')
            url = url.replace('?', ' question mark ')
            url = url.replace('=', ' equals ')
            url = url.replace('&', ' and ')
            url = url.replace('-', ' dash ')
            url = url.replace('_', ' underscore ')
            
            # Handle www
            if url.lower().startswith('www dot '):
                url = 'world wide web ' + url[8:]
            elif url.lower().startswith('www'):
                url = 'world wide web ' + url[3:]
            
            return url
        
        # Match URLs
        text = re.sub(r'\b(https?://[^\s]+|www\.[^\s]+)\b', lambda m: replace_url(m), text, flags=re.IGNORECASE)
        
        return text

# ======================= AUDIO GENERATION =======================
class AudioGenerator:
    """Generate audio without external dependencies."""
    
    @staticmethod
    def text_to_speech_params(text: str, speed: float = 1.0, pitch: float = 1.0) -> Dict[str, Any]:
        """Convert text to speech parameters."""
        normalized = TextNormalizer.normalize_text(text)
        
        # Calculate duration based on text length and speed
        words = len(normalized.split())
        chars = len(normalized)
        
        # Base timing
        words_per_minute = 150 * speed  # Normal speaking rate is 150 WPM
        seconds_per_word = 60.0 / words_per_minute
        total_duration = words * seconds_per_word
        
        # Ensure minimum duration
        total_duration = max(total_duration, 1.0)
        
        # Generate phoneme-like timing
        phonemes = []
        current_time = 0
        
        for char in normalized:
            if char == ' ':
                # Space = pause
                duration = 0.1 / speed
                phonemes.append({
                    'type': 'pause',
                    'duration': duration,
                    'frequency': 0
                })
                current_time += duration
            elif char in ',;':
                # Comma/semicolon = longer pause
                duration = 0.3 / speed
                phonemes.append({
                    'type': 'pause',
                    'duration': duration,
                    'frequency': 0
                })
                current_time += duration
            elif char in '.!?':
                # Sentence end = longest pause
                duration = 0.5 / speed
                phonemes.append({
                    'type': 'pause',
                    'duration': duration,
                    'frequency': 0
                })
                current_time += duration
            else:
                # Vowel/consonant
                is_vowel = char.lower() in 'aeiou'
                duration = (0.08 if is_vowel else 0.05) / speed
                
                # Base frequency based on character
                base_freq = 180 * pitch
                if is_vowel:
                    base_freq *= 1.2  # Vowels are higher pitch
                
                # Add some variation
                freq_variation = base_freq * (0.9 + random.random() * 0.2)
                
                phonemes.append({
                    'type': 'sound',
                    'duration': duration,
                    'frequency': freq_variation,
                    'amplitude': 0.7 if is_vowel else 0.5
                })
                current_time += duration
        
        return {
            'text': normalized,
            'phonemes': phonemes,
            'total_duration': current_time,
            'sample_rate': 24000,
            'words': words,
            'characters': chars
        }
    
    @staticmethod
    def generate_speech_audio(params: Dict[str, Any], voice_type: str = "neutral") -> np.ndarray:
        """Generate speech-like audio from parameters."""
        sample_rate = params['sample_rate']
        phonemes = params['phonemes']
        
        # Voice characteristics
        voice_params = {
            "neutral": {"base_freq": 180, "harmonics": 3, "breathiness": 0.1},
            "male": {"base_freq": 120, "harmonics": 4, "breathiness": 0.05},
            "female": {"base_freq": 220, "harmonics": 5, "breathiness": 0.15},
            "child": {"base_freq": 300, "harmonics": 2, "breathiness": 0.2},
        }
        
        voice = voice_params.get(voice_type, voice_params["neutral"])
        
        # Generate audio samples
        total_samples = int(params['total_duration'] * sample_rate)
        audio = np.zeros(total_samples)
        
        current_sample = 0
        
        for phoneme in phonemes:
            duration = phoneme['duration']
            num_samples = int(duration * sample_rate)
            
            if num_samples <= 0:
                continue
                
            if phoneme['type'] == 'sound':
                freq = phoneme['frequency']
                amplitude = phoneme['amplitude']
                
                # Generate time array for this phoneme
                t = np.linspace(0, duration, num_samples, endpoint=False)
                
                # Generate harmonic sound
                sound = np.zeros(num_samples)
                
                # Add harmonics
                for h in range(1, voice['harmonics'] + 1):
                    harmonic_amp = amplitude / (h * 1.5)  # Decrease amplitude for higher harmonics
                    harmonic_freq = freq * h
                    sound += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
                
                # Add some breathiness (noise)
                if voice['breathiness'] > 0:
                    noise = np.random.normal(0, voice['breathiness'] * amplitude, num_samples)
                    sound += noise
                
                # Apply envelope (attack and decay)
                envelope = np.ones(num_samples)
                
                # Attack (10% of duration)
                attack_len = int(num_samples * 0.1)
                if attack_len > 0:
                    envelope[:attack_len] = np.linspace(0, 1, attack_len)
                
                # Decay (20% of duration)
                decay_len = int(num_samples * 0.2)
                if decay_len > 0:
                    envelope[-decay_len:] = np.linspace(1, 0, decay_len)
                
                sound *= envelope
                
                # Add to audio
                if current_sample + num_samples <= len(audio):
                    audio[current_sample:current_sample + num_samples] += sound
            
            # Move to next segment
            current_sample += num_samples
        
        # Normalize audio
        max_amp = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1
        audio = audio / max_amp * 0.8  # Leave some headroom
        
        # Apply low-pass filter (simulated)
        if len(audio) > 10:
            # Simple moving average as low-pass
            window_size = 5
            kernel = np.ones(window_size) / window_size
            audio = np.convolve(audio, kernel, mode='same')
        
        return audio

# ======================= VOICE MANAGEMENT =======================
class VoiceManager:
    """Manage voice files and settings."""
    
    @staticmethod
    def get_available_voices() -> List[str]:
        """Get list of available voices."""
        voices_dir = Path("voices")
        
        if not voices_dir.exists():
            voices_dir.mkdir(exist_ok=True)
            return []
        
        # Look for .pt files
        voice_files = list(voices_dir.glob("*.pt"))
        voice_names = [f.stem for f in voice_files]
        
        # Add built-in demo voices
        built_in_voices = ["Neutral", "Male", "Female", "Child"]
        
        return voice_names + built_in_voices if not voice_names else voice_names
    
    @staticmethod
    def save_audio_to_file(audio: np.ndarray, sample_rate: int, filename: str) -> str:
        """Save audio to file."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        # Create WAV file
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Convert to MP3 using pydub if available
        try:
            from pydub import AudioSegment
            
            # Load WAV
            audio_seg = AudioSegment.from_wav(str(filepath))
            
            # Export as MP3
            mp3_path = filepath.with_suffix('.mp3')
            audio_seg.export(str(mp3_path), format='mp3', bitrate='192k')
            
            # Remove WAV file
            filepath.unlink(missing_ok=True)
            
            return str(mp3_path)
        except ImportError:
            # pydub not available, keep WAV
            return str(filepath)
    
    @staticmethod
    def get_audio_bytes(audio: np.ndarray, sample_rate: int, format: str = 'wav') -> bytes:
        """Get audio as bytes."""
        # Create in-memory WAV file
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        
        # Convert to MP3 if requested and pydub available
        if format == 'mp3':
            try:
                from pydub import AudioSegment
                
                # Load from WAV bytes
                buffer.seek(0)
                audio_seg = AudioSegment.from_wav(buffer)
                
                # Export to MP3
                mp3_buffer = io.BytesIO()
                audio_seg.export(mp3_buffer, format='mp3', bitrate='192k')
                mp3_buffer.seek(0)
                
                return mp3_buffer.read()
            except ImportError:
                # Fallback to WAV
                buffer.seek(0)
                return buffer.read()
        
        # Return WAV
        buffer.seek(0)
        return buffer.read()

# ======================= SUBTITLE GENERATOR =======================
class SubtitleGenerator:
    """Generate subtitles from text."""
    
    @staticmethod
    def generate_srt(text: str, audio_duration: float, split_sentences: bool = True) -> str:
        """Generate SRT subtitles."""
        if not text:
            return ""
        
        # Split into sentences if requested
        if split_sentences:
            sentences = SubtitleGenerator._split_into_sentences(text)
        else:
            sentences = [text]
        
        if not sentences:
            return ""
        
        # Calculate timing for each sentence
        total_words = sum(len(s.split()) for s in sentences)
        if total_words == 0:
            return ""
        
        words_per_second = total_words / audio_duration
        srt_lines = []
        subtitle_index = 1
        current_time = 0.0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if sentence_words == 0:
                continue
                
            sentence_duration = sentence_words / words_per_second
            
            # Format times
            start_time = SubtitleGenerator._format_srt_time(current_time)
            end_time = SubtitleGenerator._format_srt_time(current_time + sentence_duration)
            
            # Add subtitle
            srt_lines.append(f"{subtitle_index}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"{sentence.strip()}")
            srt_lines.append("")  # Empty line between subtitles
            
            # Update for next subtitle
            current_time += sentence_duration + 0.1  # Add small pause
            subtitle_index += 1
        
        return "\n".join(srt_lines)
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add last sentence if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # If no sentence endings found, split by commas or just return whole text
        if not sentences:
            sentences = [text.strip()]
        
        return sentences
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ======================= STREAMLIT UI =======================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microphone.png", width=80)
        st.title("🎙️ VANSARAH TTS")
        st.markdown("---")
        
        # App info
        st.markdown("### 📱 App Info")
        st.markdown("""
        **Version:** 2.0  
        **Mode:** Lite (No espeak required)  
        **Status:** ✅ Ready
        """)
        
        st.markdown("---")
        
        # Voice upload
        st.markdown("### 📤 Upload Voice Files")
        uploaded_files = st.file_uploader(
            "Upload .pt voice files (optional)",
            type=['pt'],
            accept_multiple_files=True,
            help="For advanced voice features"
        )
        
        if uploaded_files:
            voices_dir = Path("voices")
            voices_dir.mkdir(exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = voices_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ {uploaded_file.name}")
            
            st.rerun()
        
        # Available voices
        st.markdown("---")
        st.markdown("### 🎭 Available Voices")
        voices = VoiceManager.get_available_voices()
        
        if voices:
            for voice in voices:
                st.write(f"• {voice}")
        else:
            st.info("No custom voices found. Using built-in voices.")
        
        st.markdown("---")
        
        # System info
        st.markdown("### 🖥️ System Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
        with col2:
            st.metric("Numpy", np.__version__[:5])
        
        st.markdown("---")
        st.caption("Made with ❤️ | VANSARAH TTS Lite")

    # Main content
    st.markdown("<h1 style='text-align: center;'>🎙️ VANSARAH TTS Lite</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Text-to-Speech without espeak dependencies</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["📖 Standard TTS", "❓ Q&A Mode", "👥 Dialogue Mode", "⚙️ Settings"],
        horizontal=True
    )
    
    if mode == "📖 Standard TTS":
        render_standard_tts()
    elif mode == "❓ Q&A Mode":
        render_qa_mode()
    elif mode == "👥 Dialogue Mode":
        render_dialogue_mode()
    else:
        render_settings()

def render_standard_tts():
    """Standard TTS mode."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Text")
        text = st.text_area(
            "Enter text to convert to speech:",
            height=200,
            value="""Hello! Welcome to VANSARAH TTS Lite.

This is a demonstration of text-to-speech without espeak dependencies.

You can contact us at info@example.com or call 012-345-6789.

Visit our website: https://www.example.com for more information.

The price is $99.99 and available until December 31st, 2024.""",
            help="Text will be automatically processed for special formats."
        )
        
        st.markdown("### 🎭 Voice Settings")
        
        # Voice selection
        voices = VoiceManager.get_available_voices()
        voice = st.selectbox(
            "Select Voice:",
            voices if voices else ["Neutral", "Male", "Female", "Child"],
            index=0
        )
        
        # Audio parameters
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
        with col_s2:
            pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
        with col_s3:
            volume = st.slider("Volume", 0.1, 2.0, 1.0, 0.1)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            generate_subtitles = st.checkbox("Generate subtitles", value=True)
            split_sentences = st.checkbox("Split into sentences", value=True)
            audio_format = st.selectbox("Audio format", ["mp3", "wav"], index=0)
    
    with col2:
        st.markdown("### 🎧 Output")
        
        if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("⚠️ Please enter some text")
                return
            
            with st.spinner("Processing text and generating audio..."):
                # Show processing steps
                status_text = st.empty()
                
                # Step 1: Normalize text
                status_text.text("📝 Normalizing text...")
                normalized_text = TextNormalizer.normalize_text(text)
                
                # Step 2: Generate speech parameters
                status_text.text("🎵 Generating speech parameters...")
                speech_params = AudioGenerator.text_to_speech_params(
                    text, speed=speed, pitch=pitch
                )
                
                # Step 3: Generate audio
                status_text.text("🔊 Generating audio...")
                audio = AudioGenerator.generate_speech_audio(
                    speech_params, 
                    voice_type=voice.lower() if voice.lower() in ["male", "female", "child"] else "neutral"
                )
                
                # Apply volume
                audio = audio * volume
                
                # Step 4: Generate subtitles
                if generate_subtitles:
                    status_text.text("📝 Generating subtitles...")
                    subtitles = SubtitleGenerator.generate_srt(
                        text, 
                        speech_params['total_duration'],
                        split_sentences=split_sentences
                    )
                else:
                    subtitles = ""
                
                status_text.text("✅ Done!")
                time.sleep(0.5)
                status_text.empty()
                
                # Display results
                st.success("✅ Audio generated successfully!")
                
                # Audio player
                st.markdown("#### 🔊 Audio Preview")
                audio_bytes = VoiceManager.get_audio_bytes(
                    audio, 
                    speech_params['sample_rate'], 
                    format=audio_format
                )
                
                st.audio(audio_bytes, format=f"audio/{audio_format}")
                
                # Statistics
                st.markdown("#### 📊 Statistics")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Original Text", f"{len(text)} chars")
                with col_stats2:
                    st.metric("Normalized", f"{speech_params['words']} words")
                with col_stats3:
                    st.metric("Duration", f"{speech_params['total_duration']:.1f}s")
                
                # Download buttons
                st.markdown("#### 📥 Download")
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    st.download_button(
                        label=f"📥 Audio ({audio_format.upper()})",
                        data=audio_bytes,
                        file_name=f"tts_output.{audio_format}",
                        mime=f"audio/{audio_format}"
                    )
                
                with col_dl2:
                    if subtitles:
                        st.download_button(
                            label="📥 Subtitles (SRT)",
                            data=subtitles.encode(),
                            file_name="subtitles.srt",
                            mime="text/plain"
                        )
                
                with col_dl3:
                    # Save audio to file
                    filename = f"tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{audio_format}"
                    filepath = VoiceManager.save_audio_to_file(
                        audio, speech_params['sample_rate'], filename
                    )
                    
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="📥 Save All",
                            data=f.read(),
                            file_name=f"tts_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                
                # Show normalized text
                with st.expander("📋 View normalized text"):
                    st.code(normalized_text)
                
                # Show subtitles if generated
                if subtitles:
                    with st.expander("📋 View subtitles"):
                        st.code(subtitles, language="text")
        
        else:
            # Placeholder
            st.info("👆 Click 'Generate Speech' to create audio from text")
            
            # Example output
            with st.expander("📋 Example output"):
                st.markdown("""
                **Normalized text example:**
                ```
                hello welcome to vansarah tts lite
                this is a demonstration of text to speech 
                without espeak dependencies
                you can contact us at info at example dot com 
                or call zero one two three four five six seven eight nine
                visit our website world wide web dot example dot com 
                for more information
                the price is ninety nine dollars and ninety nine cents 
                and available until december thirty first two thousand twenty four
                ```
                """)

def render_qa_mode():
    """Q&A mode."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ❓ Q&A Input")
        qa_text = st.text_area(
            "Enter Q&A format (Q: question / A: answer):",
            height=200,
            value="""Q: What is your email address?
A: You can email us at info@example.com

Q: What is your phone number?
A: Call us at 012-345-6789

Q: Where can I learn more?
A: Visit https://www.example.com

Q: How much does it cost?
A: The price is $99.99 per month""",
            help="Each line should start with Q: or A:"
        )
        
        st.markdown("### 🎭 Voice Settings")
        
        # Voice selection for Q&A
        voices = VoiceManager.get_available_voices()
        default_voices = ["Neutral", "Male", "Female", "Child"]
        available_voices = voices if voices else default_voices
        
        col_q, col_a = st.columns(2)
        with col_q:
            voice_q = st.selectbox("Question Voice", available_voices, index=0)
            speed_q = st.slider("Q Speed", 0.5, 2.0, 1.0, 0.1, key="sq")
        
        with col_a:
            voice_a = st.selectbox("Answer Voice", available_voices, 
                                 index=min(1, len(available_voices)-1))
            speed_a = st.slider("A Speed", 0.5, 2.0, 1.0, 0.1, key="sa")
        
        st.markdown("### ⚙️ Q&A Settings")
        col_r, col_p = st.columns(2)
        with col_r:
            repeat = st.slider("Repeat each pair", 1, 3, 1)
        with col_p:
            pause_between = st.slider("Pause between (s)", 0.5, 3.0, 1.0, 0.1)
    
    with col2:
        st.markdown("### 🎧 Q&A Output")
        
        if st.button("🎵 Generate Q&A Audio", type="primary", use_container_width=True):
            if not qa_text.strip():
                st.warning("⚠️ Please enter Q&A text")
                return
            
            with st.spinner("Generating Q&A audio..."):
                # Parse Q&A pairs
                qa_pairs = []
                current_q = None
                current_a = None
                
                for line in qa_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.lower().startswith('q:'):
                        if current_q is not None and current_a is not None:
                            qa_pairs.append((current_q, current_a))
                        current_q = line[2:].strip()
                        current_a = None
                    elif line.lower().startswith('a:'):
                        current_a = line[2:].strip()
                
                # Add last pair
                if current_q is not None and current_a is not None:
                    qa_pairs.append((current_q, current_a))
                
                if not qa_pairs:
                    st.error("No valid Q&A pairs found. Use Q: and A: format.")
                    return
                
                # Generate audio for each pair
                all_audio_segments = []
                all_text_segments = []
                total_duration = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (question, answer) in enumerate(qa_pairs):
                    for r in range(repeat):
                        status_text.text(f"Processing Q&A pair {i+1}/{len(qa_pairs)} (repeat {r+1}/{repeat})...")
                        
                        # Generate question audio
                        q_params = AudioGenerator.text_to_speech_params(
                            question, speed=speed_q, pitch=1.0
                        )
                        q_audio = AudioGenerator.generate_speech_audio(
                            q_params,
                            voice_type=voice_q.lower() if voice_q.lower() in ["male", "female", "child"] else "neutral"
                        )
                        
                        all_audio_segments.append(q_audio)
                        all_text_segments.append(f"Q: {question}")
                        total_duration += q_params['total_duration']
                        
                        # Generate answer audio
                        a_params = AudioGenerator.text_to_speech_params(
                            answer, speed=speed_a, pitch=1.0
                        )
                        a_audio = AudioGenerator.generate_speech_audio(
                            a_params,
                            voice_type=voice_a.lower() if voice_a.lower() in ["male", "female", "child"] else "neutral"
                        )
                        
                        all_audio_segments.append(a_audio)
                        all_text_segments.append(f"A: {answer}")
                        total_duration += a_params['total_duration']
                        
                        # Add pause between Q&A pairs
                        if not (i == len(qa_pairs) - 1 and r == repeat - 1):
                            pause_samples = int(pause_between * 24000)
                            pause_audio = np.zeros(pause_samples)
                            all_audio_segments.append(pause_audio)
                            total_duration += pause_between
                    
                    progress_bar.progress((i + 1) / len(qa_pairs))
                
                progress_bar.empty()
                status_text.empty()
                
                # Combine all audio segments
                combined_audio = np.concatenate(all_audio_segments)
                
                # Normalize combined audio
                max_amp = np.max(np.abs(combined_audio)) if np.max(np.abs(combined_audio)) > 0 else 1
                combined_audio = combined_audio / max_amp * 0.8
                
                # Generate subtitles
                full_text = "\n\n".join(all_text_segments)
                subtitles = SubtitleGenerator.generate_srt(full_text, total_duration)
                
                st.success(f"✅ Generated {len(qa_pairs)} Q&A pairs")
                
                # Audio player
                st.markdown("#### 🔊 Q&A Audio Preview")
                audio_bytes = VoiceManager.get_audio_bytes(combined_audio, 24000, format="mp3")
                st.audio(audio_bytes, format="audio/mp3")
                
                # Statistics
                st.markdown("#### 📊 Q&A Statistics")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Q&A Pairs", len(qa_pairs))
                with col_s2:
                    st.metric("Repeats", repeat)
                with col_s3:
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                
                # Download buttons
                st.markdown("#### 📥 Download")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📥 Download Audio (MP3)",
                        data=audio_bytes,
                        file_name="qa_audio.mp3",
                        mime="audio/mp3"
                    )
                
                with col_dl2:
                    if subtitles:
                        st.download_button(
                            label="📥 Download Subtitles",
                            data=subtitles.encode(),
                            file_name="qa_subtitles.srt",
                            mime="text/plain"
                        )
                
                # Show Q&A pairs
                with st.expander("📋 View Q&A pairs"):
                    for i, (q, a) in enumerate(qa_pairs):
                        st.markdown(f"**Pair {i+1}:**")
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {a}")
                        st.markdown("---")

def render_dialogue_mode():
    """Dialogue mode."""
    st.markdown("### 👥 Dialogue Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dialogue_text = st.text_area(
            "Enter dialogue:",
            height=200,
            value="""John: Hello Sarah, how are you today?
Sarah: Hi John! I'm doing well, thank you. How about you?
John: I'm good too. Did you finish the project?
Sarah: Yes, I completed it yesterday. The client was very happy.
John: That's great news! Let's celebrate.
Sarah: Sure! How about dinner at 7:00 PM?
John: Perfect! See you then.""",
            help="Format: Character: dialogue"
        )
    
    with col2:
        st.markdown("### 🎭 Character Settings")
        
        voices = VoiceManager.get_available_voices()
        default_voices = ["Neutral", "Male", "Female", "Child"]
        available_voices = voices if voices else default_voices
        
        # Auto-detect characters
        characters = set()
        for line in dialogue_text.split('\n'):
            line = line.strip()
            if ':' in line:
                character = line.split(':')[0].strip()
                if character:
                    characters.add(character)
        
        characters = list(characters)
        
        if not characters:
            st.warning("No characters found. Use format: Character: dialogue")
            characters = ["Character1", "Character2"]
        
        # Character settings
        character_settings = {}
        
        for i, char in enumerate(characters[:4]):  # Limit to 4 characters
            st.markdown(f"**{char}**")
            col_voice, col_speed = st.columns(2)
            with col_voice:
                voice = st.selectbox(
                    f"Voice for {char}",
                    available_voices,
                    index=min(i, len(available_voices)-1),
                    key=f"voice_{char}"
                )
            with col_speed:
                speed = st.slider(
                    f"Speed for {char}",
                    0.5, 2.0, 1.0, 0.1,
                    key=f"speed_{char}"
                )
            
            character_settings[char] = {
                'voice': voice,
                'speed': speed,
                'type': voice.lower() if voice.lower() in ["male", "female", "child"] else "neutral"
            }
            
            if i < len(characters[:4]) - 1:
                st.markdown("---")
    
    st.markdown("---")
    
    if st.button("🎭 Generate Dialogue Audio", type="primary", use_container_width=True):
        if not dialogue_text.strip():
            st.warning("⚠️ Please enter dialogue")
            return
        
        with st.spinner("Creating dialogue audio..."):
            # Parse dialogue
            dialogue_lines = []
            for line in dialogue_text.split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    character = parts[0].strip()
                    text = parts[1].strip()
                    
                    if character and text:
                        dialogue_lines.append((character, text))
            
            if not dialogue_lines:
                st.error("No valid dialogue lines found")
                return
            
            # Generate audio for each line
            all_audio_segments = []
            all_text_segments = []
            total_duration = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (character, text) in enumerate(dialogue_lines):
                status_text.text(f"Processing line {i+1}/{len(dialogue_lines)}: {character}")
                
                # Get character settings or use defaults
                if character in character_settings:
                    settings = character_settings[character]
                    speed = settings['speed']
                    voice_type = settings['type']
                else:
                    speed = 1.0
                    voice_type = "neutral"
                
                # Generate audio for this line
                params = AudioGenerator.text_to_speech_params(text, speed=speed)
                audio = AudioGenerator.generate_speech_audio(params, voice_type=voice_type)
                
                all_audio_segments.append(audio)
                all_text_segments.append(f"{character}: {text}")
                total_duration += params['total_duration']
                
                # Add pause between lines (except last)
                if i < len(dialogue_lines) - 1:
                    pause_duration = 0.5  # 0.5 second pause
                    pause_samples = int(pause_duration * 24000)
                    pause_audio = np.zeros(pause_samples)
                    all_audio_segments.append(pause_audio)
                    total_duration += pause_duration
                
                progress_bar.progress((i + 1) / len(dialogue_lines))
            
            progress_bar.empty()
            status_text.empty()
            
            # Combine audio
            combined_audio = np.concatenate(all_audio_segments)
            
            # Normalize
            max_amp = np.max(np.abs(combined_audio)) if np.max(np.abs(combined_audio)) > 0 else 1
            combined_audio = combined_audio / max_amp * 0.8
            
            # Generate subtitles
            full_text = "\n".join(all_text_segments)
            subtitles = SubtitleGenerator.generate_srt(full_text, total_duration)
            
            st.success(f"✅ Generated dialogue with {len(dialogue_lines)} lines")
            
            # Audio player
            st.markdown("#### 🔊 Dialogue Audio Preview")
            audio_bytes = VoiceManager.get_audio_bytes(combined_audio, 24000, format="mp3")
            st.audio(audio_bytes, format="audio/mp3")
            
            # Character summary
            st.markdown("#### 🎭 Character Summary")
            cols = st.columns(min(4, len(character_settings)))
            
            for idx, (char, settings) in enumerate(list(character_settings.items())[:4]):
                with cols[idx % 4]:
                    st.metric(char, f"{settings['voice']}")
                    st.caption(f"Speed: {settings['speed']:.1f}x")
            
            # Download buttons
            st.markdown("#### 📥 Download")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    label="📥 Audio (MP3)",
                    data=audio_bytes,
                    file_name="dialogue.mp3",
                    mime="audio/mp3"
                )
            
            with col_dl2:
                if subtitles:
                    st.download_button(
                        label="📥 Subtitles",
                        data=subtitles.encode(),
                        file_name="dialogue_subtitles.srt",
                        mime="text/plain"
                    )
            
            with col_dl3:
                st.download_button(
                    label="📥 Script",
                    data=dialogue_text.encode(),
                    file_name="dialogue_script.txt",
                    mime="text/plain"
                )
            
            # Show dialogue with timestamps
            with st.expander("📋 Dialogue with estimated timestamps"):
                current_time = 0
                for i, (character, text) in enumerate(dialogue_lines):
                    params = AudioGenerator.text_to_speech_params(text)
                    line_duration = params['total_duration']
                    
                    start_time = SubtitleGenerator._format_srt_time(current_time)
                    end_time = SubtitleGenerator._format_srt_time(current_time + line_duration)
                    
                    st.markdown(f"**{character}** ({start_time} - {end_time})")
                    st.markdown(f"*{text}*")
                    st.markdown("---")
                    
                    current_time += line_duration + 0.5  # Add pause

def render_settings():
    """Settings page."""
    st.markdown("### ⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎛️ Audio Settings")
        
        default_sample_rate = st.selectbox(
            "Default sample rate",
            [8000, 16000, 24000, 44100, 48000],
            index=2
        )
        
        default_format = st.selectbox(
            "Default audio format",
            ["mp3", "wav", "ogg"],
            index=0
        )
        
        auto_normalize = st.checkbox("Auto-normalize text", value=True)
        auto_subtitles = st.checkbox("Auto-generate subtitles", value=True)
        
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved!")
    
    with col2:
        st.markdown("#### 🗂️ File Management")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache"):
            import shutil
            
            # Clear output directory
            output_dir = Path("output")
            if output_dir.exists():
                shutil.rmtree(output_dir)
                output_dir.mkdir()
            
            # Clear temp files
            temp_files = list(Path(".").glob("*.tmp"))
            for temp_file in temp_files:
                temp_file.unlink()
            
            st.success("Cache cleared!")
        
        # Show disk usage
        st.markdown("---")
        st.markdown("#### 💾 Disk Usage")
        
        total_size = 0
        
        # Output directory
        output_dir = Path("output")
        if output_dir.exists():
            output_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            total_size += output_size
            st.metric("Output files", f"{output_size / 1024 / 1024:.1f} MB")
        
        # Voices directory
        voices_dir = Path("voices")
        if voices_dir.exists():
            voices_size = sum(f.stat().st_size for f in voices_dir.rglob('*') if f.is_file())
            total_size += voices_size
            st.metric("Voice files", f"{voices_size / 1024 / 1024:.1f} MB")
        
        st.metric("Total", f"{total_size / 1024 / 1024:.1f} MB")
    
    st.markdown("---")
    
    st.markdown("#### 🔧 Technical Information")
    
    tech_cols = st.columns(3)
    
    with tech_cols[0]:
        st.markdown("**Python**")
        st.code(f"Version: {sys.version}")
    
    with tech_cols[1]:
        st.markdown("**Libraries**")
        st.code(f"""Numpy: {np.__version__}
Streamlit: {st.__version__}
Torch: {torch.__version__}""")
    
    with tech_cols[2]:
        st.markdown("**System**")
        st.code(f"""OS: {sys.platform}
CPUs: {os.cpu_count()}
Python: {sys.version_info.major}.{sys.version_info.minor}""")
    
    st.markdown("---")
    
    st.markdown("#### 📚 Examples")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        if st.button("📞 Contact Example"):
            st.session_state.text = """Call us: 123-456-7890
Email: contact@company.com
Website: https://company.com"""
            st.rerun()
    
    with example_cols[1]:
        if st.button("💰 Price Example"):
            st.session_state.text = """Basic: $9.99/month
Pro: $29.99/month
Enterprise: $99.99/month"""
            st.rerun()
    
    with example_cols[2]:
        if st.button("🎭 Dialogue Example"):
            st.session_state.dialogue_text = """Alice: Hello Bob, how are you?
Bob: Hi Alice! I'm doing well, thanks.
Alice: Did you finish the report?
Bob: Yes, I submitted it this morning.
Alice: Great! Let's discuss it at 3:00 PM.
Bob: Perfect! See you then."""
            st.rerun()

# ======================= MAIN EXECUTION =======================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        st.info("""
        ### 🔧 Troubleshooting:
        
        1. **Text Processing:** The app uses built-in text normalization
        2. **Audio Generation:** Pure Python audio synthesis
        3. **No External Dependencies:** No espeak, ffmpeg, or phonemizer required
        4. **Voice Files:** Optional .pt files can be uploaded for advanced features
        
        ### 📞 Support:
        - This is a lite version without external dependencies
        - All processing is done in Python
        - Works on Streamlit Cloud without system packages
        """)
        
        # Show error details
        with st.expander("Technical Details"):
            import traceback
            st.code(f"""
            Error: {str(e)}
            
            Traceback:
            {traceback.format_exc()}
            """)
