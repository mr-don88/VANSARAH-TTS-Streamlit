"""
🎙️ VANSARAH TTS - Complete Web Application
Tích hợp hoàn chỉnh TTS engine + Streamlit interface
Deploy: https://share.streamlit.io
"""

# ======================= IMPORTS & SETUP =======================
import streamlit as st
import os
import sys
import io
import time
import re
import json
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import timedelta
import wave
import random
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range, low_pass_filter, high_pass_filter

# Suppress warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

# Page config
st.set_page_config(
    page_title="VANSARAH TTS 🎙️",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= SPECIAL CASES =======================
SPECIAL_CASES = {
    # Add your special pronunciation cases here
    'example': 'ek-SAM-pul',
    'banana': 'buh-NAN-uh',
    'tomato': 'tuh-MAY-toh',
    'photo': 'FOH-toh',
    'either': 'EE-thur',
    'neither': 'NEE-thur',
    'vitamin': 'VAI-tuh-min',
    'Los Angeles': 'Loss AN-juh-luhs',
    'Angeles': 'AN-juh-luhs',
    'live': 'LYVE',  # for concerts
    'read': 'REED',  # present tense
    'read': 'RED',   # past tense (context handled separately)
    'lead': 'LEED',  # verb
    'lead': 'LED',   # metal
}

# ======================= TTS ENGINE CLASSES =======================
class Tokenizer:
    def __init__(self):
        self.VOCAB = self._get_vocab()
        self.special_cases = self._build_special_cases()
        self.special_regex = self._build_special_regex()
        self.abbreviation_patterns = self._build_abbreviation_patterns()
        
        try:
            from phonemizer import backend
            self.phonemizers = {
                'en-us': backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
            }
        except ImportError:
            self.phonemizers = {}
            st.warning("Phonemizer not available. Using fallback text processing.")
        
    def _fix_heteronyms(self, text: str) -> str:
        """Fix common English heteronyms by context."""
        # LIVE
        text = re.sub(r"\blive\s+(concert|show|event|music|performance|broadcast|session|version|album|gig)\b",
                      r"lyve \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(is|are|was|were|go|goes|going|went)\s+live\b",
                      r"\1 lyve", text, flags=re.IGNORECASE)
        text = re.sub(r"\bto live\b", "to liv", text, flags=re.IGNORECASE)
        text = re.sub(r"\blive(s|d|ing)?\b", r"liv\1", text, flags=re.IGNORECASE)
        
        # LEAD
        text = re.sub(r"\blead (pipe|metal|balloon|paint|weight)\b",
                      r"led \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\blead\b", "leed", text, flags=re.IGNORECASE)
        
        # READ
        text = re.sub(r'\bWhat did you read', 'What did you reed', text, flags=re.IGNORECASE)
        text = re.sub(r"\bto\s+read\b", "to reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdidn't\s+read\b", "didn't reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bDo\s+(you|we|they)\s+read\b", r"Do \1 reed", text, flags=re.IGNORECASE)
        
        # Past tense handling
        past_auxiliaries = r'\b(had|was|were|have|has|haven\'t|hasn\'t|hadn\'t|wasn\'t|weren\'t)'
        text = re.sub(rf'{past_auxiliaries}\s+([^.!?]*?)\bread\b', r'\1 \2red', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _get_vocab():
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»" '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        return {symbol: index for index, symbol in enumerate(symbols)}

    def _build_special_cases(self) -> Dict[str, str]:
        cases = SPECIAL_CASES.copy()
        SILENT = " " + "\u200B" + " "
        
        final_cases = {}
        for original, processed in cases.items():
            if ("'" in original or 
                not processed.isupper() or 
                len(processed) <= 1 or 
                processed == "I" or
                ' ' in processed):
                final_cases[original] = processed
            else:
                final_cases[original] = f"{SILENT}{processed}{SILENT}"
        
        return final_cases

    def _build_special_regex(self):
        words = sorted(self.special_cases.keys(), key=len, reverse=True)
        return re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', flags=re.IGNORECASE)

    def apply_special_cases(self, text: str) -> str:
        def repl(match):
            word = match.group(0)
            return self.special_cases.get(word.lower(), word)
        
        return self.special_regex.sub(repl, text)

    def _build_abbreviation_patterns(self):
        return {
            # Time
            r"\ba\.m\.?\b": "AM",
            r"\bp\.m\.?\b": "PM",      
            
            # Common abbreviations
            r"\betc\.?\b": "et cetera ",
            r"\bvs\.?\b": "versus ",
            r"\be\.g\.\b": "for example ",
            r"\bi\.e\.\b": "that is ",
            r"\bDr\.?\b": "Doctor ",
            r"\bMr\.?\b": "Mister ",
            r"\bMrs\.?\b": "Misses ",
            r"\bMs\.?\b": "Miss ",
            r"\bProf\.?\b": "Professor ",
            r"\bSt\.?\b": "Saint ",
            
            # World Wars
            r"\bWorld War I\b": "World War One",
            r"\bWorld War II\b": "World War Two",
            
            # Roman numerals for names
            r"\bHenry I\b": "Henry the First",
            r"\bHenry II\b": "Henry the Second",
            r"\bHenry VIII\b": "Henry the Eighth",
            r"\bElizabeth I\b": "Elizabeth the First",
            r"\bElizabeth II\b": "Elizabeth the Second",
            
            # Ordinal numbers
            r"\b1st\b": "first ",
            r"\b2nd\b": "second ",
            r"\b3rd\b": "third ",
            r"\b4th\b": "fourth ",
            r"\b5th\b": "fifth ",
            r"\b10th\b": "tenth ",
            r"\b20th\b": "twentieth ",
            
            # Cardinal numbers 0-20
            r"\b0\b": "zero ",
            r"\b1\b": "one ",
            r"\b2\b": "two ",
            r"\b3\b": "three ",
            r"\b4\b": "four ",
            r"\b5\b": "five ",
            r"\b10\b": "ten ",
            r"\b11\b": "eleven ",
            r"\b12\b": "twelve ",
            r"\b20\b": "twenty ",
        }

    def _add_invisible_space_to_conjunctions(self, text: str) -> str:
        ZWS = "\u200B"
        conjunctions = [
            "and", "or", "but", "so", "because", "then", "yet", "nor",
            "although", "though", "even though", "however", "nevertheless", "nonetheless", "still",
            "therefore", "thus", "hence", "consequently", "accordingly", "as", "since",
            "moreover", "furthermore", "besides", "also", "in addition",
            "meanwhile", "afterwards", "afterward", "later", "before", "until", "unless", "while", "when", "once",
            "otherwise", "instead", "rather", "whether"
        ]
        pattern = r'\b(' + '|'.join(conjunctions) + r')\b'
        return re.sub(pattern, lambda m: f"{ZWS}{m.group(0)}", text, flags=re.IGNORECASE)

    def process_text(self, text: str) -> str:
        text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
        text = text.replace('"', '')
        
        abbreviation_dot_map = {
            r"\bMr\.": "mr",
            r"\bMrs\.": "mrs",
            r"\bMs\.": "ms",
            r"\bDr\.": "dr",
            r"\bProf\.": "prof",             
            r"\bSt\.": "st",
            r"\bAve\.": "ave",
            r"\bJr\.": "jr",
            r"\bSr\.": "sr",
            r"\bvs\.": "vs",
            r"\betc\.": "etc",           
        }
        for pattern, repl in abbreviation_dot_map.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        text = text.replace(';', '.').replace(':', '.')
        
        # American English fixes
        text = re.sub(r'\bphotos\b', 'foh-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\bphoto\b', 'foh-toh', text, flags=re.IGNORECASE)
        text = re.sub(r'\btomatoes\b', 'tuh-may-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\btomato\b', 'tuh-may-toh', text, flags=re.IGNORECASE)
        text = re.sub(r'\bneither\b', 'nee-thur', text, flags=re.IGNORECASE)
        text = re.sub(r'\beither\b', 'ee-thur', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvitamin\b', 'vai-tuh-min', text, flags=re.IGNORECASE)

        text = self._fix_heteronyms(text)

        for pattern, replacement in self.abbreviation_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = self.apply_special_cases(text)

        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s*\n\s*', '. ', text)
        text = text.strip()

        text = self._add_invisible_space_to_conjunctions(text)

        return text

    def tokenize(self, phonemes: str) -> List[int]:
        return [self.VOCAB[x] for x in phonemes if x in self.VOCAB]

    def phonemize(self, text: str, lang: str = 'en-us', normalize: bool = True) -> str:
        if normalize:
            text = self.process_text(text)

        if lang not in self.phonemizers:
            lang = 'en-us'

        if not self.phonemizers:
            # Fallback: return processed text
            return text

        phonemes = self.phonemizers[lang].phonemize([text])
        phonemes = phonemes[0] if phonemes else ''

        replacements = {
            'kəkˈoːɹoʊ': 'kˈoʊkəɹoʊ',
            'kəkˈɔːɹəʊ': 'kˈəʊkəɹəʊ',
            'ʲ': 'j',
            'r': 'ɹ',
            'x': 'k',
            'ɬ': 'l',
        }
        for old, new in replacements.items():
            phonemes = phonemes.replace(old, new)

        phonemes = ''.join(filter(lambda p: p in self.VOCAB, phonemes))
        return phonemes.strip()

# ======================= TTS MODEL =======================
class TTSModel:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.models = {}
        self.tokenizer = Tokenizer()
        self.voice_cache = {}
        self.voice_files = self._discover_voices()
        
        try:
            # Try to import vansarah
            from vansarah import KModel, KPipeline
            if self.use_cuda:
                self.models['cuda'] = torch.compile(KModel().to('cuda').eval(), mode='max-autotune')
            self.models['cpu'] = KModel().to('cpu').eval()
            
            self.pipelines = {
                'a': KPipeline(lang_code='a', model=False),
                'b': KPipeline(lang_code='b', model=False)
            }
            self.vansarah_available = True
        except ImportError:
            st.warning("⚠️ VANSARAH library not found. Using demo mode.")
            self.vansarah_available = False
            self.pipelines = {}
    
    def _discover_voices(self):
        """Discover voice files in voices folder."""
        voice_files = {}
        voices_dir = Path("voices")
        
        if not voices_dir.exists():
            voices_dir.mkdir(exist_ok=True)
            st.info(f"Created voices directory: {voices_dir.absolute()}")
            return voice_files
            
        for file in voices_dir.glob("*.pt"):
            voice_name = file.stem
            voice_files[voice_name] = str(file)
        
        return voice_files

    def get_voice_list(self):
        """Get list of available voices."""
        return list(self.voice_files.keys())

# Initialize model globally
@st.cache_resource
def load_tts_model():
    """Load TTS model once."""
    try:
        model = TTSModel()
        return model
    except Exception as e:
        st.error(f"Failed to load TTS model: {e}")
        return None

# ======================= TEXT PROCESSOR =======================
class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = TextProcessor._process_special_cases(text)
        
        re_tab = re.compile(r'[\r\t]')
        re_spaces = re.compile(r' +')
        re_punctuation = re.compile(r'(\s)([,.!?])')
        
        text = re_tab.sub(' ', text)
        text = re_spaces.sub(' ', text)
        text = re_punctuation.sub(r'\2', text)
        return text.strip()

    @staticmethod
    def _process_special_cases(text: str) -> str:
        """Process emails, phones, URLs, etc."""
        # Emails
        text = re.sub(
            r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b',
            lambda m: m.group(0).replace('@', ' at ').replace('.', ' dot '),
            text
        )
        
        # URLs
        text = re.sub(
            r'\bhttps?://[^\s]+',
            lambda m: m.group(0).replace('.', ' dot ').replace('/', ' slash '),
            text
        )
        
        # Phone numbers
        text = re.sub(
            r'\b(\d{3})[-. ]?(\d{3})[-. ]?(\d{4})\b',
            lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}",
            text
        )
        
        # Currency
        text = re.sub(
            r'\$(\d+(?:\.\d+)?)',
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} dollars",
            text
        )
        
        return text

    @staticmethod
    def _number_to_words(number: str) -> str:
        """Convert number to words (simplified)."""
        number_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
        }
        return number_map.get(number, number)

    def split_sentences(self, text: str, max_chars: int = 250) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
            "a.m.", "p.m.", "etc.", "e.g.", "i.e.", "vs."
        ]
        
        protected = text
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<ABB>"))
        
        protected = re.sub(r"\b([A-Z])\.", r"\1<ABB>", protected)
        
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        
        sentences = []
        for s in raw_sentences:
            s = s.replace("<ABB>", ".").strip()
            if not s:
                continue
            
            if len(s) > max_chars:
                chunks = self._split_long_sentence(s, max_chars)
                sentences.extend(chunks)
            else:
                sentences.append(s)
        
        return sentences

    def _split_long_sentence(self, text: str, max_chars: int) -> List[str]:
        """Split a long sentence."""
        sentences = []
        current_text = text
        
        while current_text and len(current_text) > max_chars:
            split_pos = -1
            
            for punct in ['.', '!', '?', ';', ':']:
                punct_pos = current_text.rfind(punct, 0, max_chars)
                if punct_pos > 0:
                    split_pos = punct_pos + 1
                    break
            
            if split_pos == -1:
                comma_pos = current_text.rfind(',', 0, max_chars)
                if comma_pos > 0:
                    split_pos = comma_pos + 1
            
            if split_pos == -1:
                space_pos = current_text.rfind(' ', 0, max_chars)
                if space_pos > 0:
                    split_pos = space_pos + 1
                else:
                    split_pos = max_chars
            
            part = current_text[:split_pos].strip()
            if part:
                sentences.append(part)
            current_text = current_text[split_pos:].strip()
        
        if current_text:
            sentences.append(current_text)
        
        return sentences

    @staticmethod
    def parse_dialogues(text: str, prefixes: List[str]) -> List[Tuple[str, str]]:
        """Parse dialogue text."""
        dialogues = []
        current = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            found_prefix = None
            for prefix in prefixes:
                if line.lower().startswith(prefix.lower() + ':'):
                    found_prefix = prefix
                    break
                    
            if found_prefix:
                if current:
                    dialogues.append((current[0], current[1]))
                
                speaker = found_prefix
                content = line[len(found_prefix)+1:].strip()
                current = (speaker, content)
            elif current:
                current = (current[0], current[1] + ' ' + line)
                
        if current:
            dialogues.append((current[0], current[1]))
            
        return dialogues

# ======================= AUDIO PROCESSOR =======================
class AudioProcessor:
    @staticmethod
    def enhance_audio(audio: np.ndarray, volume: float = 1.0, pitch: float = 1.0) -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        max_sample = np.max(np.abs(audio)) + 1e-8
        audio = (audio / max_sample) * 0.9 * volume
        
        audio = np.tanh(audio * 1.5) / 1.5
        
        try:
            audio_seg = AudioSegment(
                (audio * 32767).astype(np.int16).tobytes(),
                frame_rate=24000,
                sample_width=2,
                channels=1
            )
            
            if pitch != 1.0:
                audio_seg = audio_seg._spawn(
                    audio_seg.raw_data,
                    overrides={"frame_rate": int(audio_seg.frame_rate * pitch)}
                ).set_frame_rate(24000)
            
            return np.array(audio_seg.get_array_of_samples()) / 32768.0
        except:
            return audio

    @staticmethod
    def calculate_pause(text: str, pause_settings: Dict[str, int]) -> int:
        text = text.strip()
        if not text:
            return 0
            
        last_char = text[-1]
        return pause_settings.get(last_char, pause_settings['default_pause'])

# ======================= SUBTITLE GENERATOR =======================
class SubtitleGenerator:
    @staticmethod
    def generate_srt(audio_segments: List[AudioSegment], sentences: List[str], pause_settings: Dict[str, int]) -> str:
        """Generate SRT subtitles."""
        subtitles = []
        current_time_ms = 0
        srt_index = 1

        for i, (audio_seg, sentence) in enumerate(zip(audio_segments, sentences)):
            segment_duration_ms = len(audio_seg)
            
            # Simple split for long sentences
            if len(sentence) > 100:
                chunks = [sentence[:len(sentence)//2], sentence[len(sentence)//2:]]
            else:
                chunks = [sentence]
            
            total_chars = sum(len(chunk) for chunk in chunks)
            
            for j, chunk in enumerate(chunks):
                chunk_chars = len(chunk)
                chunk_duration_ms = int(segment_duration_ms * (chunk_chars / total_chars)) if total_chars > 0 else segment_duration_ms
                chunk_duration_ms = max(500, chunk_duration_ms)
                
                start_ms = current_time_ms
                end_ms = start_ms + chunk_duration_ms
                
                start_td = timedelta(milliseconds=start_ms)
                end_td = timedelta(milliseconds=end_ms)
                
                hours_start = int(start_td.total_seconds() // 3600)
                minutes_start = int((start_td.total_seconds() % 3600) // 60)
                seconds_start = int(start_td.total_seconds() % 60)
                milliseconds_start = int((start_td.total_seconds() - int(start_td.total_seconds())) * 1000)
                
                hours_end = int(end_td.total_seconds() // 3600)
                minutes_end = int((end_td.total_seconds() % 3600) // 60)
                seconds_end = int(end_td.total_seconds() % 60)
                milliseconds_end = int((end_td.total_seconds() - int(end_td.total_seconds())) * 1000)
                
                start_str = f"{hours_start:02d}:{minutes_start:02d}:{seconds_start:02d},{milliseconds_start:03d}"
                end_str = f"{hours_end:02d}:{minutes_end:02d}:{seconds_end:02d},{milliseconds_end:03d}"
                
                subtitles.append(f"{srt_index}\n{start_str} --> {end_str}\n{chunk.strip()}\n")
                srt_index += 1
                
                current_time_ms += chunk_duration_ms
            
            if i < len(audio_segments) - 1:
                pause_ms = AudioProcessor.calculate_pause(sentence, pause_settings)
                current_time_ms += pause_ms

        return "\n".join(subtitles)

# ======================= TTS GENERATOR =======================
class TTSGenerator:
    def __init__(self, model: TTSModel):
        self.model = model
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.tokenizer = Tokenizer()
        self.subtitle_generator = SubtitleGenerator()
        
        # Demo mode fallback
        self.demo_mode = not model.vansarah_available

    def generate_demo_audio(self, text: str, duration_sec: int = 3) -> Tuple[int, np.ndarray]:
        """Generate demo audio (sin wave)."""
        sample_rate = 24000
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        freq = 440
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        return sample_rate, audio

    def generate_audio(self, text: str, voice: str, speed: float, device: str,
                      volume: float = 1.0, pitch: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """Generate audio from text."""
        if self.demo_mode:
            # Demo mode - generate simple tone
            duration = len(text.split()) * 0.5  # 0.5 sec per word
            return self.generate_demo_audio(text, min(duration, 10))
        
        try:
            # Real TTS generation
            if voice not in self.model.voice_files:
                st.error(f"Voice {voice} not found")
                return None
            
            if voice not in self.model.voice_cache:
                voice_path = self.model.voice_files[voice]
                try:
                    voice_data = torch.load(voice_path, map_location='cpu')
                    pipeline = self.model.pipelines['a']
                    self.model.voice_cache[voice] = (pipeline, voice_data)
                except Exception as e:
                    st.error(f"Error loading voice: {e}")
                    return None
            else:
                pipeline, pack = self.model.voice_cache[voice]
            
            processed_text = self.tokenizer.process_text(text)
            
            for _, ps, _ in pipeline(processed_text, voice, speed):
                ref_s = pack[len(ps)-1]
                
                if device == 'cuda':
                    ps = ps.cuda()
                    ref_s = ref_s.cuda()
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    audio = self.model.models[device](ps, ref_s, speed).cpu().numpy()
                
                return (24000, self.audio_processor.enhance_audio(audio, volume, pitch))
                
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None
        
        return None

    def generate_story_audio(self, text: str, voice: str, speed: float, device: str,
                           pause_settings: Dict[str, int], volume: float = 1.0, 
                           pitch: float = 1.0, max_chars_per_segment: int = 250) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str]:
        """Generate audio for story/text."""
        clean_text = self.text_processor.clean_text(text)
        sentences = self.text_processor.split_sentences(clean_text, max_chars_per_segment)
        
        if not sentences:
            return None, "No content", ""
        
        speed_factor = max(0.5, min(2.0, speed))
        adjusted_pause_settings = {k: int(v / speed_factor) for k, v in pause_settings.items()}
        
        audio_segments = []
        pause_durations = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sentence in enumerate(sentences):
            status_text.text(f"Processing sentence {i+1}/{len(sentences)}...")
            progress_bar.progress((i + 1) / len(sentences))
            
            result = self.generate_audio(sentence, voice, speed, device, volume, pitch)
            if not result:
                continue
                
            sample_rate, audio_data = result
            audio_seg = AudioSegment(
                (audio_data * 32767).astype(np.int16).tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            audio_segments.append(audio_seg)
            
            pause = self.audio_processor.calculate_pause(sentence, adjusted_pause_settings)
            pause_durations.append(pause)
        
        progress_bar.empty()
        status_text.empty()
        
        if not audio_segments:
            return None, "Failed to generate audio", ""
        
        combined = AudioSegment.empty()
        for i, (seg, pause) in enumerate(zip(audio_segments, pause_durations)):
            combined += seg
            if i < len(audio_segments) - 1:
                combined += AudioSegment.silent(duration=max(50, pause))
        
        with io.BytesIO() as buffer:
            combined.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", str(combined.frame_rate)])
            buffer.seek(0)
            audio_data = np.frombuffer(buffer.read(), dtype=np.uint8)
        
        subtitles = self.subtitle_generator.generate_srt(audio_segments, sentences, adjusted_pause_settings)
        
        stats = (f"Characters: {len(clean_text)} | Words: {len(clean_text.split())}\n"
                f"Audio duration: {len(combined)/1000:.2f}s | Speed: {speed:.1f}x")
        
        return (combined.frame_rate, audio_data), stats, subtitles

    def generate_qa_audio(self, text: str, voice_q: str, voice_a: str, speed_q: float, speed_a: float,
                         device: str, repeat_times: int, pause_q: int, pause_a: int,
                         volume_q: float = 1.0, volume_a: float = 1.0,
                         pitch_q: float = 1.0, pitch_a: float = 1.0) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str]:
        """Generate Q&A audio."""
        dialogues = self.text_processor.parse_dialogues(text, ['Q', 'A'])
        
        if not dialogues:
            return None, "No Q/A pairs found", ""
        
        qa_pairs = []
        current_q = None
        for speaker, content in dialogues:
            if speaker.upper() == 'Q':
                current_q = content
            elif speaker.upper() == 'A' and current_q:
                qa_pairs.append((current_q, content))
                current_q = None
        
        if not qa_pairs:
            return None, "No valid Q/A pairs", ""
        
        combined = AudioSegment.empty()
        timing_info = []
        current_pos = 0
        
        progress_bar = st.progress(0)
        
        for pair_idx, (q_text, a_text) in enumerate(qa_pairs):
            for rep in range(repeat_times):
                # Question
                q_result = self.generate_audio(q_text, voice_q, speed_q, device, volume_q, pitch_q)
                if q_result:
                    sr_q, q_audio = q_result
                    q_seg = AudioSegment(
                        (q_audio * 32767).astype(np.int16).tobytes(),
                        frame_rate=sr_q,
                        sample_width=2,
                        channels=1
                    )
                    combined += q_seg
                    timing_info.append({
                        'start': current_pos,
                        'end': current_pos + len(q_seg),
                        'text': q_text
                    })
                    current_pos += len(q_seg)
                
                # Pause after question
                if pause_q > 0:
                    combined += AudioSegment.silent(duration=pause_q)
                    current_pos += pause_q
                
                # Answer
                a_result = self.generate_audio(a_text, voice_a, speed_a, device, volume_a, pitch_a)
                if a_result:
                    sr_a, a_audio = a_result
                    a_seg = AudioSegment(
                        (a_audio * 32767).astype(np.int16).tobytes(),
                        frame_rate=sr_a,
                        sample_width=2,
                        channels=1
                    )
                    combined += a_seg
                    timing_info.append({
                        'start': current_pos,
                        'end': current_pos + len(a_seg),
                        'text': a_text
                    })
                    current_pos += len(a_seg)
                
                # Pause after answer (except last)
                if rep < repeat_times - 1 or pair_idx < len(qa_pairs) - 1:
                    if pause_a > 0:
                        combined += AudioSegment.silent(duration=pause_a)
                        current_pos += pause_a
            
            progress_bar.progress((pair_idx + 1) / len(qa_pairs))
        
        progress_bar.empty()
        
        if len(combined) == 0:
            return None, "Failed to generate audio", ""
        
        with io.BytesIO() as buffer:
            combined.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", "24000"])
            audio_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        
        # Generate subtitles
        subtitles = []
        for idx, info in enumerate(timing_info, 1):
            start_str = f"{info['start']//3600000:02d}:{(info['start']%3600000)//60000:02d}:{(info['start']%60000)//1000:02d},{info['start']%1000:03d}"
            end_str = f"{info['end']//3600000:02d}:{(info['end']%3600000)//60000:02d}:{(info['end']%60000)//1000:02d},{info['end']%1000:03d}"
            
            subtitles.append(f"{idx}\n{start_str} --> {end_str}\n{info['text']}\n")
        
        stats = (f"Q/A Pairs: {len(qa_pairs)} | Repeat: {repeat_times}x\n"
                f"Duration: {len(combined)/1000:.2f}s | Q: {speed_q:.1f}x | A: {speed_a:.1f}x")
        
        return (24000, audio_data), stats, "\n".join(subtitles)

    def generate_multi_char_audio(self, text: str, voices: Dict[str, str], 
                                speeds: Dict[str, float], volumes: Dict[str, float], 
                                pitches: Dict[str, float], device: str,
                                pause_settings: Dict[str, int]) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str]:
        """Generate multi-character audio."""
        dialogues = self.text_processor.parse_dialogues(text, list(voices.keys()))
        
        if not dialogues:
            return None, "No dialogues found", ""
        
        combined = AudioSegment.empty()
        timing_info = []
        current_pos = 0
        char_stats = {char: {'lines': 0, 'duration': 0} for char in voices.keys()}
        
        progress_bar = st.progress(0)
        total_dialogues = len(dialogues)
        
        for idx, (speaker, content) in enumerate(dialogues):
            if speaker not in voices:
                continue
                
            voice = voices[speaker]
            speed = speeds.get(speaker, 1.0)
            volume = volumes.get(speaker, 1.0)
            pitch = pitches.get(speaker, 1.0)
            
            result = self.generate_audio(content, voice, speed, device, volume, pitch)
            if not result:
                continue
                
            sample_rate, audio_data = result
            
            audio_seg = AudioSegment(
                (audio_data * 32767).astype(np.int16).tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            
            seg_start = current_pos
            seg_end = seg_start + len(audio_seg)
            
            combined += audio_seg
            char_stats[speaker]['lines'] += 1
            char_stats[speaker]['duration'] += len(audio_seg)
            
            timing_info.append({
                'start': seg_start,
                'end': seg_end,
                'text': f"{speaker}: {content}"
            })
            
            current_pos = seg_end
            
            # Add pause
            pause = self.audio_processor.calculate_pause(content, pause_settings)
            if pause > 0:
                combined += AudioSegment.silent(duration=pause)
                current_pos += pause
            
            progress_bar.progress((idx + 1) / total_dialogues)
        
        progress_bar.empty()
        
        if len(combined) == 0:
            return None, "Failed to generate audio", ""
        
        with io.BytesIO() as buffer:
            combined.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", "24000"])
            audio_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        
        # Generate subtitles
        subtitles = []
        for idx, info in enumerate(timing_info, 1):
            start_str = f"{info['start']//3600000:02d}:{(info['start']%3600000)//60000:02d}:{(info['start']%60000)//1000:02d},{info['start']%1000:03d}"
            end_str = f"{info['end']//3600000:02d}:{(info['end']%3600000)//60000:02d}:{(info['end']%60000)//1000:02d},{info['end']%1000:03d}"
            
            subtitles.append(f"{idx}\n{start_str} --> {end_str}\n{info['text']}\n")
        
        # Generate stats
        stats_lines = [f"Multi-character dialogue ({len(combined)/1000:.2f}s)"]
        for char, stats in char_stats.items():
            if stats['lines'] > 0:
                stats_lines.append(
                    f"{char}: {stats['lines']} lines ({voices[char]}, "
                    f"{speeds.get(char, 1.0):.1f}x speed, {volumes.get(char, 1.0):.1f}x volume)"
                )
        
        return (24000, audio_data), "\n".join(stats_lines), "\n".join(subtitles)

# ======================= STREAMLIT UI =======================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microphone.png", width=80)
        st.title("🎙️ VANSARAH TTS")
        st.markdown("---")
        
        # Load model
        with st.spinner("Loading TTS engine..."):
            model = load_tts_model()
        
        if model:
            voice_list = model.get_voice_list()
            if voice_list:
                st.success(f"✅ Loaded {len(voice_list)} voices")
            else:
                st.warning("⚠️ No voice files found")
        else:
            st.error("❌ Failed to load TTS model")
            st.info("Running in demo mode")
        
        st.markdown("---")
        
        # Voice upload
        st.markdown("### 📂 Upload Voice Files")
        uploaded_files = st.file_uploader(
            "Upload .pt voice files",
            type=['pt'],
            accept_multiple_files=True,
            help="Voice files must be .pt format"
        )
        
        if uploaded_files:
            voices_dir = Path("voices")
            voices_dir.mkdir(exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = voices_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✓ {uploaded_file.name}")
            
            st.rerun()
        
        st.markdown("---")
        
        # System info
        st.markdown("### 🖥️ System Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
        with col2:
            st.metric("CUDA", "✅" if torch.cuda.is_available() else "❌")
        
        st.markdown("---")
        st.caption("Made with ❤️ | VANSARAH TTS")

    # Main content
    st.markdown("<h1 style='text-align: center;'>🎭 VANSARAH TTS - Complete Web App</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Advanced Text-to-Speech with Multiple Voices & Subtitles</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["📖 Standard Mode", "❓ Q&A Mode", "👥 Multi-Character Mode", "⚙️ Settings"],
        horizontal=True
    )
    
    if mode == "📖 Standard Mode":
        render_standard_mode(model)
    elif mode == "❓ Q&A Mode":
        render_qa_mode(model)
    elif mode == "👥 Multi-Character Mode":
        render_multi_char_mode(model)
    else:
        render_settings()

def render_standard_mode(model):
    """Standard TTS mode."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Text")
        text = st.text_area(
            "Enter text:",
            height=200,
            value="""Hello! Welcome to VANSARAH TTS.
This is a demonstration of the text-to-speech system.
Contact us at info@example.com or call 012-345-6789.
Visit: https://www.example.com for more information.""",
            help="Text will be automatically processed for special formats."
        )
        
        st.markdown("### 🎭 Voice Settings")
        
        voice_list = model.get_voice_list() if model else ["Demo Voice"]
        voice = st.selectbox("Select Voice:", voice_list, index=0)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            speed = st.slider("Speed", 0.7, 1.3, 1.0, 0.05)
        with col_s2:
            volume = st.slider("Volume", 0.5, 2.0, 1.0, 0.1)
        with col_s3:
            pitch = st.slider("Pitch", 0.8, 1.2, 1.0, 0.05)
        
        with st.expander("⚙️ Advanced Settings"):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                max_chars = st.slider("Max chars/segment", 100, 500, 250, 50)
                device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
            with col_a2:
                st.markdown("**Pause Settings (ms)**")
                default_pause = st.slider("Default", 0, 1000, 200)
                dot_pause = st.slider("After .", 0, 1500, 600)
                comma_pause = st.slider("After ,", 0, 800, 300)
        
        if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Please enter text")
                return
            
            with st.spinner("Generating audio..."):
                pause_settings = {
                    'default_pause': default_pause,
                    'dot_pause': dot_pause,
                    'ques_pause': dot_pause,
                    'comma_pause': comma_pause,
                    'colon_pause': comma_pause,
                    'excl_pause': dot_pause,
                    'semi_pause': comma_pause,
                    'dash_pause': comma_pause,
                    'time_colon_pause': 50
                }
                
                generator = TTSGenerator(model)
                result, stats, subtitles = generator.generate_story_audio(
                    text, voice, speed, device, pause_settings, volume, pitch, max_chars
                )
                
                if result:
                    sample_rate, audio_data = result
                    
                    # Save files
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    audio_path = output_dir / "output.mp3"
                    with open(audio_path, "wb") as f:
                        f.write(audio_data.tobytes())
                    
                    srt_path = output_dir / "subtitles.srt"
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(subtitles)
                    
                    # Display results
                    st.success("✅ Audio generated!")
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.audio(audio_path)
                    with col_r2:
                        st.download_button(
                            "📥 Download Audio",
                            data=audio_data.tobytes(),
                            file_name="tts_output.mp3",
                            mime="audio/mpeg"
                        )
                    
                    st.markdown("#### 📊 Statistics")
                    st.info(stats)
                    
                    st.markdown("#### 📝 Subtitles")
                    with st.expander("View subtitles"):
                        st.code(subtitles, language="text")
                        st.download_button(
                            "📥 Download SRT",
                            data=subtitles.encode(),
                            file_name="subtitles.srt",
                            mime="text/plain"
                        )
                else:
                    st.error("Failed to generate audio")

def render_qa_mode(model):
    """Q&A mode."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ❓ Q&A Input")
        qa_text = st.text_area(
            "Enter Q&A (format: Q: question / A: answer):",
            height=200,
            value="""Q: What is your email address?
A: You can email us at info@example.com

Q: What is your phone number?
A: Call us at 012-345-6789

Q: Where can I learn more?
A: Visit https://www.example.com""",
            help="Each line should start with Q: or A:"
        )
        
        st.markdown("### 🎭 Voice Settings")
        
        voice_list = model.get_voice_list() if model else ["Demo Q", "Demo A"]
        
        col_q, col_a = st.columns(2)
        with col_q:
            voice_q = st.selectbox("Question Voice", voice_list, index=0)
            speed_q = st.slider("Q Speed", 0.7, 1.3, 1.0, 0.05, key="sq")
            volume_q = st.slider("Q Volume", 0.5, 2.0, 1.0, 0.1, key="vq")
        
        with col_a:
            voice_a = st.selectbox("Answer Voice", voice_list, index=min(1, len(voice_list)-1))
            speed_a = st.slider("A Speed", 0.7, 1.3, 1.0, 0.05, key="sa")
            volume_a = st.slider("A Volume", 0.5, 2.0, 1.0, 0.1, key="va")
        
        st.markdown("### ⚙️ Q&A Settings")
        col_r, col_p = st.columns(2)
        with col_r:
            repeat = st.slider("Repeat each pair", 1, 5, 1)
        with col_p:
            pause_q = st.slider("Pause after Q (ms)", 100, 1000, 300)
            pause_a = st.slider("Pause after A (ms)", 100, 1500, 500)
        
        device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], key="qa_device")
    
    with col2:
        if st.button("🎵 Generate Q&A Audio", type="primary", use_container_width=True):
            if not qa_text.strip():
                st.warning("Please enter Q&A text")
                return
            
            with st.spinner("Generating Q&A audio..."):
                generator = TTSGenerator(model)
                result, stats, subtitles = generator.generate_qa_audio(
                    qa_text, voice_q, voice_a, speed_q, speed_a,
                    device, repeat, pause_q, pause_a,
                    volume_q, volume_a
                )
                
                if result:
                    sample_rate, audio_data = result
                    
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    audio_path = output_dir / "qa_output.mp3"
                    with open(audio_path, "wb") as f:
                        f.write(audio_data.tobytes())
                    
                    st.success("✅ Q&A audio generated!")
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.audio(audio_path)
                    with col_r2:
                        st.download_button(
                            "📥 Download Audio",
                            data=audio_data.tobytes(),
                            file_name="qa_output.mp3",
                            mime="audio/mpeg"
                        )
                    
                    st.markdown("#### 📊 Statistics")
                    st.info(stats)
                    
                    st.markdown("#### 📝 Subtitles")
                    with st.expander("View subtitles"):
                        st.code(subtitles, language="text")
                        st.download_button(
                            "📥 Download SRT",
                            data=subtitles.encode(),
                            file_name="qa_subtitles.srt",
                            mime="text/plain"
                        )
                else:
                    st.error("Failed to generate Q&A audio")

def render_multi_char_mode(model):
    """Multi-character mode."""
    st.markdown("### 👥 Multi-Character Dialogue")
    
    col_input, col_settings = st.columns([2, 1])
    
    with col_input:
        char_text = st.text_area(
            "Enter dialogue (format: CHAR1: text / CHAR2: text):",
            height=200,
            value="""CHAR1: Hello everyone, my email is info@example.com
CHAR2: Hi there! You can call me at 012-345-6789
CHAR3: Our website is https://www.example.com
CHAR1: How are you today?
CHAR2: I'm doing great, thanks for asking!""",
            help="Format: CHAR1: dialogue\nCHAR2: dialogue\nCHAR3: dialogue"
        )
    
    with col_settings:
        st.markdown("### 🎭 Character Setup")
        
        voice_list = model.get_voice_list() if model else ["Voice1", "Voice2", "Voice3", "Voice4"]
        
        characters = {}
        for i in range(1, 5):
            char_name = st.text_input(f"Character {i}", value=f"CHAR{i}", key=f"char{i}")
            
            col_voice, col_speed = st.columns([2, 1])
            with col_voice:
                voice = st.selectbox(f"Voice", voice_list, 
                                    index=i-1 if len(voice_list) > i-1 else 0,
                                    key=f"char{i}_voice")
            with col_speed:
                speed = st.slider("Speed", 0.7, 1.3, 1.0, 0.05, key=f"char{i}_speed")
            
            characters[char_name] = {"voice": voice, "speed": speed, "volume": 1.0, "pitch": 1.0}
            
            if i < 4:
                st.markdown("---")
        
        device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], key="char_device")
        
        with st.expander("Pause Settings"):
            default_pause = st.slider("Default pause (ms)", 0, 1000, 300, key="char_pause")
    
    st.markdown("---")
    
    if st.button("🎭 Generate Multi-Character Audio", type="primary", use_container_width=True):
        if not char_text.strip():
            st.warning("Please enter dialogue")
            return
        
        with st.spinner("Creating multi-character audio..."):
            # Prepare voice settings
            voices = {}
            speeds = {}
            volumes = {}
            pitches = {}
            
            for char_name, settings in characters.items():
                if char_name.strip():
                    voices[char_name.strip()] = settings["voice"]
                    speeds[char_name.strip()] = settings["speed"]
                    volumes[char_name.strip()] = settings["volume"]
                    pitches[char_name.strip()] = settings["pitch"]
            
            if not voices:
                st.error("Please configure at least one character")
                return
            
            pause_settings = {
                'default_pause': default_pause,
                'dot_pause': default_pause * 2,
                'ques_pause': default_pause * 2,
                'comma_pause': default_pause // 2,
                'colon_pause': default_pause,
                'excl_pause': default_pause * 2,
                'semi_pause': default_pause,
                'dash_pause': default_pause // 2,
                'time_colon_pause': 0
            }
            
            generator = TTSGenerator(model)
            result, stats, subtitles = generator.generate_multi_char_audio(
                char_text, voices, speeds, volumes, pitches, device, pause_settings
            )
            
            if result:
                sample_rate, audio_data = result
                
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                audio_path = output_dir / "multi_char.mp3"
                with open(audio_path, "wb") as f:
                    f.write(audio_data.tobytes())
                
                st.success("✅ Multi-character audio generated!")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.audio(audio_path)
                with col_r2:
                    st.download_button(
                        "📥 Download Audio",
                        data=audio_data.tobytes(),
                        file_name="multi_char.mp3",
                        mime="audio/mpeg"
                    )
                
                st.markdown("#### 📊 Character Summary")
                for char_name, settings in characters.items():
                    if char_name.strip():
                        st.info(f"**{char_name}**: {settings['voice']} ({settings['speed']:.1f}x speed)")
                
                st.markdown("#### 📝 Subtitles")
                with st.expander("View subtitles"):
                    st.code(subtitles, language="text")
                    st.download_button(
                        "📥 Download SRT",
                        data=subtitles.encode(),
                        file_name="multi_char.srt",
                        mime="text/plain"
                    )
            else:
                st.error("Failed to generate multi-character audio")

def render_settings():
    """Settings page."""
    st.markdown("### ⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗃️ Voice Management")
        
        voices_dir = Path("voices")
        if voices_dir.exists():
            voice_files = list(voices_dir.glob("*.pt"))
            if voice_files:
                st.success(f"Found {len(voice_files)} voice files:")
                for vf in voice_files:
                    st.write(f"- {vf.stem}")
                
                if st.button("🗑️ Clear Voice Cache"):
                    model = load_tts_model()
                    if model:
                        model.voice_cache.clear()
                        st.success("Voice cache cleared")
            else:
                st.warning("No voice files found")
        else:
            st.warning("Voices directory doesn't exist")
    
    with col2:
        st.markdown("#### 🔧 System Configuration")
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**CUDA Device:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Application"):
            st.rerun()
        
        if st.button("🧹 Clear Temporary Files"):
            import shutil
            output_dir = Path("output")
            if output_dir.exists():
                shutil.rmtree(output_dir)
                st.success("Temporary files cleared")
    
    st.markdown("---")
    
    st.markdown("#### 📚 Examples")
    
    examples = st.columns(3)
    
    with examples[0]:
        if st.button("📞 Contact Example"):
            st.session_state.text = """Call us: 123-456-7890
Email: contact@company.com
Website: https://company.com"""
            st.rerun()
    
    with examples[1]:
        if st.button("💰 Price Example"):
            st.session_state.text = """Basic: $9.99/month
Pro: $29.99/month
Enterprise: $99.99/month"""
            st.rerun()
    
    with examples[2]:
        if st.button("🎭 Dialogue Example"):
            st.session_state.char_text = """CHAR1: Hello, how are you?
CHAR2: I'm good, thanks!
CHAR3: What's the plan today?
CHAR4: Let's meet at 3:00 PM"""
            st.rerun()

# ======================= MAIN EXECUTION =======================
if __name__ == "__main__":
    # Check for dependencies
    try:
        import phonemizer
    except ImportError:
        st.warning("Installing phonemizer...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "phonemizer==3.2.1", "--quiet"])
        st.rerun()
    
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("""
        ### 🔧 Troubleshooting:
        
        1. Ensure all voice files (.pt) are in 'voices/' folder
        2. Check internet connection for initial setup
        3. Verify sufficient disk space
        4. For Streamlit Sharing: files must be under 200MB total
        
        ### 📞 Support:
        - Check console for detailed errors
        - Upload voice files in the sidebar
        - Use demo mode if voice files unavailable
        """)
