import streamlit as st
import torch
import numpy as np
import io
import time
import re
import json
import os
import sys
from typing import List, Tuple, Optional, Dict
from datetime import timedelta

# Kiểm tra và cài đặt các package cần thiết
try:
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range
except ImportError:
    st.error("Please install pydub: pip install pydub")
    st.stop()

try:
    from phonemizer import backend
except ImportError:
    st.error("Please install phonemizer: pip install phonemizer")
    st.stop()

# Thêm message loading
st.set_page_config(
    page_title="Advanced TTS System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hiển thị loading message
with st.spinner("Loading TTS system..."):
    # Import các modules custom
    try:
        # Giả lập các modules nếu chưa có
        class KModel:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
            
        class KPipeline:
            def __init__(self, lang_code='a', model=False):
                self.lang_code = lang_code
                
            def __call__(self, text, voice, speed):
                # Giả lập pipeline
                yield None, None, None
        
        # Giả lập SPECIAL_CASES nếu chưa có
        SPECIAL_CASES = {}
        
    except Exception as e:
        st.error(f"Error loading custom modules: {e}")
        # Tạo các class giả lập để tiếp tục
        class KModel:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
            
        class KPipeline:
            def __init__(self, lang_code='a', model=False):
                self.lang_code = lang_code
                
            def __call__(self, text, voice, speed):
                yield None, None, None
        
        SPECIAL_CASES = {}

# Tokenizer class with enhanced text processing
class Tokenizer:
    def __init__(self):
        self.VOCAB = self._get_vocab()
        self.special_cases = self._build_special_cases()
        self.special_regex = self._build_special_regex()
        self.abbreviation_patterns = self._build_abbreviation_patterns()
        self.phonemizers = {
            'en-us': backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
        }
        
    def _fix_heteronyms(self, text: str) -> str:
        """
        Fix common English heteronyms by context so TTS reads correctly.
        """
    
        # --- LIVE ---
        # Adjective/adverb cases -> "laiv"
        text = re.sub(r"\blive\s+(concert|show|event|music|performance|broadcast)\b",
                      r"lyve \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\blive\s+(stream|coverage|demo|recording)\b",
                      r"lyve \1", text, flags=re.IGNORECASE)
        # Verb cases -> "liv"
        text = re.sub(r"\blive\b", "liv", text, flags=re.IGNORECASE)
    
        # --- LEAD ---
        # Metal = led
        text = re.sub(r"\blead (pipe|metal|balloon|paint|weight)\b",
                      r"led \1", text, flags=re.IGNORECASE)
        # Verb = leed
        text = re.sub(r"\blead\b", "leed", text, flags=re.IGNORECASE)
    
        # --- READ ---
        text = re.sub(r'What did you read', 'What did you reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did he read', 'What did he reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did she read', 'What did she reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did we read', 'What did we reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did they read', 'What did they reed', text, flags=re.IGNORECASE)
        
        text = re.sub(r"\b(Sometimes|sometimes)\s+I\s+read\b", r"\1 I reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+we\s+read\b", r"\1 we reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+they\s+read\b", r"\1 they reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+you\s+read\b", r"\1 you reed", text)
        
        text = re.sub(r"\bto\s+read\b", "to reed", text, flags=re.IGNORECASE)
        
        text = re.sub(r"\b(I|You|We|They)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)

        text = re.sub(r"\b(He|She|It)\s+reads\b", r"\1 reeds", text, flags=re.IGNORECASE)
      
        
        text = re.sub(r'^I read a', 'I red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^He read a', 'He red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^She read a', 'She red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^We read a', 'We red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^They read a', 'They red a', text, flags=re.IGNORECASE)

        text = re.sub(r"(^|\.\s+)(Please\s+)?Read\b", r"\1Reed", text)
        text = re.sub(r"(^|\.\s+)Read\b", r"\1Reed", text)


        text = re.sub(r"\b(don't|do not)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(doesn't|does not)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)


        text = re.sub(r"\bDo\s+(you|we|they)\s+read\b", r"Do \1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bDoes\s+(he|she|it)\s+read\b", r"Does \1 reeds", text, flags=re.IGNORECASE)

        text = re.sub(r"\b(is|are)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(was|were)\s+read\b", r"\1 red", text, flags=re.IGNORECASE)

        text = re.sub(r"\b(I|You|We|They)\s+(often|rarely|seldom)\s+read\b", r"\1 \2 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(He|She|It)\s+(often|rarely|seldom)\s+reads\b", r"\1 \2 reeds", text, flags=re.IGNORECASE)

        text = re.sub(r"\bIf\s+(I|You|We|They)\s+read\b", r"If \1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bIf\s+(He|She|It)\s+reads\b", r"If \1 reeds", text, flags=re.IGNORECASE)

        
        text = re.sub(r'\b(did)\s+([^.!?]*?)\bread\b', r'\1 \2reed', text, flags=re.IGNORECASE)
        text = re.sub(r"\bdidn't\s+read\b", "didn't reed", text, flags=re.IGNORECASE)
        
        past_auxiliaries = r'\b(had|was|were|have|has|haven\'t|hasn\'t|hadn\'t|wasn\'t|weren\'t)'
        text = re.sub(rf'{past_auxiliaries}\s+([^.!?]*?)\bread\b', r'\1 \2red', text, flags=re.IGNORECASE)
        
        past_time_words = r'\b(yesterday|last\s+(night|week|month|year)|(\d+\s+)?(days|weeks|months|years)\s+ago|already|just|earlier|before|previously|recently|when\s+I\s+was)'
        text = re.sub(rf'{past_time_words}[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' red').replace(' Read', ' Red'),
                     text, flags=re.IGNORECASE)
        text = re.sub(rf'\bread\b[^.!?]*?{past_time_words}',
                     lambda m: m.group(0).replace('read ', 'red ').replace('Read ', 'Red '),
                     text, flags=re.IGNORECASE)
        
        text = re.sub(r'\b(today)[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' reed').replace(' Read', ' Reed'),
                     text, flags=re.IGNORECASE)
        text = re.sub(r'\bread\b[^.!?]*?\b(today)\b',
                     lambda m: m.group(0).replace('read ', 'reed ').replace('Read ', 'Reed '),
                     text, flags=re.IGNORECASE)
        
        future_present_words = r'\b(will|shall|going to|plan to|want to|need to|would like to|tomorrow|next|every|always|usually|now)'
        text = re.sub(rf'{future_present_words}[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' reed').replace(' Read', ' Reed'),
                     text, flags=re.IGNORECASE)
        text = re.sub(rf'\bread\b[^.!?]*?{future_present_words}',
                     lambda m: m.group(0).replace('read ', 'reed ').replace('Read ', 'Reed '),
                     text, flags=re.IGNORECASE)
        
        def replace_remaining_read(match):
            word = match.group(0)
            if getattr(self, 'is_past_story', False):
                return 'red' if word.islower() else 'Red'
            else:
                return 'reed' if word.islower() else 'Reed'
        
        text = re.sub(r'\bread\b', replace_remaining_read, text, flags=re.IGNORECASE)
        
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
        """Regex để nhận diện các từ đặc biệt"""
        words = sorted(self.special_cases.keys(), key=len, reverse=True)
        return re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', flags=re.IGNORECASE)

    def apply_special_cases(self, text: str) -> str:
        """Áp dụng thay thế các từ đặc biệt trong văn bản"""
        def repl(match):
            word = match.group(0)
            return self.special_cases.get(word.lower(), word)
        
        return self.special_regex.sub(repl, text)
        
    def _build_abbreviation_patterns(self):
        """Xử lý các từ viết tắt"""
        return {
            r"\ba\.m\.?\b": "AM",
            r"\bp\.m\.?\b": "PM",      
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
            r"\bAve\.?\b": "Avenue ",
            r"\bBlvd\.?\b": "Boulevard ",
            r"\bRd\.?\b": "Road ",
            r"\bJan\.?\b": "January ",
            r"\bFeb\.?\b": "February ",
            r"\bMar\.?\b": "March ",
            r"\bApr\.?\b": "April ",
            r"\bJun\.?\b": "June ",
            r"\bJul\.?\b": "July ",
            r"\bAug\.?\b": "August ",
            r"\bSep(?:t)?\.?\b": "September ",
            r"\bOct\.?\b": "October ",
            r"\bNov\.?\b": "November ",
            r"\bDec\.?\b": "December ",
            r"\b1st\b": "first ",
            r"\b2nd\b": "second ",
            r"\b3rd\b": "third ",
            r"\b4th\b": "fourth ",
            r"\b5th\b": "fifth ",
            r"\b6th\b": "sixth ",
            r"\b7th\b": "seventh ",
            r"\b8th\b": "eighth ",
            r"\b9th\b": "ninth ",
            r"\b10th\b": "tenth ",
            r"\b11th\b": "eleventh ",
            r"\b12th\b": "twelfth ",
            r"\b13th\b": "thirteenth ",
            r"\b14th\b": "fourteenth ",
            r"\b15th\b": "fifteenth ",
            r"\b20th\b": "twentieth ",
            r"\b21st\b": "twenty first ",
            r"\b22nd\b": "twenty second ",
            r"\b23rd\b": "twenty third ",
            r"\b30th\b": "thirtieth ",
            r"\b31st\b": "thirty first ",
            r"\b0\b": "zero ",
            r"\b1\b": "one ",
            r"\b2\b": "two ",
            r"\b3\b": "three ",
            r"\b4\b": "four ",
            r"\b5\b": "five ",
            r"\b6\b": "six ",
            r"\b7\b": "seven ",
            r"\b8\b": "eight ",
            r"\b9\b": "nine ",
            r"\b10\b": "ten ",
            r"\b11\b": "eleven ",
            r"\b12\b": "twelve ",
            r"\b13\b": "thirteen ",
            r"\b14\b": "fourteen ",
            r"\b15\b": "fifteen ",
            r"\b16\b": "sixteen ",
            r"\b17\b": "seventeen ",
            r"\b18\b": "eighteen ",
            r"\b19\b": "nineteen ",
            r"\b20\b": "twenty ",
            r"\b21\b": "twenty one ",
            r"\b22\b": "twenty two ",
            r"\b23\b": "twenty three ",
            r"\b24\b": "twenty four ",
            r"\b25\b": "twenty five ",
            r"\b26\b": "twenty six ",
            r"\b27\b": "twenty seven ",
            r"\b28\b": "twenty eight ",
            r"\b29\b": "twenty nine ",
            r"\b30\b": "thirty ",
            r"\b31\b": "thirty one ",
            r"\b32\b": "thirty two ",
            r"\b33\b": "thirty three ",
            r"\b34\b": "thirty four ",
            r"\b35\b": "thirty five ",
            r"\b36\b": "thirty six ",
            r"\b37\b": "thirty seven ",
            r"\b38\b": "thirty eight ",
            r"\b39\b": "thirty nine ",
            r"\b40\b": "forty ",
            r"\b41\b": "forty one ",
            r"\b42\b": "forty two ",
            r"\b43\b": "forty three ",
            r"\b44\b": "forty four ",
            r"\b45\b": "forty five ",
            r"\b46\b": "forty six ",
            r"\b47\b": "forty seven ",
            r"\b48\b": "forty eight ",
            r"\b49\b": "forty nine ",
            r"\b50\b": "fifty ",
            r"\b51\b": "fifty one ",
            r"\b52\b": "fifty two ",
            r"\b53\b": "fifty three ",
            r"\b54\b": "fifty four ",
            r"\b55\b": "fifty five ",
            r"\b56\b": "fifty six ",
            r"\b57\b": "fifty seven ",
            r"\b58\b": "fifty eight ",
            r"\b59\b": "fifty nine ",
            r"\b60\b": "sixty ",
            r"\b61\b": "sixty one ",
            r"\b62\b": "sixty two ",
            r"\b63\b": "sixty three ",
            r"\b64\b": "sixty four ",
            r"\b65\b": "sixty five ",
            r"\b66\b": "sixty six ",
            r"\b67\b": "sixty seven ",
            r"\b68\b": "sixty eight ",
            r"\b69\b": "sixty nine ",
            r"\b70\b": "seventy ",
            r"\b71\b": "seventy one ",
            r"\b72\b": "seventy two ",
            r"\b73\b": "seventy three ",
            r"\b74\b": "seventy four ",
            r"\b75\b": "seventy five ",
            r"\b76\b": "seventy six ",
            r"\b77\b": "seventy seven ",
            r"\b78\b": "seventy eight ",
            r"\b79\b": "seventy nine ",
            r"\b80\b": "eighty ",
            r"\b81\b": "eighty one ",
            r"\b82\b": "eighty two ",
            r"\b83\b": "eighty three ",
            r"\b84\b": "eighty four ",
            r"\b85\b": "eighty five ",
            r"\b86\b": "eighty six ",
            r"\b87\b": "eighty seven ",
            r"\b88\b": "eighty eight ",
            r"\b89\b": "eighty nine ",
            r"\b90\b": "ninety ",
            r"\b91\b": "ninety one ",
            r"\b92\b": "ninety two ",
            r"\b93\b": "ninety three ",
            r"\b94\b": "ninety four ",
            r"\b95\b": "ninety five ",
            r"\b96\b": "ninety six ",
            r"\b97\b": "ninety seven ",
            r"\b98\b": "ninety eight ",
            r"\b99\b": "ninety nine ",
            r"\b100\b": "one hundred ",
        }

    def _add_invisible_space_to_conjunctions(self, text: str) -> str:
        """Thêm khoảng vô hình TRƯỚC từ nối để TTS đọc tự nhiên"""
        ZWS = "\u200B"
        conjunctions = [
            "and", "or", "but", "So", "Because", "Although",
            "However", "Then", "Therefore", "Meanwhile", "Yet", "Nor"
        ]
        pattern = r'\b(' + '|'.join(conjunctions) + r')\b'
        return re.sub(pattern, lambda m: f"{ZWS}{m.group(0)}", text, flags=re.IGNORECASE)

    def process_text(self, text: str) -> str:
        """Chuẩn hóa văn bản giữ nguyên ngữ cảnh và thêm pause giữa các từ"""
        text = text.replace("'", "'").replace('"', '').replace('"', '').replace('"', '')
        
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
        text = re.sub(r'\bphotos\b', 'foh-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\bphoto\b', 'foh-toh', text, flags=re.IGNORECASE)
        text = re.sub(r'\btomatoes\b', 'tuh-may-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\btomato\b', 'tuh-may-toh', text, flags=re.IGNORECASE)
        text = re.sub(r'\bLos Angeles\b', 'Loss an-juh-luhs', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAngeles\b', 'an-juh-luhs', text, flags=re.IGNORECASE)

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
            st.warning(f"Language '{lang}' not supported. Defaulting to 'en-us'.")
            lang = 'en-us'

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

        phonemes = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', phonemes)
        phonemes = re.sub(r' z(?=[;:,.!?¡¿—…"«»" ]|$)', 'z', phonemes)

        if lang == 'a':
            phonemes = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', phonemes)

        phonemes = ''.join(filter(lambda p: p in self.VOCAB, phonemes))
        return phonemes.strip()

# Khởi tạo môi trường - Ưu tiên GPU
CUDA_AVAILABLE = torch.cuda.is_available()

class TTSModel:
    def __init__(self):
        self.use_cuda = CUDA_AVAILABLE
        self.models = {}
        self.tokenizer = Tokenizer()
        self.voice_cache = {}
        self.voice_files = self._discover_voices()
        
        try:
            if self.use_cuda:
                self.models['cuda'] = torch.compile(KModel().to('cuda').eval(), mode='max-autotune')
                with torch.no_grad():
                    _ = self.models['cuda'](torch.randn(1, 64).cuda(), torch.randn(1, 80, 100).cuda(), 1.0)
            
            self.models['cpu'] = KModel().to('cpu').eval()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.models = {'cpu': KModel().to('cpu').eval()}
        
        self.pipelines = {
            'a': KPipeline(lang_code='a', model=False),
            'b': KPipeline(lang_code='b', model=False)
        }
    
    def _discover_voices(self):
        """Discover available voice files in the voices folder"""
        voice_files = {}
        voices_dir = "voices"
        
        if not os.path.exists(voices_dir):
            os.makedirs(voices_dir)
            st.info(f"Created voices directory at {os.path.abspath(voices_dir)}")
            # Tạo một số voice mẫu
            for i in range(5):
                voice_name = f"Voice_{i+1}"
                voice_path = os.path.join(voices_dir, f"{voice_name}.pt")
                # Tạo file giả lập
                dummy_data = torch.randn(100, 80)
                torch.save(dummy_data, voice_path)
                voice_files[voice_name] = voice_path
                st.info(f"Created sample voice: {voice_name}")
            return voice_files
            
        for file in os.listdir(voices_dir):
            if file.endswith(".pt"):
                voice_name = os.path.splitext(file)[0]
                voice_files[voice_name] = os.path.join(voices_dir, file)
                st.info(f"Found voice: {voice_name}")
                
        return voice_files

    def get_voice_list(self):
        """Get list of available voices for the UI"""
        voices = list(self.voice_files.keys())
        if not voices:
            st.warning("No voice files found in voices folder")
            # Tạo voices mẫu
            voices = ["Voice_1", "Voice_2", "Voice_3", "Voice_4", "Voice_5"]
        return voices

model_manager = TTSModel()

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

    def smart_text_split(self, text: str, max_length: int = 250) -> List[str]:
        if len(text) <= max_length:
            return [text]
    
        sentences = []
        current_text = text.strip()
    
        abbreviations = {
            "a.m.", "p.m.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.",
            "etc.", "e.g.", "i.e.", "vs.", "approx.", "no.", "vol.", "fig.", "p."
        }
    
        while current_text and len(current_text) > max_length:
            ideal_split_pos = -1
            last_period_pos = current_text.rfind('.', 0, max_length)
    
            if last_period_pos > 0:
                prev_text = current_text[max(0, last_period_pos - 10):last_period_pos + 1].lower().strip()
                if not any(prev_text.endswith(abbr) for abbr in abbreviations):
                    ideal_split_pos = last_period_pos + 1
    
            if ideal_split_pos == -1:
                for punct in ['!', '?', ';', ':']:
                    punct_pos = current_text.rfind(punct, 0, max_length)
                    if punct_pos > 0:
                        ideal_split_pos = punct_pos + 1
                        break
    
            if ideal_split_pos == -1:
                comma_pos = current_text.rfind(',', 0, max_length)
                if comma_pos > 0:
                    ideal_split_pos = comma_pos + 1
    
            if ideal_split_pos == -1:
                space_pos = current_text.rfind(' ', 0, max_length)
                if space_pos > 0:
                    ideal_split_pos = space_pos + 1
                else:
                    ideal_split_pos = max_length
    
            part = current_text[:ideal_split_pos].strip()
            if part:
                sentences.append(part)
            current_text = current_text[ideal_split_pos:].strip()
    
        if current_text:
            sentences.append(current_text)
    
        return sentences

    @staticmethod
    def _process_special_cases(text: str) -> str:
        text = TextProcessor._process_emails(text)
        text = TextProcessor._process_websites(text)
        text = TextProcessor._process_phone_numbers(text)
        text = TextProcessor._process_temperatures(text)
        text = TextProcessor._process_measurements(text)
        text = TextProcessor._process_currency(text)
        text = TextProcessor._process_percentages(text)
        text = TextProcessor._process_math_operations(text)
        text = TextProcessor._process_times(text)
        text = TextProcessor._process_years(text)
        text = TextProcessor._process_special_symbols(text)
        
        return text
    
    @staticmethod
    def _process_emails(text: str) -> str:
        def convert_email(match):
            full_email = match.group(0)
            processed = (full_email
                        .replace('@', ' at ')
                        .replace('.', ' dot ')
                        .replace('-', ' dash ')
                        .replace('_', ' underscore ')
                        .replace('+', ' plus ')
                        .replace('/', ' slash ')
                        .replace('=', ' equals '))
            return processed

        email_pattern = r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b'
        return re.sub(email_pattern, convert_email, text)

    @staticmethod
    def _process_websites(text: str) -> str:
        def convert_website(match):
            url = match.group(1)
            return (url.replace('.', ' dot ')
                     .replace('-', ' dash ')
                     .replace('_', ' underscore ')
                     .replace('/', ' slash ')
                     .replace('?', ' question mark ')
                     .replace('=', ' equals ')
                     .replace('&', ' ampersand '))

        website_pattern = r'\b(?![\w.-]*@)((?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:[/?=&#][\w.-]*)*)\b'
        return re.sub(website_pattern, convert_website, text, flags=re.IGNORECASE)

    @staticmethod
    def _process_temperatures(text: str) -> str:
        def temp_to_words(temp, unit):
            temp_text = TextProcessor._number_to_words(temp)
            unit = unit.upper() if unit else ''
            
            unit_map = {
                'C': 'degrees Celsius',
                'F': 'degrees Fahrenheit',
                'N': 'degrees north',
                'S': 'degrees south',
                'E': 'degrees east', 
                'W': 'degrees west',
                '': 'degrees'
            }
            unit_text = unit_map.get(unit, f'degrees {unit}')
            
            return f"{temp_text} {unit_text}"
        
        text = re.sub(
            r'(-?\d+)°([NSEWCFnsewcf]?)',
            lambda m: temp_to_words(m.group(1), m.group(2)),
            text,
            flags=re.IGNORECASE
        )
        
        text = re.sub(r'°', ' degrees ', text)
        
        return text

    @staticmethod
    def _process_measurements(text: str) -> str:
        units_map = {
            'km/h': 'kilometers per hour',
            'mph': 'miles per hour',
            'kg': 'kilograms',
            'g': 'grams',
            'cm': 'centimeters',
            'm': 'meters',
            'mm': 'millimeters',
            'L': 'liters',
            'l': 'liters',
            'ml': 'milliliters',
            'mL': 'milliliters',
            'h': 'hours',
            'min': 'minutes'
        }

        for unit, word in units_map.items():
            pattern = rf'(\d+(?:\.\d+)?)\s*{unit}\b'
            text = re.sub(pattern, lambda m: f"{TextProcessor._number_to_words(m.group(1))} {word}", text)
        
        text = re.sub(r'(\d+)\s+s\b', lambda m: f"{TextProcessor._number_to_words(m.group(1))} seconds", text)
        
        return text
    
    @staticmethod
    def _process_currency(text: str) -> str:
        currency_map = {
            '$': 'dollars',
            '€': 'euros',
            '£': 'pounds',
            '¥': 'yen',
            '₩': 'won',
            '₽': 'rubles'
        }
    
        def currency_to_words(value, symbol):
            if value.endswith('.'):
                value = value[:-1]
                return f"{TextProcessor._number_to_words(value)} {currency_map.get(symbol, '')}."
    
            if '.' in value:
                integer_part, decimal_part = value.split('.')
                decimal_part = decimal_part.ljust(2, '0')
                return (
                    f"{TextProcessor._number_to_words(integer_part)} {currency_map.get(symbol, '')} "
                    f"and {TextProcessor._number_to_words(decimal_part)} cents"
                )
    
            return f"{TextProcessor._number_to_words(value)} {currency_map.get(symbol, '')}"
    
        text = re.sub(
            r'([$€£¥₩₽])(\d+(?:\.\d+)?)(?=\s|$|\.|,|;)',
            lambda m: currency_to_words(m.group(2), m.group(1)),
            text
        )
    
        return text

    @staticmethod
    def _process_percentages(text: str) -> str:
        text = re.sub(
            r'(\d+\.?\d*)%',
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} percent",
            text
        )
        return text

    @staticmethod
    def _process_math_operations(text: str) -> str:
        math_map = {
            '+': 'plus',
            '-': 'minus',
            '×': 'times',
            '*': 'times',
            '÷': 'divided by',
            '/': 'divided by',
            '=': 'equals',
            '>': 'is greater than',
            '<': 'is less than'
        }
    
        text = re.sub(
            r'(\d+)\s*-\s*(\d+)(?!\s*[=+×*÷/><])',
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} to {TextProcessor._number_to_words(m.group(2))}",
            text
        )
    
        text = re.sub(
            r'(\d+)\s*-\s*(\d+)(?=\s*[=+×*÷/><])',
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} minus {TextProcessor._number_to_words(m.group(2))}",
            text
        )
    
        text = re.sub(
            r'(\d+)\s*([+×*÷/=><])\s*(\d+)',
            lambda m: (f"{TextProcessor._number_to_words(m.group(1))} "
                      f"{math_map.get(m.group(2), m.group(2))} "
                      f"{TextProcessor._number_to_words(m.group(3))}"),
            text
        )
    
        text = re.sub(
            r'(\d+)/(\d+)',
            lambda m: (f"{TextProcessor._number_to_words(m.group(1))} "
                      f"divided by {TextProcessor._number_to_words(m.group(2))}"),
            text
        )
    
        return text

    @staticmethod
    def _process_special_symbols(text: str) -> str:
        symbol_map = {
            '@': 'at',
            '#': 'number',
            '&': 'and',
            '_': 'underscore'
        }

        text = re.sub(
            r'@(\w+)',
            lambda m: f"at {m.group(1)}",
            text
        )

        text = re.sub(
            r'#(\d+)',
            lambda m: f"number {TextProcessor._number_to_words(m.group(1))}",
            text
        )

        for symbol, replacement in symbol_map.items():
            text = text.replace(symbol, f' {replacement} ')

        return text

    @staticmethod
    def _process_times(text: str) -> str:
        text = re.sub(
            r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b',
            lambda m: TextProcessor._time_to_words(m.group(1), m.group(2), m.group(3), m.group(4)),
            text
        )
        return text
    
    @staticmethod
    def _time_to_words(hour: str, minute: str, second: str = None, period: str = None) -> str:
        hour_int = int(hour)
        minute_int = int(minute)
        
        period_text = f" {period.upper()}" if period else ""
        
        hour_12 = hour_int % 12
        hour_text = "twelve" if hour_12 == 0 else TextProcessor._number_to_words(str(hour_12))
        
        minute_text = " \u200Bo'clock\u200B " if minute_int == 0 else \
                     f"oh {TextProcessor._number_to_words(minute)}" if minute_int < 10 else \
                     TextProcessor._number_to_words(minute)
        
        second_text = ""
        if second and int(second) > 0:
            second_text = f" and {TextProcessor._number_to_words(second)} seconds"
        
        if minute_int == 0 and not second_text:
            return f"{hour_text}{minute_text}{period_text}"
        else:
            return f"{hour_text} {minute_text}{second_text}{period_text}"

    @staticmethod
    def _process_years(text: str) -> str:
        text = re.sub(
            r'\b(1[0-9]{2}0|2[0-9]{2}0)s\b',
            lambda m: TextProcessor._decade_to_words(m.group(1)),
            text
        )
        
        text = re.sub(
            r'\b(1[0-9]{3}|2[0-9]{3})\b',
            lambda m: TextProcessor._year_to_words(m.group(1)),
            text
        )
        
        return text

    @staticmethod
    def _decade_to_words(year: str) -> str:
        if len(year) != 4:
            return year
        
        century = year[:2]
        decade = year[2:]
        
        century_words = TextProcessor._year_part_to_words(century)
        decade_words = TextProcessor._decade_part_to_words(decade)
        
        return f"{century_words} {decade_words}"

    @staticmethod
    def _year_to_words(year: str) -> str:
        if len(year) != 4:
            return year
        
        if year == "2000":
            return "two thousand"
        elif year.startswith('200') and year[3] != '0':
            return f"two thousand {TextProcessor._digit_to_word(year[3])}"
        
        if year.startswith('20'):
            return f"twenty {TextProcessor._two_digit_year_to_words(year[2:])}"
        
        century = year[:2]
        decade = year[2:]
        
        century_words = TextProcessor._year_part_to_words(century)
        decade_words = TextProcessor._two_digit_year_to_words(decade)
        
        return f"{century_words} {decade_words}"

    @staticmethod
    def _year_part_to_words(part: str) -> str:
        numbers = {
            '19': 'nineteen', '20': 'twenty', '21': 'twenty-one',
            '18': 'eighteen', '17': 'seventeen', '16': 'sixteen',
            '15': 'fifteen', '14': 'fourteen', '13': 'thirteen'
        }
        return numbers.get(part, part)

    @staticmethod
    def _decade_part_to_words(decade: str) -> str:
        decades = {
            '00': 'hundreds', '10': 'tens', '20': 'twenties',
            '30': 'thirties', '40': 'forties', '50': 'fifties',
            '60': 'sixties', '70': 'seventies', '80': 'eighties',
            '90': 'nineties'
        }
        return decades.get(decade, f"{decade}s")

    @staticmethod
    def _two_digit_year_to_words(num: str) -> str:
        if len(num) != 2:
            return num
        
        num_int = int(num)
        if num_int == 0:
            return "hundred"
        if num_int < 10:
            return f"oh {TextProcessor._digit_to_word(num[1])}"
        
        ones = ['', ' one ', ' two ', ' three ', ' four ', ' five ', ' six ', ' seven ', 
               ' eight ', ' nine ', ' ten ', ' eleven ', ' twelve ', ' thirteen ', 
               ' fourteen ', ' fifteen ', ' sixteen ', ' seventeen ', ' eighteen ', 
               ' nineteen ']
        tens = ['', '', ' twenty ', ' thirty ', ' forty ', ' fifty ', ' sixty ', 
               ' seventy ', ' eighty ', ' ninety ']
        
        if num_int < 20:
            return ones[num_int]
        
        ten, one = divmod(num_int, 10)
        if one == 0:
            return tens[ten]
        return f"{tens[ten]}-{ones[one]}"

    @staticmethod
    def _number_to_words(number: str) -> str:
        if '.' in number:
            integer, decimal = number.split('.')
            integer_words = TextProcessor._integer_to_words(integer)
            decimal_words = ' '.join(TextProcessor._digit_to_word(d) for d in decimal)
            return f"{integer_words} point {decimal_words}"
        else:
            return TextProcessor._integer_to_words(number)

    @staticmethod
    def _integer_to_words(number: str) -> str:
        num_int = int(number)
        if num_int < 1000:
            return TextProcessor._two_digit_year_to_words(number)
        
        thousands = num_int // 1000
        remainder = num_int % 1000
        
        if remainder == 0:
            return f"{TextProcessor._integer_to_words(str(thousands))} thousand"
        return f"{TextProcessor._integer_to_words(str(thousands))} thousand {TextProcessor._two_digit_year_to_words(str(remainder).zfill(3))}"

    @staticmethod
    def _digit_to_word(digit: str) -> str:
        digits = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
            '8': 'eight', '9': 'nine'
        }
        return digits.get(digit, digit)

    @staticmethod
    def _process_phone_numbers(text: str) -> str:
        phone_pattern = r'\b(\d{3})[-. ]?(\d{3})[-. ]?(\d{4})\b'
    
        def phone_to_words(match):
            groups = match.groups()
            parts = []
            for part in groups:
                digits = ' '.join([TextProcessor._digit_to_word(d) for d in part])
                parts.append(digits)
            return ', '.join(parts)
    
        return re.sub(phone_pattern, phone_to_words, text)

    def split_sentences(self, text: str, max_chars: int = 250) -> List[str]:
        if not text:
            return []
    
        abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
            "a.m.", "p.m.", "etc.", "e.g.", "i.e.", "vs.", "No.", "Vol.", "Fig.", "Approx."
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
                chunks = self.smart_text_split(s, max_chars)
                sentences.extend(chunks)
            else:
                sentences.append(s)
    
        return sentences

    @staticmethod
    def parse_dialogues(text: str, prefixes: List[str]) -> List[Tuple[str, str]]:
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
                    processed_content = TextProcessor._process_special_cases(current[1])
                    dialogues.append((current[0], processed_content))
                
                speaker = found_prefix
                content = line[len(found_prefix)+1:].strip()
                current = (speaker, content)
            elif current:
                current = (current[0], current[1] + ' ' + line)
                
        if current:
            processed_content = TextProcessor._process_special_cases(current[1])
            dialogues.append((current[0], processed_content))
            
        return dialogues

class AudioProcessor:
    @staticmethod
    def enhance_audio(audio: np.ndarray, volume: float = 1.0, pitch: float = 1.0) -> np.ndarray:
        max_sample = np.max(np.abs(audio)) + 1e-8
        audio = (audio / max_sample) * 0.9 * volume
        
        audio = np.tanh(audio * 1.5) / 1.5
        
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
            ).set_frame_rate(24000).fade_in(10).fade_out(10)
        
        audio_seg = compress_dynamic_range(
            audio_seg,
            threshold=-12.0,
            ratio=3.5,
            attack=5,
            release=50
        )
        
        if audio_seg.max_dBFS > -1.0:
            audio_seg = audio_seg.apply_gain(-audio_seg.max_dBFS * 0.8)
        
        return np.array(audio_seg.get_array_of_samples()) / 32768.0

    @staticmethod
    def calculate_pause(text: str, pause_settings: Dict[str, int]) -> int:
        text = text.strip()
        if not text:
            return 0
            
        if re.search(r'(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|St|A\.M|P\.M|etc|e\.g|i\.e)\.$', text, re.IGNORECASE):
            return 0
            
        if re.search(r'\b\d{1,2}:\d{2}\b', text):
            return pause_settings.get('time_colon_pause', 50)
            
        last_char = text[-1]
        return pause_settings.get(last_char, pause_settings['default_pause'])

    @staticmethod
    def combine_segments(segments: List[AudioSegment], pauses: List[int]) -> AudioSegment:
        combined = AudioSegment.silent(duration=0)
        
        for i, (seg, pause) in enumerate(zip(segments, pauses)):
            seg = seg.fade_in(10).fade_out(10)
            
            combined += seg
            
            if i < len(segments) - 1:
                combined += AudioSegment.silent(duration=max(50, pause))
        
        return combined

class SubtitleGenerator:
    @staticmethod
    def split_long_sentences(text: str, max_length: int = 150) -> List[str]:
        text = text.strip()
        if len(text) <= max_length:
            return [text]
    
        sentences = []
        current_text = text
        
        while current_text and len(current_text) > max_length:
            split_pos = -1
            
            end_punct_match = re.search(r'[.!?][)\]\'"\s]*', current_text[:max_length])
            if end_punct_match:
                split_pos = end_punct_match.end()
            
            if split_pos == -1:
                comma_pos = current_text.rfind(',', 0, max_length)
                if comma_pos > 0:
                    if comma_pos > 0 and comma_pos < len(current_text) - 1:
                        next_char = current_text[comma_pos + 1]
                        if not next_char.isdigit():
                            split_pos = comma_pos + 1
            
            if split_pos == -1:
                space_pos = current_text.rfind(' ', 0, max_length)
                if space_pos > 0:
                    split_pos = space_pos + 1
                else:
                    split_pos = max_length
            
            part = current_text[:split_pos].strip()
            if part:
                sentences.append(part)
            current_text = current_text[split_pos:].strip()
        
        if current_text:
            sentences.append(current_text)
        
        return sentences

    @staticmethod
    def clean_subtitle_text(text: str) -> str:
        cleaned = re.sub(r'^(Q|A|CHAR\d+):\s*', '', text.strip())
        return cleaned

    @staticmethod
    def generate_srt(audio_segments: List[AudioSegment], sentences: List[str], pause_settings: Dict[str, int]) -> str:
        subtitles = []
        current_time_ms = 0
        srt_index = 1

        for i, (audio_seg, original_sentence) in enumerate(zip(audio_segments, sentences)):
            cleaned_sentence = SubtitleGenerator.clean_subtitle_text(original_sentence)
            
            segment_duration_ms = len(audio_seg)
            
            text_chunks = SubtitleGenerator.split_long_sentences(cleaned_sentence, 150)
            
            if not text_chunks:
                continue

            total_chars = sum(len(re.sub(r'\s+', ' ', chunk).strip()) for chunk in text_chunks)
            
            for j, chunk in enumerate(text_chunks):
                clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
                chunk_chars = len(clean_chunk)
                
                if total_chars == 0:
                    chunk_duration_ms = segment_duration_ms / len(text_chunks)
                else:
                    chunk_duration_ms = int(segment_duration_ms * (chunk_chars / total_chars))
                
                chunk_duration_ms = max(500, chunk_duration_ms)
                
                start_ms = current_time_ms
                end_ms = start_ms + chunk_duration_ms
                
                start_td = timedelta(milliseconds=start_ms)
                end_td = timedelta(milliseconds=end_ms)
                
                hours_start = int(start_td.total_seconds() // 3600)
                minutes_start = int((start_td.total_seconds() % 3600) // 60)
                seconds_start = start_td.total_seconds() % 60
                milliseconds_start = int((seconds_start - int(seconds_start)) * 1000)
                
                hours_end = int(end_td.total_seconds() // 3600)
                minutes_end = int((end_td.total_seconds() % 3600) // 60)
                seconds_end = end_td.total_seconds() % 60
                milliseconds_end = int((seconds_end - int(seconds_end)) * 1000)
                
                start_str = f"{hours_start:02d}:{minutes_start:02d}:{int(seconds_start):02d},{milliseconds_start:03d}"
                end_str = f"{hours_end:02d}:{minutes_end:02d}:{int(seconds_end):02d},{milliseconds_end:03d}"
                
                subtitles.append({
                    'index': srt_index,
                    'start_str': start_str,
                    'end_str': end_str,
                    'text': chunk.strip()
                })
                srt_index += 1
                
                current_time_ms += chunk_duration_ms
            
            if i < len(audio_segments) - 1:
                pause_ms = AudioProcessor.calculate_pause(original_sentence, pause_settings)
                current_time_ms += pause_ms

        srt_lines = []
        for sub in subtitles:
            srt_lines.append(f"{sub['index']}")
            srt_lines.append(f"{sub['start_str']} --> {sub['end_str']}")
            srt_lines.append(f"{sub['text']}")
            srt_lines.append("")

        return "\n".join(srt_lines)

class TTSGenerator:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.tokenizer = Tokenizer()
        self.subtitle_generator = SubtitleGenerator()
        self.voices_dir = "voices"
        
        # Tạo thư mục voices nếu chưa tồn tại
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir)
            st.info(f"Created voices directory: {self.voices_dir}")

    def generate_sentence_audio(self, sentence: str, voice: str, speed: float,
                             device: str, volume: float = 1.0, pitch: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        try:
            # Tạo file voice mẫu nếu chưa có
            voice_path = os.path.join(self.voices_dir, f"{voice}.pt")
            if not os.path.exists(voice_path):
                # Tạo embedding giả lập
                dummy_embedding = torch.randn(1, 192)  # Kích thước embedding giả lập
                torch.save(dummy_embedding, voice_path)
                st.info(f"Created sample voice: {voice}")
            
            # Tạo audio giả lập cho demo
            # Trong thực tế, bạn sẽ gọi model TTS thật ở đây
            duration = len(sentence) * 0.05  # 50ms mỗi ký tự
            sample_rate = 24000
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Tạo tone cơ bản
            frequency = 220.0 * (2 ** ((pitch - 1.0) * 12))  # Điều chỉnh pitch
            audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Thêm một số harmonic
            for harmonic in range(2, 6):
                audio += 0.1 * np.sin(2 * np.pi * frequency * harmonic * t) / harmonic
            
            # Áp dụng volume
            audio *= volume
            
            # Điều chỉnh speed
            if speed != 1.0:
                from scipy import interpolate
                original_length = len(audio)
                new_length = int(original_length / speed)
                x_original = np.linspace(0, 1, original_length)
                x_new = np.linspace(0, 1, new_length)
                interpolator = interpolate.interp1d(x_original, audio, kind='linear')
                audio = interpolator(x_new)
                sample_rate = int(sample_rate * speed)
            
            return (sample_rate, audio)
                
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            # Trả về audio silent nếu có lỗi
            sample_rate = 24000
            duration = 1.0  # 1 giây silent
            audio = np.zeros(int(sample_rate * duration))
            return (sample_rate, audio)

    def generate_story_audio(self, text: str, voice: str, speed: float, device: str,
                           pause_settings: Dict[str, int], volume: float = 1.0, 
                           pitch: float = 1.0, max_chars_per_segment: int = 250) -> Tuple[Tuple[int, np.ndarray], str, str]:
        start_time = time.time()
        clean_text = self.text_processor.clean_text(text)
        
        sentences = self.text_processor.split_sentences(clean_text, max_chars_per_segment)
        
        if not sentences:
            return None, "No content to read", ""
        
        audio_segments = []
        pause_durations = []
        
        speed_factor = max(0.5, min(2.0, speed))
        adjusted_pause_settings = {
            'default_pause': int(pause_settings['default_pause'] / speed_factor),
            'dot_pause': int(pause_settings['dot_pause'] / speed_factor),
            'ques_pause': int(pause_settings['ques_pause'] / speed_factor),
            'comma_pause': int(pause_settings['comma_pause'] / speed_factor),
            'colon_pause': int(pause_settings['colon_pause'] / speed_factor),
            'excl_pause': int(pause_settings['dot_pause'] / speed_factor),
            'semi_pause': int(pause_settings['colon_pause'] / speed_factor),
            'dash_pause': int(pause_settings['comma_pause'] / speed_factor),
            'time_colon_pause': 50
        }
        
        for sentence in sentences:
            result = self.generate_sentence_audio(sentence, voice, speed, device, volume, pitch)
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
        
        if not audio_segments:
            return None, "Failed to generate audio", ""
        
        combined_audio = self.audio_processor.combine_segments(audio_segments, pause_durations)
        
        with io.BytesIO() as buffer:
            combined_audio.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", str(combined_audio.frame_rate)])
            buffer.seek(0)
            audio_data = np.frombuffer(buffer.read(), dtype=np.uint8)
        
        subtitles = self.subtitle_generator.generate_srt(audio_segments, sentences, adjusted_pause_settings)
        
        stats = (f"Processed {len(clean_text)} chars, {len(clean_text.split())} words\n"
                f"Audio duration: {len(combined_audio)/1000:.2f}s\n"
                f"Time: {time.time() - start_time:.2f}s\n"
                f"Device: {device.upper()}")
        
        return (combined_audio.frame_rate, audio_data), stats, subtitles

    def generate_qa_audio(self, text: str, voice_q: str, voice_a: str, speed_q: float, speed_a: float,
                         device: str, repeat_times: int, pause_q: int, pause_a: int,
                         volume_q: float = 1.0, volume_a: float = 1.0,
                         pitch_q: float = 1.0, pitch_a: float = 1.0) -> Tuple[Tuple[int, np.ndarray], str, str]:
        start_time = time.time()
        dialogues = self.text_processor.parse_dialogues(text, ['Q', 'A'])
        
        if not dialogues:
            return None, "No Q/A content found", ""
        
        combined = AudioSegment.empty()
        timing_info = []
        current_pos = 0
        
        qa_pairs = []
        current_q = None
        for speaker, content in dialogues:
            if speaker.upper() == 'Q':
                current_q = (content, [])
            elif speaker.upper() == 'A' and current_q:
                current_q[1].append(content)
                qa_pairs.append(current_q)
                current_q = None
        
        for q_text, a_texts in qa_pairs:
            q_result = self.generate_sentence_audio(q_text, voice_q, speed_q, device, volume_q, pitch_q)
            if not q_result:
                continue
                
            sr_q, q_audio = q_result
            q_seg = AudioSegment(
                (q_audio * 32767).astype(np.int16).tobytes(),
                frame_rate=sr_q,
                sample_width=2,
                channels=1
            ).fade_in(10).fade_out(10)
            
            a_text = a_texts[0] if a_texts else ""
            a_result = self.generate_sentence_audio(a_text, voice_a, speed_a, device, volume_a, pitch_a)
            if not a_result:
                continue
                
            sr_a, a_audio = a_result
            a_seg = AudioSegment(
                (a_audio * 32767).astype(np.int16).tobytes(),
                frame_rate=sr_a,
                sample_width=2,
                channels=1
            ).fade_in(10).fade_out(10)
            
            for i in range(repeat_times):
                q_start = current_pos
                q_end = q_start + len(q_seg)
                combined += q_seg
                timing_info.append({
                    'start': q_start,
                    'end': q_end,
                    'text': q_text
                })
                
                current_pos = q_end + pause_q
                combined += AudioSegment.silent(duration=pause_q)
                
                a_start = current_pos
                a_end = a_start + len(a_seg)
                combined += a_seg
                timing_info.append({
                    'start': a_start,
                    'end': a_end,
                    'text': a_text
                })
                
                if i < repeat_times - 1:
                    combined += AudioSegment.silent(duration=pause_a)
                    current_pos = a_end + pause_a
                else:
                    current_pos = a_end
        
        if len(combined) == 0:
            return None, "Failed to generate audio", ""
        
        with io.BytesIO() as buffer:
            combined.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", "24000"])
            audio_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        
        subtitles = []
        for idx, info in enumerate(timing_info, 1):
            start_str = f"{info['start']//3600000:02d}:{(info['start']%3600000)//60000:02d}:{(info['start']%60000)//1000:02d},{info['start']%1000:03d}"
            end_str = f"{info['end']//3600000:02d}:{(info['end']%3600000)//60000:02d}:{(info['end']%60000)//1000:02d},{info['end']%1000:03d}"
            
            subtitles.append(
                f"{idx}\n"
                f"{start_str} --> {end_str}\n"
                f"{info['text']}\n"
            )
        
        stats = (f"Generated {len(qa_pairs)} Q/A pairs | "
                f"Repeated {repeat_times}x | "
                f"Duration: {len(combined)/1000:.2f}s | "
                f"Q: {speed_q:.1f}x | A: {speed_a:.1f}x | "
                f"Processing time: {time.time()-start_time:.2f}s")
        
        return (24000, audio_data), stats, "\n".join(subtitles)

def main():
    st.title("🎙️ Advanced TTS System Demo")
    st.markdown("---")
    
    # Khởi tạo generator
    generator = TTSGenerator()
    
    # Lấy danh sách voice
    voices_dir = "voices"
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)
    
    voice_files = [f.replace('.pt', '') for f in os.listdir(voices_dir) if f.endswith('.pt')]
    if not voice_files:
        # Tạo voices mẫu
        for i in range(5):
            voice_name = f"Voice_{i+1}"
            voice_path = os.path.join(voices_dir, f"{voice_name}.pt")
            dummy_embedding = torch.randn(1, 192)
            torch.save(dummy_embedding, voice_path)
            voice_files.append(voice_name)
    
    # Tạo tabs
    tab1, tab2 = st.tabs(["Standard TTS", "Q&A TTS"])
    
    with tab1:
        st.subheader("Standard Text-to-Speech")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text to convert to speech:",
                value="Hello! This is a demonstration of the TTS system. You can enter any text here and it will be converted to speech.",
                height=200
            )
            
            voice = st.selectbox("Select Voice", voice_files, index=0)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
            with col_s2:
                volume = st.slider("Volume", 0.0, 2.0, 1.0, 0.1)
            with col_s3:
                pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
            
            device = st.radio("Device", ["CPU", "GPU"] if CUDA_AVAILABLE else ["CPU"])
        
        with col2:
            st.subheader("Pause Settings")
            default_pause = st.slider("Default pause (ms)", 0, 1000, 200)
            dot_pause = st.slider("After period (ms)", 0, 2000, 600)
            comma_pause = st.slider("After comma (ms)", 0, 1000, 300)
        
        if st.button("Generate Speech", type="primary", use_container_width=True):
            with st.spinner("Generating audio..."):
                pause_settings = {
                    'default_pause': default_pause,
                    'dot_pause': dot_pause,
                    'ques_pause': dot_pause,
                    'comma_pause': comma_pause,
                    'colon_pause': default_pause,
                    'excl_pause': dot_pause,
                    'semi_pause': default_pause,
                    'dash_pause': comma_pause,
                    'time_colon_pause': 50
                }
                
                result, stats, subtitles = generator.generate_story_audio(
                    text_input, voice, speed, device.lower(), 
                    pause_settings, volume, pitch
                )
                
                if result:
                    sample_rate, audio_data = result
                    
                    # Save to file
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    filepath = os.path.join(output_dir, "output.mp3")
                    with open(filepath, "wb") as f:
                        f.write(audio_data.tobytes())
                    
                    # Display audio
                    st.audio(filepath, format="audio/mp3")
                    
                    # Display stats
                    st.success("Audio generated successfully!")
                    st.text_area("Processing Stats", stats, height=100)
                    
                    # Download button
                    with open(filepath, "rb") as f:
                        st.download_button(
                            "Download Audio",
                            f,
                            file_name="tts_output.mp3",
                            mime="audio/mp3"
                        )
                else:
                    st.error("Failed to generate audio")
    
    with tab2:
        st.subheader("Q&A Text-to-Speech")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            qa_input = st.text_area(
                "Enter Q&A format text:",
                value="Q: What is your name?\nA: My name is TTS Assistant.\n\nQ: What can you do?\nA: I can convert text to speech with different voices and settings.",
                height=200
            )
            
            col_q, col_a = st.columns(2)
            with col_q:
                voice_q = st.selectbox("Question Voice", voice_files, index=0, key="voice_q")
                speed_q = st.slider("Q Speed", 0.5, 2.0, 1.0, 0.1, key="speed_q")
                volume_q = st.slider("Q Volume", 0.0, 2.0, 1.0, 0.1, key="volume_q")
            
            with col_a:
                voice_a = st.selectbox("Answer Voice", voice_files, index=1 if len(voice_files) > 1 else 0, key="voice_a")
                speed_a = st.slider("A Speed", 0.5, 2.0, 1.0, 0.1, key="speed_a")
                volume_a = st.slider("A Volume", 0.0, 2.0, 1.0, 0.1, key="volume_a")
        
        with col2:
            st.subheader("Q&A Settings")
            repeat_times = st.slider("Repeat times", 1, 5, 1)
            pause_q = st.slider("Pause after Q (ms)", 0, 2000, 500)
            pause_a = st.slider("Pause after A (ms)", 0, 2000, 800)
            device_qa = st.radio("Device", ["CPU", "GPU"] if CUDA_AVAILABLE else ["CPU"], key="device_qa")
        
        if st.button("Generate Q&A Audio", type="primary", use_container_width=True):
            with st.spinner("Generating Q&A audio..."):
                result, stats, subtitles = generator.generate_qa_audio(
                    qa_input, voice_q, voice_a, speed_q, speed_a,
                    volume_q, 1.0, volume_a, 1.0,
                    device_qa.lower(), repeat_times, pause_q, pause_a
                )
                
                if result:
                    sample_rate, audio_data = result
                    
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    filepath = os.path.join(output_dir, "qa_output.mp3")
                    with open(filepath, "wb") as f:
                        f.write(audio_data.tobytes())
                    
                    st.audio(filepath, format="audio/mp3")
                    st.success("Q&A audio generated successfully!")
                    st.text_area("Processing Stats", stats, height=100)
                    
                    with open(filepath, "rb") as f:
                        st.download_button(
                            "Download Q&A Audio",
                            f,
                            file_name="qa_tts_output.mp3",
                            mime="audio/mp3"
                        )
                else:
                    st.error("Failed to generate Q&A audio")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Advanced TTS System Demo | Made with Streamlit</p>
        <p><small>Note: This is a demo version. In production, real TTS models would be used.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
