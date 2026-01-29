import streamlit as st
import torch
import numpy as np
import io
import time
import re
import json
import os
import wave
from typing import List, Tuple, Optional, Dict
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from phonemizer import backend
from datetime import timedelta

# Import your custom modules
from vansarah import KModel, KPipeline
from SPECIAL_CASES import SPECIAL_CASES

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
    
        # --- READ --- (Sửa để xử lý "today" chính xác)
        # 1. Xử lý CỨNG các cặp hỏi-đáp
        text = re.sub(r'What did you read', 'What did you reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did he read', 'What did he reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did she read', 'What did she reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did we read', 'What did we reed', text, flags=re.IGNORECASE)
        text = re.sub(r'What did they read', 'What did they reed', text, flags=re.IGNORECASE)
        # Sometimes ... read -> reed (thói quen hiện tại)
        text = re.sub(r"\b(Sometimes|sometimes)\s+I\s+read\b", r"\1 I reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+we\s+read\b", r"\1 we reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+they\s+read\b", r"\1 they reed", text)
        text = re.sub(r"\b(Sometimes|sometimes)\s+you\s+read\b", r"\1 you reed", text)
        # "to read" (infinitive) -> reed
        text = re.sub(r"\bto\s+read\b", "to reed", text, flags=re.IGNORECASE)
        # Present simple (I/you/we/they read ...) -> reed
        text = re.sub(r"\b(I|You|We|They)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)

        # Present simple (He/She/It reads ...) -> reeds
        text = re.sub(r"\b(He|She|It)\s+reads\b", r"\1 reeds", text, flags=re.IGNORECASE)
      
        
        text = re.sub(r'^I read a', 'I red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^He read a', 'He red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^She read a', 'She red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^We read a', 'We red a', text, flags=re.IGNORECASE)
        text = re.sub(r'^They read a', 'They red a', text, flags=re.IGNORECASE)

        # Imperative (mệnh lệnh) -> reed
        text = re.sub(r"(^|\.\s+)(Please\s+)?Read\b", r"\1Reed", text)
        text = re.sub(r"(^|\.\s+)Read\b", r"\1Reed", text)


        # Phủ định hiện tại đơn
        text = re.sub(r"\b(don't|do not)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(doesn't|does not)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)


        # Nghi vấn hiện tại đơn
        text = re.sub(r"\bDo\s+(you|we|they)\s+read\b", r"Do \1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bDoes\s+(he|she|it)\s+read\b", r"Does \1 reeds", text, flags=re.IGNORECASE)

        # Bị động
        text = re.sub(r"\b(is|are)\s+read\b", r"\1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(was|were)\s+read\b", r"\1 red", text, flags=re.IGNORECASE)

        # Trạng từ tần suất thêm (often, rarely, seldom)
        text = re.sub(r"\b(I|You|We|They)\s+(often|rarely|seldom)\s+read\b", r"\1 \2 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(He|She|It)\s+(often|rarely|seldom)\s+reads\b", r"\1 \2 reeds", text, flags=re.IGNORECASE)

        # If ... read (hiện tại)
        text = re.sub(r"\bIf\s+(I|You|We|They)\s+read\b", r"If \1 reed", text, flags=re.IGNORECASE)
        text = re.sub(r"\bIf\s+(He|She|It)\s+reads\b", r"If \1 reeds", text, flags=re.IGNORECASE)

        
        # 1a. Câu hỏi với "did ... read" -> reed
        text = re.sub(r'\b(did)\s+([^.!?]*?)\bread\b', r'\1 \2reed', text, flags=re.IGNORECASE)
        text = re.sub(r"\bdidn't\s+read\b", "didn't reed", text, flags=re.IGNORECASE)
        
        # 1b. Với trợ động từ quá khứ + read
        past_auxiliaries = r'\b(had|was|were|have|has|haven\'t|hasn\'t|hadn\'t|wasn\'t|weren\'t)'
        text = re.sub(rf'{past_auxiliaries}\s+([^.!?]*?)\bread\b', r'\1 \2red', text, flags=re.IGNORECASE)
        
        # 1c. Với trạng từ thời gian quá khứ + read
        past_time_words = r'\b(yesterday|last\s+(night|week|month|year)|(\d+\s+)?(days|weeks|months|years)\s+ago|already|just|earlier|before|previously|recently|when\s+I\s+was)'
        text = re.sub(rf'{past_time_words}[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' red').replace(' Read', ' Red'),
                     text, flags=re.IGNORECASE)
        text = re.sub(rf'\bread\b[^.!?]*?{past_time_words}',
                     lambda m: m.group(0).replace('read ', 'red ').replace('Read ', 'Red '),
                     text, flags=re.IGNORECASE)
        
        # 1d. Xử lý "today" RIÊNG BIỆT - mặc định là HIỆN TẠI (reed)
        text = re.sub(r'\b(today)[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' reed').replace(' Read', ' Reed'),
                     text, flags=re.IGNORECASE)
        text = re.sub(r'\bread\b[^.!?]*?\b(today)\b',
                     lambda m: m.group(0).replace('read ', 'reed ').replace('Read ', 'Reed '),
                     text, flags=re.IGNORECASE)
        
        # 2. Các trường hợp hiện tại/tương lai -> reed (LOẠI BỎ "today" và "now")
        future_present_words = r'\b(will|shall|going to|plan to|want to|need to|would like to|tomorrow|next|every|always|usually|now)'
        text = re.sub(rf'{future_present_words}[^.!?]*?\bread\b',
                     lambda m: m.group(0).replace(' read', ' reed').replace(' Read', ' Reed'),
                     text, flags=re.IGNORECASE)
        text = re.sub(rf'\bread\b[^.!?]*?{future_present_words}',
                     lambda m: m.group(0).replace('read ', 'reed ').replace('Read ', 'Reed '),
                     text, flags=re.IGNORECASE)
        
        # 3. Mặc định còn lại -> red nếu toàn chuyện là quá khứ, reed nếu không
        def replace_remaining_read(match):
            word = match.group(0)
            if getattr(self, 'is_past_story', False):  # nếu câu chuyện là quá khứ để  True, nếu không phải để False
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
    
        # Chỉ Space + Zero Width Space để tách vừa đủ
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
            # Thêm riêng cho a.m. / p.m.
            r"\ba\.m\.?\b": "AM",
            r"\bp\.m\.?\b": "PM",      
        
            # Viết tắt chung
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

            
            # World Wars
            r"\bWorld War I\b": "World War One",
            r"\bWorld War II\b": "World War Two",
            r"\bWorld War III\b": "World War Three",   # Optional
            r"\bWorld War IV\b": "World War Four",     # Optional
            
            # Roman Emperors & Byzantine rulers
            r"\bTheodosius I\b": "Theodosius the First",
            r"\bTheodosius II\b": "Theodosius the Second",
            r"\bConstantine I\b": "Constantine the First",
            r"\bConstantine II\b": "Constantine the Second",
            r"\bConstantine III\b": "Constantine the Third",
            r"\bJustinian I\b": "Justinian the First",
            r"\bJustinian II\b": "Justinian the Second",
            r"\bAlexander I\b": "Alexander the First",
            r"\bAlexander II\b": "Alexander the Second",
            r"\bAlexander III\b": "Alexander the Third",
            
            # English monarchs
            r"\bHenry I\b": "Henry the First",
            r"\bHenry II\b": "Henry the Second",
            r"\bHenry III\b": "Henry the Third",
            r"\bHenry IV\b": "Henry the Fourth",
            r"\bHenry V\b": "Henry the Fifth",
            r"\bHenry VI\b": "Henry the Sixth",
            r"\bHenry VII\b": "Henry the Seventh",
            r"\bHenry VIII\b": "Henry the Eighth",
            r"\bEdward I\b": "Edward the First",
            r"\bEdward II\b": "Edward the Second",
            r"\bEdward III\b": "Edward the Third",
            r"\bEdward IV\b": "Edward the Fourth",
            r"\bEdward V\b": "Edward the Fifth",
            r"\bEdward VI\b": "Edward the Sixth",
            r"\bEdward VII\b": "Edward the Seventh",
            r"\bEdward VIII\b": "Edward the Eighth",
            r"\bCharles I\b": "Charles the First",
            r"\bCharles II\b": "Charles the Second",
            r"\bCharles III\b": "Charles the Third",
            r"\bJames I\b": "James the First",
            r"\bJames II\b": "James the Second",
            r"\bWilliam I\b": "William the First",
            r"\bWilliam II\b": "William the Second",
            r"\bElizabeth I\b": "Elizabeth the First",
            r"\bElizabeth II\b": "Elizabeth the Second",
            
            # French monarchs
            r"\bLouis I\b": "Louis the First",
            r"\bLouis II\b": "Louis the Second",
            r"\bLouis III\b": "Louis the Third",
            r"\bLouis IV\b": "Louis the Fourth",
            r"\bLouis V\b": "Louis the Fifth",
            r"\bLouis VI\b": "Louis the Sixth",
            r"\bLouis VII\b": "Louis the Seventh",
            r"\bLouis VIII\b": "Louis the Eighth",
            r"\bLouis IX\b": "Louis the Ninth",
            r"\bLouis X\b": "Louis the Tenth",
            r"\bLouis XI\b": "Louis the Eleventh",
            r"\bLouis XII\b": "Louis the Twelfth",
            r"\bLouis XIII\b": "Louis the Thirteenth",
            r"\bLouis XIV\b": "Louis the Fourteenth",
            r"\bLouis XV\b": "Louis the Fifteenth",
            r"\bLouis XVI\b": "Louis the Sixteenth",
            r"\bPhilip II\b": "Philip the Second",
            r"\bPhilip IV\b": "Philip the Fourth",
            
            # Russian monarchs
            r"\bNicholas I\b": "Nicholas the First",
            r"\bNicholas II\b": "Nicholas the Second",
            r"\bPeter I\b": "Peter the First",
            r"\bPeter II\b": "Peter the Second",
            r"\bPeter III\b": "Peter the Third",
            r"\bCatherine II\b": "Catherine the Second",
            
            # Popes
            r"\bPope John Paul I\b": "Pope John Paul the First",
            r"\bPope John Paul II\b": "Pope John Paul the Second",
            r"\bPope Benedict XVI\b": "Pope Benedict the Sixteenth",
            r"\bPope Pius XII\b": "Pope Pius the Twelfth",
            r"\bPope Leo XIII\b": "Pope Leo the Thirteenth",
            r"\bPope Innocent III\b": "Pope Innocent the Third",
            r"\bPope Gregory XIII\b": "Pope Gregory the Thirteenth",
            
            # Other famous Roman numeral events/titles
            r"\bSuper Bowl I\b": "Super Bowl One",
            r"\bSuper Bowl II\b": "Super Bowl Two",
            r"\bSuper Bowl III\b": "Super Bowl Three",
            r"\bSuper Bowl IV\b": "Super Bowl Four",
            r"\bSuper Bowl V\b": "Super Bowl Five",
            r"\bSuper Bowl X\b": "Super Bowl Ten",
            r"\bSuper Bowl XX\b": "Super Bowl Twenty",
            r"\bFinal Fantasy VII\b": "Final Fantasy Seven",
            r"\bFinal Fantasy VIII\b": "Final Fantasy Eight",
            r"\bFinal Fantasy IX\b": "Final Fantasy Nine",
            r"\bFinal Fantasy X\b": "Final Fantasy Ten" ,        
    
            # Số thứ tự
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
    
            # Số đếm
            # Số đếm 0–20
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
    
            # 21–29
            r"\b21\b": "twenty one ",
            r"\b22\b": "twenty two ",
            r"\b23\b": "twenty three ",
            r"\b24\b": "twenty four ",
            r"\b25\b": "twenty five ",
            r"\b26\b": "twenty six ",
            r"\b27\b": "twenty seven ",
            r"\b28\b": "twenty eight ",
            r"\b29\b": "twenty nine ",
    
            # 30–39
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
    
            # 40–49
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
    
            # 50–59
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
    
            # 60–69
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
    
            # 70–79
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
    
            # 80–89
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
    
            # 90–99
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
    
            # 100
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
        # Chỉ thêm ZWS TRƯỚC từ nối, không thêm SAU
        return re.sub(pattern, lambda m: f"{ZWS}{m.group(0)}", text, flags=re.IGNORECASE)


    def process_text(self, text: str) -> str:
        """Chuẩn hóa văn bản giữ nguyên ngữ cảnh và thêm pause giữa các từ"""

        # Bước -1: Chuẩn hóa dấu nháy cong thành thẳng
        text = text.replace("'", "'").replace("'", "'").replace("'", "'")

        # Bước -0.5: Bỏ các dấu ngoặc kép kiểu " " và "
        text = text.replace('"', '').replace('"', '').replace('"', '')

        # 🔥 Chuẩn hóa các từ viết tắt có dấu chấm để tránh TTS pause
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


        # 🔥 THÊM: Thay thế ; và : thành . để tạo điểm dừng
        text = text.replace(';', '.').replace(':', '.')

        # 🔥 Sửa lỗi "banana" trước khi xử lý tiếp
        #text = re.sub(r'\bbananas\b', 'buh-nan-uhs', text, flags=re.IGNORECASE)
        #text = re.sub(r'\bbanana\b', 'buh-nan-uh', text, flags=re.IGNORECASE)

        # 🔥 Sửa lỗi "photo" (American English: FOH-toh)
        text = re.sub(r'\bphotos\b', 'foh-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\bphoto\b', 'foh-toh', text, flags=re.IGNORECASE)

        # 🔥 Sửa lỗi "tomato" (American English: tuh-MAY-toh)
        text = re.sub(r'\btomatoes\b', 'tuh-may-tohz', text, flags=re.IGNORECASE)
        text = re.sub(r'\btomato\b', 'tuh-may-toh', text, flags=re.IGNORECASE)

        text = re.sub(r'\bLos Angeles\b', 'Loss an-juh-luhs', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAngeles\b', 'an-juh-luhs', text, flags=re.IGNORECASE)


        # Bước 0.5: Xử lý heteronyms (live, lead, read)
        text = self._fix_heteronyms(text)

        # Bước 1: Xử lý từ viết tắt
        for pattern, replacement in self.abbreviation_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Bước 2: Chuẩn hóa từ đặc biệt (giữ nguyên case)
        text = self.apply_special_cases(text)

        # Bước 4: Chuẩn hóa khoảng trắng (không xoá mất câu ngắn)
        text = re.sub(r'[ \t]+', ' ', text)         # gom khoảng trắng thừa
        text = re.sub(r'\s*\n\s*', '. ', text)      # xuống dòng => chấm + space
        text = text.strip()

        # Bước 3.5: Thêm khoảng vô hình quanh các từ nối
        text = self._add_invisible_space_to_conjunctions(text)

        # 🔥 Sửa mạo từ "a" cho tự nhiên và rõ
        #text = re.sub(r"\ba\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", r"uh \1", text)


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
        """
        Chia văn bản thành các đoạn ngắn khoảng max_length ký tự,
        ưu tiên tách tại các dấu chấm câu tự nhiên, tránh tách tại a.m, p.m, Mr., ...
        """
        if len(text) <= max_length:
            return [text]
    
        sentences = []
        current_text = text.strip()
    
        # Danh sách các từ viết tắt (chữ thường để so sánh)
        abbreviations = {
            "a.m.", "p.m.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.",
            "etc.", "e.g.", "i.e.", "vs.", "approx.", "no.", "vol.", "fig.", "p."
        }
    
        while current_text and len(current_text) > max_length:
            ideal_split_pos = -1
            last_period_pos = current_text.rfind('.', 0, max_length)
    
            # Ưu tiên 1: Tìm dấu chấm (kiểm tra viết tắt)
            if last_period_pos > 0:
                # Lấy 10 ký tự trước dấu chấm
                prev_text = current_text[max(0, last_period_pos - 10):last_period_pos + 1].lower().strip()
                # Nếu không phải viết tắt → cho tách
                if not any(prev_text.endswith(abbr) for abbr in abbreviations):
                    ideal_split_pos = last_period_pos + 1
    
            # Ưu tiên 2: Tìm dấu ! ? ; :
            if ideal_split_pos == -1:
                for punct in ['!', '?', ';', ':']:
                    punct_pos = current_text.rfind(punct, 0, max_length)
                    if punct_pos > 0:
                        ideal_split_pos = punct_pos + 1
                        break
    
            # ⚠️ QUAN TRỌNG: Thêm xử lý dấu phẩy
            if ideal_split_pos == -1:
                comma_pos = current_text.rfind(',', 0, max_length)
                if comma_pos > 0:
                    ideal_split_pos = comma_pos + 1
    
            # Ưu tiên cuối: cắt theo khoảng trắng
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
        """Pipeline xử lý đặc biệt với thứ tự tối ưu"""
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
        """Process emails with correct English pronunciation for all special characters"""
        def convert_email(match):
            full_email = match.group(0)
            # Replace each special character with its English pronunciation
            processed = (full_email
                        .replace('@', ' at ')
                        .replace('.', ' dot ')
                        .replace('-', ' dash ')
                        .replace('_', ' underscore ')
                        .replace('+', ' plus ')
                        .replace('/', ' slash ')
                        .replace('=', ' equals '))
            return processed

        # Regex to match all email formats
        email_pattern = r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b'
        return re.sub(email_pattern, convert_email, text)

    @staticmethod
    def _process_websites(text: str) -> str:
        """Process websites with correct English pronunciation for special characters"""
        def convert_website(match):
            url = match.group(1)
            # Replace each special character with its English pronunciation
            return (url.replace('.', ' dot ')
                     .replace('-', ' dash ')
                     .replace('_', ' underscore ')
                     .replace('/', ' slash ')
                     .replace('?', ' question mark ')
                     .replace('=', ' equals ')
                     .replace('&', ' ampersand '))

        # Only process websites that don't contain @ (to avoid conflict with emails)
        website_pattern = r'\b(?![\w.-]*@)((?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:[/?=&#][\w.-]*)*)\b'
        return re.sub(website_pattern, convert_website, text, flags=re.IGNORECASE)

    @staticmethod
    def _process_temperatures(text: str) -> str:
        """Process temperatures and cardinal directions with degree symbols"""
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
                '': 'degrees'  # Default case for just number with degree symbol
            }
            unit_text = unit_map.get(unit, f'degrees {unit}')
            
            return f"{temp_text} {unit_text}"
        
        # Process formats like 75°F, 100°C, 15°N, 120°E
        text = re.sub(
            r'(-?\d+)°([NSEWCFnsewcf]?)',
            lambda m: temp_to_words(m.group(1), m.group(2)),
            text,
            flags=re.IGNORECASE
        )
        
        # Add degree symbol pronunciation when standalone
        text = re.sub(r'°', ' degrees ', text)
        
        return text

    @staticmethod
    def _process_measurements(text: str) -> str:
        """Xử lý đơn vị đo lường, đọc chuẩn số thập phân"""
        
        units_map = {
            'km/h': 'kilometers per hour',
            'mph': 'miles per hour',
            'kg': 'kilograms',
            'g': 'grams',
            'cm': 'centimeters',
            'm': 'meters',  # Đã sửa thành số nhiều
            'mm': 'millimeters',
            'L': 'liters',
            'l': 'liters',
            'ml': 'milliliters',
            'mL': 'milliliters',
            'h': 'hours',
            'min': 'minutes'
        }

        # ✅ Xử lý số thập phân với đơn vị: 1.65m → one point six five meters
        for unit, word in units_map.items():
            # Pattern cho số nguyên và số thập phân
            pattern = rf'(\d+(?:\.\d+)?)\s*{unit}\b'
            text = re.sub(pattern, lambda m: f"{TextProcessor._number_to_words(m.group(1))} {word}", text)
        
        # ✅ Xử lý riêng "s" thành seconds (chỉ khi có khoảng trắng trước)
        text = re.sub(r'(\d+)\s+s\b', lambda m: f"{TextProcessor._number_to_words(m.group(1))} seconds", text)
        
        return text
    
        def measurement_to_words(value, unit):
            try:
                unit_lower = unit.lower()
                unit_text = units_map.get(unit, units_map.get(unit_lower, unit))
    
                # Đọc số thập phân: one point six five
                if '.' in value:
                    integer, decimal = value.split('.')
                    value_text = (
                        f"{TextProcessor._number_to_words(integer)} "
                        f"point {' '.join(TextProcessor._digit_to_word(d) for d in decimal)}"
                    )
                else:
                    value_text = TextProcessor._number_to_words(value)
    
                # Xử lý số nhiều (thêm 's' nếu value != 1 và đơn vị không nằm trong plural_units)
                if float(value) != 1 and unit in units_map and unit not in plural_units:
                    unit_text += 's'
    
                return f"{value_text} {unit_text}"
            except:
                return f"{value}{unit}"  # Giữ nguyên nếu có lỗi
    
        # Regex bắt các số + đơn vị (kể cả viết liền như 1.65m)
        text = re.sub(
            r'(-?\d+\.?\d*)\s*({})s?\b'.format('|'.join(re.escape(key) for key in units_map.keys())),
            lambda m: measurement_to_words(m.group(1), m.group(2)),
            text,
            flags=re.IGNORECASE
        )
        return text
    
    @staticmethod
    def _process_currency(text: str) -> str:
        """Xử lý tiền tệ (hỗ trợ số nguyên, thập phân, và dấu chấm cuối câu)"""
        currency_map = {
            '$': 'dollars',
            '€': 'euros',
            '£': 'pounds',
            '¥': 'yen',
            '₩': 'won',
            '₽': 'rubles'
        }
    
        def currency_to_words(value, symbol):
            # Xử lý dấu chấm kết thúc câu (ví dụ: $20.)
            if value.endswith('.'):
                value = value[:-1]
                return f"{TextProcessor._number_to_words(value)} {currency_map.get(symbol, '')}."
    
            # Xử lý số thập phân (ví dụ: $20.5 → "twenty dollars and fifty cents")
            if '.' in value:
                integer_part, decimal_part = value.split('.')
                decimal_part = decimal_part.ljust(2, '0')  # Đảm bảo 2 chữ số
                return (
                    f"{TextProcessor._number_to_words(integer_part)} {currency_map.get(symbol, '')} "
                    f"and {TextProcessor._number_to_words(decimal_part)} cents"
                )
    
            # Số nguyên (ví dụ: $20 → "twenty dollars")
            return f"{TextProcessor._number_to_words(value)} {currency_map.get(symbol, '')}"
    
        # Regex bắt tiền tệ (số nguyên hoặc thập phân, không bắt dấu chấm cuối nếu không có số)
        text = re.sub(
            r'([$€£¥₩₽])(\d+(?:\.\d+)?)(?=\s|$|\.|,|;)',  # Chỉ khớp nếu sau số là ký tự kết thúc
            lambda m: currency_to_words(m.group(2), m.group(1)),
            text
        )
    
        return text

    @staticmethod
    def _process_percentages(text: str) -> str:
        """Xử lý phần trăm"""
        text = re.sub(
            r'(\d+\.?\d*)%',
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} percent",
            text
        )
        return text

    @staticmethod
    def _process_math_operations(text: str) -> str:
        """Xử lý các phép toán và khoảng số"""
        math_map = {
            '+': 'plus',
            '-': 'minus',  # Mặc định là "minus", sẽ xử lý riêng cho khoảng số
            '×': 'times',
            '*': 'times',
            '÷': 'divided by',
            '/': 'divided by',
            '=': 'equals',
            '>': 'is greater than',
            '<': 'is less than'
        }
    
        # Xử lý KHOẢNG SỐ (3-4 → "three to four") khi KHÔNG có dấu = hoặc phép toán sau -
        text = re.sub(
            r'(\d+)\s*-\s*(\d+)(?!\s*[=+×*÷/><])',  # Chỉ áp dụng khi KHÔNG có dấu =/+/*... sau -
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} to {TextProcessor._number_to_words(m.group(2))}",
            text
        )
    
        # Xử lý PHÉP TRỪ (chỉ khi có dấu = hoặc phép toán sau -)
        text = re.sub(
            r'(\d+)\s*-\s*(\d+)(?=\s*[=+×*÷/><])',  # Chỉ áp dụng khi CÓ dấu =/+/*... sau -
            lambda m: f"{TextProcessor._number_to_words(m.group(1))} minus {TextProcessor._number_to_words(m.group(2))}",
            text
        )
    
        # Xử lý các PHÉP TOÁN KHÁC (+, *, /, ...)
        text = re.sub(
            r'(\d+)\s*([+×*÷/=><])\s*(\d+)',
            lambda m: (f"{TextProcessor._number_to_words(m.group(1))} "
                      f"{math_map.get(m.group(2), m.group(2))} "
                      f"{TextProcessor._number_to_words(m.group(3))}"),
            text
        )
    
        # Xử lý phân số 4/5
        text = re.sub(
            r'(\d+)/(\d+)',
            lambda m: (f"{TextProcessor._number_to_words(m.group(1))} "
                      f"divided by {TextProcessor._number_to_words(m.group(2))}"),
            text
        )
    
        return text

    @staticmethod
    def _process_special_symbols(text: str) -> str:
        """Xử lý các ký hiệu đặc biệt"""
        symbol_map = {
            '@': 'at',
            '#': 'number',
            '&': 'and',
            '_': 'underscore'
        }

        # Xử lý @home → at home
        text = re.sub(
            r'@(\w+)',
            lambda m: f"at {m.group(1)}",
            text
        )

        # Xử lý #1 → number one
        text = re.sub(
            r'#(\d+)',
            lambda m: f"number {TextProcessor._number_to_words(m.group(1))}",
            text
        )

        # Xử lý các ký hiệu đơn lẻ
        for symbol, replacement in symbol_map.items():
            text = text.replace(symbol, f' {replacement} ')

        return text

    @staticmethod
    def _process_times(text: str) -> str:
        """Xử lý MỌI định dạng thời gian (giờ:phút:giây, có/không AM/PM)"""
        text = re.sub(
            r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\b',
            lambda m: TextProcessor._time_to_words(m.group(1), m.group(2), m.group(3), m.group(4)),
            text
        )
        return text
    
    @staticmethod
    def _time_to_words(hour: str, minute: str, second: str = None, period: str = None) -> str:
        """Chuyển thời gian thành giọng nói tự nhiên (bao gồm giây nếu có)"""
        hour_int = int(hour)
        minute_int = int(minute)
        
        # 1. Xử lý AM/PM (viết hoa chuẩn)
        period_text = f" {period.upper()}" if period else ""
        
        # 2. Chuyển đổi giờ 24h → 12h
        hour_12 = hour_int % 12
        hour_text = "twelve" if hour_12 == 0 else TextProcessor._number_to_words(str(hour_12))
        
        # 3. Xử lý phút
        minute_text = " \u200Bo'clock\u200B " if minute_int == 0 else \
                     f"oh {TextProcessor._number_to_words(minute)}" if minute_int < 10 else \
                     TextProcessor._number_to_words(minute)
        
        # 4. Xử lý giây (nếu có)
        second_text = ""
        if second and int(second) > 0:
            second_text = f" and {TextProcessor._number_to_words(second)} seconds"
        
        # 5. Ghép câu logic
        if minute_int == 0 and not second_text:
            return f"{hour_text}{minute_text}{period_text}"  # 3:00 → "three o'clock"
        else:
            return f"{hour_text} {minute_text}{second_text}{period_text}"  # 3:05:30 → "three oh five and thirty seconds"

    @staticmethod
    def _process_years(text: str) -> str:
        """Xử lý các năm trong văn bản - ƯU TIÊN XỬ LÝ TRƯỚC ĐƠN VỊ"""
        
        # ✅ Xử lý thập niên (1920s) TRƯỚC
        text = re.sub(
            r'\b(1[0-9]{2}0|2[0-9]{2}0)s\b',
            lambda m: TextProcessor._decade_to_words(m.group(1)),
            text
        )
        
        # ✅ Xử lý năm 4 chữ số (1920)
        text = re.sub(
            r'\b(1[0-9]{3}|2[0-9]{3})\b',
            lambda m: TextProcessor._year_to_words(m.group(1)),
            text
        )
        
        return text

    @staticmethod
    def _decade_to_words(year: str) -> str:
        """Chuyển thập niên thành chữ: 1920 → nineteen twenties"""
        if len(year) != 4:
            return year
        
        century = year[:2]
        decade = year[2:]
        
        century_words = TextProcessor._year_part_to_words(century)
        decade_words = TextProcessor._decade_part_to_words(decade)
        
        return f"{century_words} {decade_words}"

    @staticmethod
    def _year_to_words(year: str) -> str:
        """Chuyển năm 4 chữ số thành chữ"""
        if len(year) != 4:
            return year
        
        # Năm đặc biệt: 2000, 2001-2009
        if year == "2000":
            return "two thousand"
        elif year.startswith('200') and year[3] != '0':
            return f"two thousand {TextProcessor._digit_to_word(year[3])}"
        
        # Năm từ 2010 trở đi: twenty ten, twenty twenty-three
        if year.startswith('20'):
            return f"twenty {TextProcessor._two_digit_year_to_words(year[2:])}"
        
        # Các năm khác: nineteen twenty
        century = year[:2]
        decade = year[2:]
        
        century_words = TextProcessor._year_part_to_words(century)
        decade_words = TextProcessor._two_digit_year_to_words(decade)
        
        return f"{century_words} {decade_words}"

    @staticmethod
    def _year_part_to_words(part: str) -> str:
        """Chuyển phần năm thành chữ (19 → nineteen)"""
        numbers = {
            '19': 'nineteen', '20': 'twenty', '21': 'twenty-one',
            '18': 'eighteen', '17': 'seventeen', '16': 'sixteen',
            '15': 'fifteen', '14': 'fourteen', '13': 'thirteen'
        }
        return numbers.get(part, part)

    @staticmethod
    def _decade_part_to_words(decade: str) -> str:
        """Chuyển thập niên thành chữ: 20 → twenties"""
        decades = {
            '00': 'hundreds', '10': 'tens', '20': 'twenties',
            '30': 'thirties', '40': 'forties', '50': 'fifties',
            '60': 'sixties', '70': 'seventies', '80': 'eighties',
            '90': 'nineties'
        }
        return decades.get(decade, f"{decade}s")

    @staticmethod
    def _two_digit_year_to_words(num: str) -> str:
        """Chuyển số 2 chữ số thành chữ"""
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
        """Chuyển số (có thể có thập phân) thành chữ"""
        if '.' in number:
            integer, decimal = number.split('.')
            integer_words = TextProcessor._integer_to_words(integer)
            decimal_words = ' '.join(TextProcessor._digit_to_word(d) for d in decimal)
            return f"{integer_words} point {decimal_words}"
        else:
            return TextProcessor._integer_to_words(number)

    @staticmethod
    def _integer_to_words(number: str) -> str:
        """Chuyển số nguyên thành chữ"""
        num_int = int(number)
        if num_int < 1000:
            return TextProcessor._two_digit_year_to_words(number)
        
        # Xử lý số lớn hơn 1000
        thousands = num_int // 1000
        remainder = num_int % 1000
        
        if remainder == 0:
            return f"{TextProcessor._integer_to_words(str(thousands))} thousand"
        return f"{TextProcessor._integer_to_words(str(thousands))} thousand {TextProcessor._two_digit_year_to_words(str(remainder).zfill(3))}"

    @staticmethod
    def _digit_to_word(digit: str) -> str:
        """Chuyển chữ số thành chữ"""
        digits = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
            '8': 'eight', '9': 'nine'
        }
        return digits.get(digit, digit)

    def process(self, text: str) -> str:
        """Xử lý toàn bộ văn bản - QUAN TRỌNG: thứ tự xử lý"""
        # 1. Xử lý năm TRƯỚC (ưu tiên cao nhất)
        text = self._process_years(text)
        
        # 2. Xử lý đơn vị đo lường SAU
        text = self._process_measurements(text)
        
        return text

    @staticmethod
    def _process_phone_numbers(text: str) -> str:
        """Xử lý số điện thoại với regex chính xác hơn"""
        # Pattern mới tránh xung đột với số La Mã
        phone_pattern = r'\b(\d{3})[-. ]?(\d{3})[-. ]?(\d{4})\b'
    
        def phone_to_words(match):
            groups = match.groups()
            # Đọc từng số trong từng nhóm và thêm dấu phẩy (,) để tạo ngắt nghỉ
            parts = []
            for part in groups:
                digits = ' '.join([TextProcessor._digit_to_word(d) for d in part])
                parts.append(digits)
            return ', '.join(parts)  # Thêm dấu phẩy để tạo ngắt nghỉ khi đọc
    
        return re.sub(phone_pattern, phone_to_words, text)    
        @staticmethod
        def _process_currency_numbers(text: str) -> str:
            return re.sub(
                r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b',
                lambda m: f"{TextProcessor._number_to_words(m.group(1))} dollars" if '$' in m.group(0) 
                         else TextProcessor._number_to_words(m.group(1)),
                text
            )

    @staticmethod
    def _digit_to_word(digit: str) -> str:
        digit_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        return digit_map.get(digit, digit)

    @staticmethod
    def _number_to_words(number: str) -> str:
        num_str = number.replace(',', '')
    
        try:
            if '.' in num_str:
                integer_part, decimal_part = num_str.split('.')
                integer_text = TextProcessor._int_to_words(integer_part)
                decimal_text = ' '.join([TextProcessor._digit_to_word(d) for d in decimal_part])
                return f"{integer_text} point {decimal_text}"
            return TextProcessor._int_to_words(num_str)
        except:
            return number

    @staticmethod
    def _digits_to_words(digits: str) -> str:
        return ' '.join([TextProcessor._digit_to_word(d) for d in digits])

    @staticmethod
    def _int_to_words(num_str: str) -> str:
        num = int(num_str)
        if num == 0:
            return 'zero'
        
        units = ['', 'thousand', 'million', 'billion', 'trillion']
        words = []
        level = 0
        
        while num > 0:
            chunk = num % 1000
            if chunk != 0:
                words.append(TextProcessor._convert_less_than_thousand(chunk) + ' ' + units[level])
            num = num // 1000
            level += 1
        
        return ' '.join(reversed(words)).strip()

    @staticmethod
    def _convert_less_than_thousand(num: int) -> str:
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 
               'eighty', 'ninety']
        
        if num == 0:
            return ''
        if num < 20:
            return ones[num]
        if num < 100:
            return tens[num // 10] + (' ' + ones[num % 10] if num % 10 != 0 else '')
        return ones[num // 100] + ' hundred' + (' ' + TextProcessor._convert_less_than_thousand(num % 100) if num % 100 != 0 else '')

    def split_sentences(self, text: str, max_chars: int = 250) -> List[str]:
        """
        Chia văn bản thành các câu, giữ nguyên viết tắt như Mr., a.m., R., ...
        """
        if not text:
            return []
    
        # Danh sách viết tắt cần bảo vệ
        abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
            "a.m.", "p.m.", "etc.", "e.g.", "i.e.", "vs.", "No.", "Vol.", "Fig.", "Approx."
        ]
    
        protected = text
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<ABB>"))
    
        # Bảo vệ chữ cái viết tắt (R. , T. , J.)
        protected = re.sub(r"\b([A-Z])\.", r"\1<ABB>", protected)
    
        # Tách câu theo . ! ? (sau đó là khoảng trắng + chữ hoa)
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
        sentences = []
        for s in raw_sentences:
            s = s.replace("<ABB>", ".").strip()
            if not s:
                continue
    
            # Nếu câu quá dài → chia nhỏ
            if len(s) > max_chars:
                chunks = self.smart_text_split(s, max_chars)
                # DEBUG: In ra để kiểm tra
                print(f"Chia câu dài: '{s[:50]}...' thành {len(chunks)} đoạn")
                for i, chunk in enumerate(chunks):
                    print(f"  Đoạn {i+1}: '{chunk}' (dài {len(chunk)} ký tự)")
                sentences.extend(chunks)
            else:
                sentences.append(s)
    
        return sentences

    @staticmethod
    def parse_dialogues(text: str, prefixes: List[str]) -> List[Tuple[str, str]]:
        """Phân tích nội dung hội thoại với các prefix chỉ định"""
        dialogues = []
        current = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra xem dòng có bắt đầu bằng bất kỳ prefix nào không
            found_prefix = None
            for prefix in prefixes:
                if line.lower().startswith(prefix.lower() + ':'):
                    found_prefix = prefix
                    break
                    
            if found_prefix:
                if current:
                    # Xử lý các trường hợp đặc biệt trước khi thêm vào dialogues
                    processed_content = TextProcessor._process_special_cases(current[1])
                    dialogues.append((current[0], processed_content))
                
                speaker = found_prefix
                content = line[len(found_prefix)+1:].strip()
                current = (speaker, content)
            elif current:
                current = (current[0], current[1] + ' ' + line)
                
        if current:
            # Xử lý các trường hợp đặc biệt cho dòng cuối cùng
            processed_content = TextProcessor._process_special_cases(current[1])
            dialogues.append((current[0], processed_content))
            
        return dialogues

class AudioProcessor:
    @staticmethod
    def enhance_audio(audio: np.ndarray, volume: float = 1.0, pitch: float = 1.0) -> np.ndarray:
        # 1. Chuẩn hóa và bảo vệ chống clipping
        max_sample = np.max(np.abs(audio)) + 1e-8
        audio = (audio / max_sample) * 0.9 * volume  # Giữ headroom 10%
        
        # 2. Soft clipping để tránh distortion
        audio = np.tanh(audio * 1.5) / 1.5  # Hàm tanh cho soft clipping mượt
        
        # 3. Chuyển sang AudioSegment với xử lý pitch
        audio_seg = AudioSegment(
            (audio * 32767).astype(np.int16).tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1
        )
        
        # 4. Xử lý pitch với crossfade
        if pitch != 1.0:
            audio_seg = audio_seg._spawn(
                audio_seg.raw_data,
                overrides={"frame_rate": int(audio_seg.frame_rate * pitch)}
            ).set_frame_rate(24000).fade_in(10).fade_out(10)
        
        # 5. Xử lý động và lọc tần
        audio_seg = compress_dynamic_range(
            audio_seg,
            threshold=-12.0,
            ratio=3.5,
            attack=5,
            release=50
        )
        
        # 6. Chuẩn hóa an toàn
        if audio_seg.max_dBFS > -1.0:
            audio_seg = audio_seg.apply_gain(-audio_seg.max_dBFS * 0.8)
        
        return np.array(audio_seg.get_array_of_samples()) / 32768.0

    @staticmethod
    def calculate_pause(text: str, pause_settings: Dict[str, int]) -> int:
        """Calculate pause duration with more precise rules"""
        text = text.strip()
        if not text:
            return 0
            
        # Special cases that should have no pause
        if re.search(r'(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|St|A\.M|P\.M|etc|e\.g|i\.e)\.$', text, re.IGNORECASE):
            return 0
            
        # Time formats (12:30) - minimal pause
        if re.search(r'\b\d{1,2}:\d{2}\b', text):
            return pause_settings.get('time_colon_pause', 50)  # Default 50ms for times
            
        # Determine pause based on last character
        last_char = text[-1]
        return pause_settings.get(last_char, pause_settings['default_pause'])

    @staticmethod
    def combine_segments(segments: List[AudioSegment], pauses: List[int]) -> AudioSegment:
        """Combine audio segments with frame-accurate timing"""
        combined = AudioSegment.silent(duration=0)  # Start with 0 silence
        
        for i, (seg, pause) in enumerate(zip(segments, pauses)):
            # Apply fades without affecting duration
            seg = seg.fade_in(10).fade_out(10)
            
            # Add segment
            combined += seg
            
            # Add pause if not the last segment
            if i < len(segments) - 1:
                combined += AudioSegment.silent(duration=max(50, pause))
        
        return combined
        
    @staticmethod
    def combine_with_pauses(segments: List[AudioSegment], pauses: List[int]) -> AudioSegment:
        combined = AudioSegment.empty()
        for i, (seg, pause) in enumerate(zip(segments, pauses)):
            seg = seg.fade_in(50).fade_out(50)
            combined += seg
            if i < len(segments) - 1:
                combined += AudioSegment.silent(duration=pause)
        return combined

class SubtitleGenerator:
    @staticmethod
    def split_long_sentences(text: str, max_length: int = 150) -> List[str]:
        """
        Split long text into subtitle lines with max_length characters.
        Priority: 1. End punctuation (.!?) → 2. Comma (,) → 3. Space
        """
        text = text.strip()
        if len(text) <= max_length:
            return [text]
    
        sentences = []
        current_text = text
        
        while current_text and len(current_text) > max_length:
            split_pos = -1
            
            # Ưu tiên 1: Tìm dấu kết thúc câu (.!?)
            end_punct_match = re.search(r'[.!?][)\]\'"\s]*', current_text[:max_length])
            if end_punct_match:
                split_pos = end_punct_match.end()
            
            # Ưu tiên 2: Tìm dấu phẩy - CẢI THIỆN
            if split_pos == -1:
                # Tìm dấu phẩy CUỐI CÙNG trong phạm vi max_length
                comma_pos = current_text.rfind(',', 0, max_length)
                if comma_pos > 0:
                    # Kiểm tra xem có phải số không (1,000, 2,500)
                    if comma_pos > 0 and comma_pos < len(current_text) - 1:
                        next_char = current_text[comma_pos + 1]
                        if not next_char.isdigit():  # Không phải số → có thể chia
                            split_pos = comma_pos + 1
            
            # Ưu tiên 3: Tìm khoảng trắng
            if split_pos == -1:
                space_pos = current_text.rfind(' ', 0, max_length)
                if space_pos > 0:
                    split_pos = space_pos + 1
                else:
                    split_pos = max_length  # Buộc chia
            
            part = current_text[:split_pos].strip()
            if part:
                sentences.append(part)
            current_text = current_text[split_pos:].strip()
        
        if current_text:
            sentences.append(current_text)
        
        return sentences

    @staticmethod
    def clean_subtitle_text(text: str) -> str:
        """Remove speaker prefixes from subtitle text"""
        cleaned = re.sub(r'^(Q|A|CHAR\d+):\s*', '', text.strip())
        return cleaned

    @staticmethod
    def calculate_characters_per_ms(text: str, duration_ms: int) -> float:
        """Tính tốc độ đọc trung bình (ký tự/ms)"""
        if duration_ms <= 0:
            return 0
        # Loại bỏ khoảng trắng thừa để tính chính xác
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return len(clean_text) / duration_ms

    @staticmethod
    def generate_srt(audio_segments: List[AudioSegment], sentences: List[str], pause_settings: Dict[str, int]) -> str:
        """Generate SRT format subtitles with frame-accurate timing"""
        subtitles = []
        current_time_ms = 0  # Thời gian bắt đầu từ 0ms
        srt_index = 1

        for i, (audio_seg, original_sentence) in enumerate(zip(audio_segments, sentences)):
            # Làm sạch văn bản phụ đề
            cleaned_sentence = SubtitleGenerator.clean_subtitle_text(original_sentence)
            
            # Tính thời lượng chính xác của audio segment (ms)
            segment_duration_ms = len(audio_seg)
            
            # Tách câu thành các dòng phụ đề nếu cần
            text_chunks = SubtitleGenerator.split_long_sentences(cleaned_sentence, 150)
            
            if not text_chunks:
                continue

            # Tính thời lượng cho mỗi chunk dựa trên tỷ lệ ký tự
            total_chars = sum(len(re.sub(r'\s+', ' ', chunk).strip()) for chunk in text_chunks)
            
            for j, chunk in enumerate(text_chunks):
                # Tính số ký tự thực (không tính khoảng trắng thừa)
                clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
                chunk_chars = len(clean_chunk)
                
                if total_chars == 0:
                    chunk_duration_ms = segment_duration_ms / len(text_chunks)
                else:
                    # Tính thời lượng dựa trên tỷ lệ ký tự
                    chunk_duration_ms = int(segment_duration_ms * (chunk_chars / total_chars))
                
                # Đảm bảo thời lượng tối thiểu 500ms cho mỗi subtitle
                chunk_duration_ms = max(500, chunk_duration_ms)
                
                # Tính thời gian bắt đầu và kết thúc chính xác
                start_ms = current_time_ms
                end_ms = start_ms + chunk_duration_ms
                
                # Định dạng thời gian SRT chính xác đến mili giây
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
                
                # Thêm subtitle entry
                subtitles.append({
                    'index': srt_index,
                    'start_str': start_str,
                    'end_str': end_str,
                    'text': chunk.strip()
                })
                srt_index += 1
                
                # Cập nhật thời gian cho chunk tiếp theo
                current_time_ms += chunk_duration_ms
            
            # Thêm pause time giữa các segments
            if i < len(audio_segments) - 1:
                pause_ms = AudioProcessor.calculate_pause(original_sentence, pause_settings)
                current_time_ms += pause_ms

        # Tạo nội dung SRT
        srt_lines = []
        for sub in subtitles:
            srt_lines.append(f"{sub['index']}")
            srt_lines.append(f"{sub['start_str']} --> {sub['end_str']}")
            srt_lines.append(f"{sub['text']}")
            srt_lines.append("")  # Dòng trống giữa các entry

        return "\n".join(srt_lines)

    @staticmethod
    def generate_vtt(audio_segments: List[AudioSegment], sentences: List[str], pause_settings: Dict[str, int]) -> str:
        """Generate WebVTT format subtitles (alternative to SRT)"""
        srt_content = SubtitleGenerator.generate_srt(audio_segments, sentences, pause_settings)
        
        # Convert SRT to VTT
        vtt_lines = ["WEBVTT", ""]
        srt_lines = srt_content.split('\n')
        
        for line in srt_lines:
            if '-->' in line:
                # Replace comma with dot for VTT format
                line = line.replace(',', '.')
            vtt_lines.append(line)
        
        return "\n".join(vtt_lines)

class TTSGenerator:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.tokenizer = Tokenizer()
        self.subtitle_generator = SubtitleGenerator()

    def generate_sentence_audio(self, sentence: str, voice: str, speed: float,
                             device: str, volume: float = 1.0, pitch: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        try:
            # Check if voice exists
            if voice not in model_manager.voice_files:
                st.error(f"Voice {voice} not found in voices folder")
                return None
                
            # Load voice if not cached
            if voice not in model_manager.voice_cache:
                voice_path = model_manager.voice_files[voice]
                try:
                    # Load the voice model file
                    voice_data = torch.load(voice_path, map_location='cpu')
                    
                    # Assuming the voice file contains the necessary components for the pipeline
                    pipeline = model_manager.pipelines['a']  # or 'b' depending on your needs
                    pack = voice_data  # or extract the relevant parts from voice_data
                    
                    model_manager.voice_cache[voice] = (pipeline, pack)
                except Exception as e:
                    st.error(f"Error loading voice {voice}: {e}")
                    return None
            else:
                pipeline, pack = model_manager.voice_cache[voice]
            
            # Process text
            processed_text = self.tokenizer.process_text(sentence)
            
            # Generate audio
            for _, ps, _ in pipeline(processed_text, voice, speed):
                ref_s = pack[len(ps)-1]
                
                if device == 'cuda':
                    ps = ps.cuda()
                    ref_s = ref_s.cuda()
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    audio = model_manager.models[device](ps, ref_s, speed).cpu().numpy()
                
                return (24000, self.audio_processor.enhance_audio(audio, volume, pitch))
                
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            return None

    def generate_story_audio(self, text: str, voice: str, speed: float, device: str,
                           pause_settings: Dict[str, int], volume: float = 1.0, 
                           pitch: float = 1.0, max_chars_per_segment: int = 250) -> Tuple[Tuple[int, np.ndarray], str, str]:
        start_time = time.time()
        clean_text = self.text_processor.clean_text(text)
        
        # Sử dụng hàm chia đoạn mới với giới hạn ký tự
        sentences = self.text_processor.split_sentences(clean_text, max_chars_per_segment)
        
        if not sentences:
            return None, "No content to read", ""
        
        audio_segments = []
        pause_durations = []
        
        # Adjust pause settings based on speed (more precise calculation)
        speed_factor = max(0.5, min(2.0, speed))  # Clamp speed factor
        adjusted_pause_settings = {
            'default_pause': int(pause_settings['default_pause'] / speed_factor),
            'dot_pause': int(pause_settings['dot_pause'] / speed_factor),
            'ques_pause': int(pause_settings['ques_pause'] / speed_factor),
            'comma_pause': int(pause_settings['comma_pause'] / speed_factor),
            'colon_pause': int(pause_settings['colon_pause'] / speed_factor),
            'excl_pause': int(pause_settings['dot_pause'] / speed_factor),
            'semi_pause': int(pause_settings['colon_pause'] / speed_factor),
            'dash_pause': int(pause_settings['comma_pause'] / speed_factor),
            'time_colon_pause': 50  # Fixed short pause for time formats
        }
        
        # Generate each audio segment
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
            
            # Calculate precise pause duration
            pause = self.audio_processor.calculate_pause(sentence, adjusted_pause_settings)
            pause_durations.append(pause)
        
        if not audio_segments:
            return None, "Failed to generate audio", ""
        
        # Combine with frame-accurate timing
        combined_audio = self.audio_processor.combine_segments(audio_segments, pause_durations)
        
        # Export with precise timing
        with io.BytesIO() as buffer:
            combined_audio.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", str(combined_audio.frame_rate)])
            buffer.seek(0)
            audio_data = np.frombuffer(buffer.read(), dtype=np.uint8)
        
        # Generate subtitles with the same timing used for audio
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
        """
        Generate Q&A audio with perfect sync between audio and subtitles
        Args:
            text: Input text in Q&A format (Q: question\nA: answer)
            voice_q: Voice for questions
            voice_a: Voice for answers
            speed_q: Speed for questions (0.7-1.3)
            speed_a: Speed for answers (0.7-1.3)
            device: 'cuda' or 'cpu'
            repeat_times: Number of times to repeat each Q&A pair
            pause_q: Pause after questions (ms)
            pause_a: Pause after answers (ms)
            volume_q: Volume for questions (0.5-2.0)
            volume_a: Volume for answers (0.5-2.0)
            pitch_q: Pitch for questions (0.8-1.2)
            pitch_a: Pitch for answers (0.8-1.2)
        Returns:
            tuple: (sample_rate, audio_data), stats, subtitles
        """
        start_time = time.time()
        dialogues = self.text_processor.parse_dialogues(text, ['Q', 'A'])
        
        if not dialogues:
            return None, "No Q/A content found", ""
        
        combined = AudioSegment.empty()
        timing_info = []  # For precise subtitle timing
        current_pos = 0   # Current position in milliseconds
        
        # Group into Q&A pairs
        qa_pairs = []
        current_q = None
        for speaker, content in dialogues:
            if speaker.upper() == 'Q':
                current_q = (content, [])
            elif speaker.upper() == 'A' and current_q:
                current_q[1].append(content)
                qa_pairs.append(current_q)
                current_q = None
        
        # Process each Q&A pair
        for q_text, a_texts in qa_pairs:
            # Generate question audio
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
            
            # Generate answer audio (take first answer if multiple)
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
            
            # Repeat as specified
            for i in range(repeat_times):
                # Question timing
                q_start = current_pos
                q_end = q_start + len(q_seg)
                combined += q_seg
                timing_info.append({
                    'start': q_start,
                    'end': q_end,
                    'text': q_text  # No Q: prefix
                })
                
                # Add question pause
                current_pos = q_end + pause_q
                combined += AudioSegment.silent(duration=pause_q)
                
                # Answer timing
                a_start = current_pos
                a_end = a_start + len(a_seg)
                combined += a_seg
                timing_info.append({
                    'start': a_start,
                    'end': a_end,
                    'text': a_text  # No A: prefix
                })
                
                # Add answer pause (except last iteration)
                if i < repeat_times - 1:
                    combined += AudioSegment.silent(duration=pause_a)
                    current_pos = a_end + pause_a
                else:
                    current_pos = a_end
        
        if len(combined) == 0:
            return None, "Failed to generate audio", ""
        
        # Export to MP3 with precise timing
        with io.BytesIO() as buffer:
            combined.export(buffer, format="mp3", bitrate="256k", parameters=["-ar", "24000"])
            audio_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        
        # Generate clean subtitles
        subtitles = []
        for idx, info in enumerate(timing_info, 1):
            # Format time (HH:MM:SS,mmm)
            start_str = f"{info['start']//3600000:02d}:{(info['start']%3600000)//60000:02d}:{(info['start']%60000)//1000:02d},{info['start']%1000:03d}"
            end_str = f"{info['end']//3600000:02d}:{(info['end']%3600000)//60000:02d}:{(info['end']%60000)//1000:02d},{info['end']%1000:03d}"
            
            subtitles.append(
                f"{idx}\n"
                f"{start_str} --> {end_str}\n"
                f"{info['text']}\n"  # Already cleaned
            )
        
        stats = (f"Generated {len(qa_pairs)} Q/A pairs | "
                f"Repeated {repeat_times}x | "
                f"Duration: {len(combined)/1000:.2f}s | "
                f"Q: {speed_q:.1f}x | A: {speed_a:.1f}x | "
                f"Processing time: {time.time()-start_time:.2f}s")
        
        return (24000, audio_data), stats, "\n".join(subtitles)

    def generate_multi_char_audio(
        self,
        text: str,
        voices: Dict[str, str],
        speeds: Dict[str, float],
        volumes: Dict[str, float],
        pitches: Dict[str, float],
        device: str,
        pause_settings: Dict[str, int],
        mix_configs: Optional[Dict[str, Dict]] = None
    ) -> Tuple[Tuple[int, np.ndarray], str, str]:
        """
        Generate multi-character dialogue audio with perfect sync
        """
        start_time = time.time()
        dialogues = self.text_processor.parse_dialogues(text, list(voices.keys()))
        
        if not dialogues:
            return None, "No character dialogues found", ""
        
        combined = AudioSegment.empty()
        timing_info = []
        current_pos = 0
        char_stats = {char: {'lines': 0, 'duration': 0} for char in voices.keys()}
        
        # Process each dialogue line
        for speaker, content in dialogues:
            if speaker not in voices:
                continue
                
            voice = voices[speaker]
            speed = speeds.get(speaker, 1.0)
            volume = volumes.get(speaker, 1.0)
            pitch = pitches.get(speaker, 1.0)
            
            # Kiểm tra nếu character này sử dụng mixed voice
            if mix_configs and speaker in mix_configs and mix_configs[speaker]:
                mix_config = mix_configs[speaker]
                mix_voices = mix_config.get('voices', [])
                mix_weights = mix_config.get('weights', [])
                
                if len(mix_voices) >= 2:
                    # Generate audio với mixed voice
                    result = self.generate_mixed_voice_audio(
                        content, mix_voices, mix_weights, speed, device, 
                        pause_settings, volume, pitch
                    )
                    if result:
                        # Kiểm tra kiểu trả về và xử lý đúng
                        if isinstance(result, tuple) and len(result) == 3:
                            # Nếu trả về 3 phần tử: (audio_data, stats, subtitles)
                            audio_result, _, _ = result
                            sample_rate, audio_data = audio_result
                        else:
                            # Nếu trả về 2 phần tử: (sample_rate, audio_data)
                            sample_rate, audio_data = result
                    else:
                        continue
                else:
                    # Fallback to single voice
                    result = self.generate_sentence_audio(content, voice, speed, device, volume, pitch)
                    if not result:
                        continue
                    sample_rate, audio_data = result
            else:
                # Use single voice
                result = self.generate_sentence_audio(content, voice, speed, device, volume, pitch)
                if not result:
                    continue
                sample_rate, audio_data = result
            
            # KIỂM TRA VÀ CHUẨN HÓA AUDIO DATA - SỬA LỖI QUAN TRỌNG
            if audio_data is None or len(audio_data) == 0:
                continue
                
            # Đảm bảo audio_data là numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Đảm bảo audio_data ở dạng float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Kiểm tra nếu audio_data có giá trị NaN hoặc infinity
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                # Nếu có lỗi, tạo audio silent thay thế
                duration_ms = 1000  # 1 giây silent
                silent_samples = int(sample_rate * duration_ms / 1000)
                audio_data = np.zeros(silent_samples, dtype=np.float32)
            
            # Normalize audio data để tránh clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Giảm 5% để tránh clipping
            else:
                # Nếu toàn bộ là 0, tạo audio silent
                duration_ms = 1000  # 1 giây silent
                silent_samples = int(sample_rate * duration_ms / 1000)
                audio_data = np.zeros(silent_samples, dtype=np.float32)
            
            # Chuyển đổi sang int16 cho AudioSegment
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Tạo audio segment
            audio_seg = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            ).fade_in(10).fade_out(10)
            
            # Apply volume adjustment
            if volume != 1.0:
                gain_db = 20 * np.log10(volume)  # Convert volume multiplier to dB
                audio_seg = audio_seg.apply_gain(gain_db)
            
            # Calculate timing
            seg_start = current_pos
            seg_end = seg_start + len(audio_seg)
            
            # Add to combined audio
            combined += audio_seg
            char_stats[speaker]['lines'] += 1
            char_stats[speaker]['duration'] += len(audio_seg)
            
            # Add to timing info
            timing_info.append({
                'start': seg_start,
                'end': seg_end,
                'text': content.strip()
            })
            
            # Calculate and add pause
            current_pos = seg_end
            pause = AudioProcessor.calculate_pause(content, pause_settings)
            if pause > 0:
                combined += AudioSegment.silent(duration=pause)
                current_pos += pause
        
        if len(combined) == 0:
            return None, "Failed to generate audio", ""
        
        # Export to MP3 - SỬA LỖI XUẤT FILE
        try:
            with io.BytesIO() as buffer:
                combined.export(
                    buffer, 
                    format="mp3", 
                    bitrate="256k", 
                    parameters=["-ar", "24000"]
                )
                buffer.seek(0)
                audio_data_mp3 = buffer.getvalue()
            
            # Convert to numpy array
            audio_data_np = np.frombuffer(audio_data_mp3, dtype=np.uint8)
            
        except Exception as e:
            st.error(f"Error exporting audio: {e}")
            return None, f"Audio export failed: {e}", ""
        
        # Generate clean subtitles
        subtitles = []
        for idx, info in enumerate(timing_info, 1):
            # Format time (HH:MM:SS,mmm)
            start_str = f"{info['start']//3600000:02d}:{(info['start']%3600000)//60000:02d}:{(info['start']%60000)//1000:02d},{info['start']%1000:03d}"
            end_str = f"{info['end']//3600000:02d}:{(info['end']%3600000)//60000:02d}:{(info['end']%60000)//1000:02d},{info['end']%1000:03d}"
            
            subtitles.append(
                f"{idx}\n"
                f"{start_str} --> {end_str}\n"
                f"{info['text']}\n"
            )
        
        # Generate stats
        stats_lines = [
            f"Multi-character dialogue ({len(combined)/1000:.2f}s)",
            *[f"{char}: {stats['lines']} lines ({voices[char]}, {speeds.get(char, 1.0):.1f}x speed, "
              f"{volumes.get(char, 1.0):.1f}x vol, {pitches.get(char, 1.0):.1f}x pitch) | "
              f"Duration: {stats['duration']/1000:.2f}s"
              for char, stats in char_stats.items()],
            f"Processing time: {time.time()-start_time:.2f}s"
        ]
        
        return (24000, audio_data_np), "\n".join(stats_lines), "\n".join(subtitles)

    def generate_mixed_voice_audio(self, text: str, voices: List[str], weights: List[float], speed: float, 
                                 device: str, pause_settings: Dict[str, int],
                                 volume: float = 1.0, pitch: float = 1.0) -> Tuple[Tuple[int, np.ndarray], str, str]:
        """
        Generate audio with blended voice embeddings
        """
        start_time = time.time()
        clean_text = self.text_processor.clean_text(text)
        sentences = self.text_processor.split_sentences(clean_text)
        
        if not sentences:
            return None, "No content to read", ""
        
        if len(voices) != len(weights):
            return None, "Number of voices must match number of weights", ""
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return None, "Weights cannot all be zero", ""
        normalized_weights = [w / total_weight for w in weights]
    
        # Load all voice embeddings
        blended_embedding = None
        for voice, weight in zip(voices, normalized_weights):
            if voice not in model_manager.voice_cache:
                if voice not in model_manager.voice_files:
                    return None, f"Voice {voice} not found", ""
                
                voice_path = model_manager.voice_files[voice]
                try:
                    voice_data = torch.load(voice_path, map_location="cpu")
                    pipeline = model_manager.pipelines['a']  # chọn pipeline mặc định
                    model_manager.voice_cache[voice] = (pipeline, voice_data)
                except Exception as e:
                    return None, f"Error loading voice {voice}: {e}", ""
            
            _, pack = model_manager.voice_cache[voice]
    
            # Nếu blended_embedding chưa có thì khởi tạo
            if blended_embedding is None:
                blended_embedding = pack * weight
            else:
                blended_embedding += pack * weight
    
        if blended_embedding is None:
            return None, "No embeddings created", ""
    
        audio_segments, pause_durations = [], []
    
        # Generate audio for each sentence
        for sentence in sentences:
            try:
                processed_text = self.tokenizer.process_text(sentence)
                
                for _, ps, _ in model_manager.pipelines['a'](processed_text, "", speed):
                    ref_s = blended_embedding[len(ps)-1]
    
                    if device == 'cuda':
                        ps = ps.cuda()
                        ref_s = ref_s.cuda()
    
                    with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                        audio = model_manager.models[device](ps, ref_s, speed).cpu().numpy()
    
                    enhanced_audio = self.audio_processor.enhance_audio(audio, volume, pitch)
                    audio_seg = AudioSegment(
                        (enhanced_audio * 32767).astype(np.int16).tobytes(),
                        frame_rate=24000,
                        sample_width=2,
                        channels=1
                    ).fade_in(10).fade_out(10)
    
                    audio_segments.append(audio_seg)
                    pause_durations.append(self.audio_processor.calculate_pause(sentence, pause_settings))
                    break
    
            except Exception as e:
                st.error(f"Error generating blended audio: {e}")
                continue
    
        if not audio_segments:
            return None, "Failed to generate any audio", ""
    
        # Combine
        combined_audio = self.audio_processor.combine_segments(audio_segments, pause_durations)
    
        # Export
        with io.BytesIO() as buffer:
            combined_audio.export(buffer, format="mp3", bitrate="256k",
                                  parameters=["-ar", str(combined_audio.frame_rate)])
            buffer.seek(0)
            audio_data = np.frombuffer(buffer.read(), dtype=np.uint8)
    
        subtitles = self.subtitle_generator.generate_srt(audio_segments, sentences, pause_settings)
    
        mix_desc = " + ".join([f"{w*100:.0f}% {v}" for v, w in zip(voices, normalized_weights)])
        stats = (f"Blended Voice: {mix_desc}\n"
                 f"Processed {len(clean_text)} chars, {len(clean_text.split())} words\n"
                 f"Audio duration: {len(combined_audio)/1000:.2f}s\n"
                 f"Time: {time.time() - start_time:.2f}s\n"
                 f"Device: {device.upper()}")
    
        return (combined_audio.frame_rate, audio_data), stats, subtitles

    def generate_mixed_voice_audio_qa(self, text: str, voices_q: List[str], weights_q: List[float], 
                                    voices_a: List[str], weights_a: List[float], 
                                    speed_q: float, speed_a: float, device: str, 
                                    repeat_times: int, pause_q: int, pause_a: int,
                                    volume_q: float = 1.0, volume_a: float = 1.0,
                                    pitch_q: float = 1.0, pitch_a: float = 1.0) -> Tuple[Tuple[int, np.ndarray], str, str]:
        """
        Generate Q&A audio with blended voice embeddings for both Q and A separately
        """
        # Implementation for mixed voice Q&A
        pass
    
    def generate_mixed_voice_audio_multi_char(self, text: str, voices: List[str], weights: List[float], 
                                            device: str, pause_settings: Dict[str, int],
                                            char_settings: Dict[str, Dict[str, float]]) -> Tuple[Tuple[int, np.ndarray], str, str]:
        """
        Generate multi-character audio with blended voice embeddings
        """
        # Implementation for mixed voice multi-character
        pass

def create_streamlit_app():
    st.set_page_config(
        page_title="Advanced Multi-Character TTS",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Advanced TTS with Multi-Character Dialogue & Subtitles")
    
    # Initialize session state for mixed voice settings
    if 'mixed_voice_checkbox' not in st.session_state:
        st.session_state.mixed_voice_checkbox = False
    if 'mixed_voice1' not in st.session_state:
        st.session_state.mixed_voice1 = None
    if 'mixed_weight1' not in st.session_state:
        st.session_state.mixed_weight1 = 50
    if 'mixed_voice2' not in st.session_state:
        st.session_state.mixed_voice2 = None
    if 'mixed_weight2' not in st.session_state:
        st.session_state.mixed_weight2 = 50
    if 'mixed_voice3' not in st.session_state:
        st.session_state.mixed_voice3 = None
    if 'mixed_weight3' not in st.session_state:
        st.session_state.mixed_weight3 = 50
    
    # Get voice list
    voice_list = model_manager.get_voice_list()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Standard Mode", "Q&A Mode", "Multi-Character Mode (4 Characters)"])
    
    # Initialize generator
    generator = TTSGenerator()
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Settings")
            
            # Mixed Voice Settings
            with st.expander("🎭 Mixed Voice Settings (Optional)"):
                mixed_enabled = st.checkbox(
                    "Enable Mixed Voice",
                    value=st.session_state.mixed_voice_checkbox,
                    help="Combine multiple voices with custom weights"
                )
                
                if mixed_enabled:
                    col_mix1, col_mix2, col_mix3 = st.columns(3)
                    
                    with col_mix1:
                        mixed_voice1 = st.selectbox(
                            "Voice 1",
                            options=voice_list,
                            index=0 if not voice_list else voice_list.index(voice_list[0]) if voice_list[0] in voice_list else 0
                        )
                        mixed_weight1 = st.slider("Weight 1 (%)", 0, 100, 50, key="mix1_weight")
                    
                    with col_mix2:
                        mixed_voice2 = st.selectbox(
                            "Voice 2",
                            options=voice_list,
                            index=1 if len(voice_list) > 1 else 0
                        )
                        mixed_weight2 = st.slider("Weight 2 (%)", 0, 100, 50, key="mix2_weight")
                    
                    with col_mix3:
                        mixed_voice3 = st.selectbox(
                            "Voice 3",
                            options=voice_list,
                            index=2 if len(voice_list) > 2 else 0
                        )
                        mixed_weight3 = st.slider("Weight 3 (%)", 0, 100, 50, key="mix3_weight")
                    
                    # Calculate and display mix summary
                    if mixed_weight1 + mixed_weight2 + mixed_weight3 > 0:
                        total = mixed_weight1 + mixed_weight2 + mixed_weight3
                        mix_summary = f"Mix: {mixed_weight1/total*100:.0f}% {mixed_voice1}"
                        if mixed_weight2 > 0:
                            mix_summary += f" + {mixed_weight2/total*100:.0f}% {mixed_voice2}"
                        if mixed_weight3 > 0:
                            mix_summary += f" + {mixed_weight3/total*100:.0f}% {mixed_voice3}"
                        st.info(mix_summary)
                else:
                    mixed_voice1 = None
                    mixed_weight1 = 0
                    mixed_voice2 = None
                    mixed_weight2 = 0
                    mixed_voice3 = None
                    mixed_weight3 = 0
            
            # Text input
            text_input = st.text_area(
                "Input Text",
                value="Contact us at info@example.com or call 012-345-6789. Our website is https://www.example.com",
                height=150
            )
            
            # Voice settings
            with st.expander("Voice Settings", expanded=True):
                if not mixed_enabled:
                    voice = st.selectbox(
                        "Select Voice",
                        options=voice_list,
                        index=0
                    )
                else:
                    voice = voice_list[0] if voice_list else None
                
                col_speed, col_volume, col_pitch = st.columns(3)
                with col_speed:
                    speed = st.slider("Speed", 0.7, 1.3, 0.8, 0.05)
                with col_volume:
                    volume = st.slider("Volume", 0.5, 2.0, 1.0, 0.1)
                with col_pitch:
                    pitch = st.slider("Pitch", 0.8, 1.2, 1.0, 0.05)
                
                max_chars = st.slider(
                    "Max Characters per Segment",
                    100, 500, 250, 50,
                    help="Chia văn bản thành các đoạn ngắn hơn nếu cần"
                )
                
                device = st.radio(
                    "Processing Device",
                    ["GPU 🚀" if CUDA_AVAILABLE else "GPU (Not Available)", "CPU"],
                    index=0 if CUDA_AVAILABLE else 1
                )
            
            # Pause settings
            with st.expander("Pause Settings (ms)"):
                col_pause1, col_pause2 = st.columns(2)
                with col_pause1:
                    default_pause = st.slider("Default", 0, 2000, 200)
                    dot_pause = st.slider("Period (.)", 0, 3000, 600)
                    ques_pause = st.slider("Question (?)", 0, 3000, 800)
                with col_pause2:
                    comma_pause = st.slider("Comma (,)", 0, 1500, 300)
                    colon_pause = st.slider("Colon (:)", 0, 2000, 400)
            
            generate_btn = st.button("Generate Speech", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Output")
            
            if generate_btn and text_input:
                with st.spinner("Generating audio..."):
                    device_str = "cuda" if "GPU" in device and CUDA_AVAILABLE else "cpu"
                    
                    pause_settings = {
                        'default_pause': default_pause,
                        'dot_pause': dot_pause,
                        'ques_pause': ques_pause,
                        'comma_pause': comma_pause,
                        'colon_pause': colon_pause,
                        'excl_pause': dot_pause,
                        'semi_pause': colon_pause,
                        'dash_pause': comma_pause,
                        'time_colon_pause': 0
                    }
                    
                    # XỬ LÝ MIX VOICE VỚI WEIGHT
                    if mixed_enabled:
                        voices = []
                        weights = []
                        
                        # Collect voices and weights
                        if mixed_voice1 and mixed_weight1 > 0:
                            voices.append(mixed_voice1)
                            weights.append(mixed_weight1)
                        if mixed_voice2 and mixed_weight2 > 0:
                            voices.append(mixed_voice2)
                            weights.append(mixed_weight2)
                        if mixed_voice3 and mixed_weight3 > 0:
                            voices.append(mixed_voice3)
                            weights.append(mixed_weight3)
                        
                        if len(voices) >= 2:
                            result, stats, subtitles = generator.generate_mixed_voice_audio(
                                text_input, voices, weights, speed, device_str, 
                                pause_settings, volume, pitch
                            )
                        else:
                            # Fallback to single voice if not enough voices selected
                            result, stats, subtitles = generator.generate_story_audio(
                                text_input, voice, speed, device_str, 
                                pause_settings, volume, pitch, max_chars
                            )
                    else:
                        # Use single voice
                        result, stats, subtitles = generator.generate_story_audio(
                            text_input, voice, speed, device_str, 
                            pause_settings, volume, pitch, max_chars
                        )
                    
                    if result:
                        sample_rate, audio_data = result
                        
                        # Save to file
                        output_dir = "out"
                        os.makedirs(output_dir, exist_ok=True)
                        filepath = os.path.join(output_dir, "output.mp3")
                        with open(filepath, "wb") as f:
                            f.write(audio_data.tobytes())
                        
                        # Save SRT file
                        srt_path = os.path.join(output_dir, "subtitles.srt")
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(subtitles)
                        
                        # Display audio
                        st.audio(filepath, format="audio/mp3")
                        
                        # Display stats
                        st.text_area("Processing Stats", stats, height=100)
                        
                        # Display subtitles
                        with st.expander("Subtitles (.srt format)"):
                            st.text(subtitles)
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            with open(filepath, "rb") as f:
                                st.download_button(
                                    "Download Audio",
                                    f,
                                    file_name="output.mp3",
                                    mime="audio/mp3"
                                )
                        with col_dl2:
                            with open(srt_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "Download SRT",
                                    f,
                                    file_name="subtitles.srt",
                                    mime="text/plain"
                                )
                    else:
                        st.error("Failed to generate audio")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Q&A Settings")
            
            qa_input = st.text_area(
                "Q&A Content (Format: Q: question / A: answer)",
                value="Q: Contact us at info@example.com\nA: Call us at 012-345-6789 or visit https://www.example.com",
                height=150
            )
            
            # Mixed Voice for Q&A
            col_q, col_a = st.columns(2)
            with col_q:
                with st.expander("Mixed Voice for Q", expanded=False):
                    mixed_enabled_q = st.checkbox("Enable Mix for Q", value=False, key="mix_q_enable")
                    if mixed_enabled_q:
                        voice1_q = st.selectbox("Voice 1 Q", voice_list, key="voice1_q")
                        weight1_q = st.slider("Weight 1 Q (%)", 0, 100, 80, key="weight1_q")
                        voice2_q = st.selectbox("Voice 2 Q", voice_list, key="voice2_q")
                        weight2_q = st.slider("Weight 2 Q (%)", 0, 100, 80, key="weight2_q")
                        voice3_q = st.selectbox("Voice 3 Q", voice_list, key="voice3_q")
                        weight3_q = st.slider("Weight 3 Q (%)", 0, 100, 0, key="weight3_q")
                if not mixed_enabled_q:
                    voice_q = st.selectbox("Question Voice", voice_list, key="voice_q")
            
            with col_a:
                with st.expander("Mixed Voice for A", expanded=False):
                    mixed_enabled_a = st.checkbox("Enable Mix for A", value=False, key="mix_a_enable")
                    if mixed_enabled_a:
                        voice1_a = st.selectbox("Voice 1 A", voice_list, key="voice1_a")
                        weight1_a = st.slider("Weight 1 A (%)", 0, 100, 80, key="weight1_a")
                        voice2_a = st.selectbox("Voice 2 A", voice_list, key="voice2_a")
                        weight2_a = st.slider("Weight 2 A (%)", 0, 100, 80, key="weight2_a")
                        voice3_a = st.selectbox("Voice 3 A", voice_list, key="voice3_a")
                        weight3_a = st.slider("Weight 3 A (%)", 0, 100, 0, key="weight3_a")
                if not mixed_enabled_a:
                    voice_a = st.selectbox("Answer Voice", voice_list, key="voice_a")
            
            # Speed, volume, pitch for Q&A
            col_q_settings, col_a_settings = st.columns(2)
            with col_q_settings:
                st.markdown("**Question Settings**")
                speed_q = st.slider("Q Speed", 0.7, 1.3, 0.8, 0.05, key="speed_q")
                volume_q = st.slider("Q Volume", 0.5, 2.0, 1.0, 0.1, key="volume_q")
                pitch_q = st.slider("Q Pitch", 0.8, 1.2, 1.0, 0.05, key="pitch_q")
            
            with col_a_settings:
                st.markdown("**Answer Settings**")
                speed_a = st.slider("A Speed", 0.7, 1.3, 0.8, 0.05, key="speed_a")
                volume_a = st.slider("A Volume", 0.5, 2.0, 1.0, 0.1, key="volume_a")
                pitch_a = st.slider("A Pitch", 0.8, 1.2, 1.0, 0.05, key="pitch_a")
            
            # Device
            device_qa = st.radio(
                "Processing Device",
                ["GPU 🚀" if CUDA_AVAILABLE else "GPU (Not Available)", "CPU"],
                index=0 if CUDA_AVAILABLE else 1,
                key="device_qa"
            )
            
            # Repetition & pause settings
            col_rep, col_pause = st.columns(2)
            with col_rep:
                repeat_times = st.slider("Repeat Times per Q&A Pair", 1, 5, 1, 1)
            with col_pause:
                pause_q = st.slider("Pause after Q (ms)", 100, 1000, 300, 50)
                pause_a = st.slider("Pause after A (ms)", 100, 1500, 500, 50)
            
            generate_qa_btn = st.button("Generate Q&A Audio", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Q&A Output")
            
            if generate_qa_btn and qa_input:
                with st.spinner("Generating Q&A audio..."):
                    device_str = "cuda" if "GPU" in device_qa and CUDA_AVAILABLE else "cpu"
                    
                    # Call the Q&A generation function
                    result, stats, subtitles = generator.generate_qa_audio(
                        qa_input, 
                        voice_q if not mixed_enabled_q else voice_list[0],
                        voice_a if not mixed_enabled_a else voice_list[0],
                        speed_q, speed_a, volume_q, pitch_q, volume_a, pitch_a,
                        device_str, repeat_times, pause_q, pause_a
                    )
                    
                    if result:
                        sample_rate, audio_data = result
                        
                        # Save to file
                        output_dir = "out_qa"
                        os.makedirs(output_dir, exist_ok=True)
                        filepath = os.path.join(output_dir, "output_qa.mp3")
                        with open(filepath, "wb") as f:
                            f.write(audio_data.tobytes())
                        
                        # Save SRT file
                        srt_path = os.path.join(output_dir, "subtitles_qa.srt")
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(subtitles)
                        
                        # Display audio
                        st.audio(filepath, format="audio/mp3")
                        
                        # Display stats
                        st.text_area("Processing Stats", stats, height=100, key="qa_stats")
                        
                        # Display subtitles
                        with st.expander("Subtitles (.srt format)"):
                            st.text(subtitles)
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            with open(filepath, "rb") as f:
                                st.download_button(
                                    "Download Audio",
                                    f,
                                    file_name="output_qa.mp3",
                                    mime="audio/mp3",
                                    key="dl_qa_audio"
                                )
                        with col_dl2:
                            with open(srt_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "Download SRT",
                                    f,
                                    file_name="subtitles_qa.srt",
                                    mime="text/plain",
                                    key="dl_qa_srt"
                                )
                    else:
                        st.error("Failed to generate Q&A audio")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Multi-Character Settings")
            
            char_input = st.text_area(
                "Multi-Character Dialogue",
                value="""CHAR1: Hello everyone, my email is info@example.com
CHAR2: Hi there! You can call me at 012-345-6789
CHAR3: Our website is https://www.example.com
CHAR4: The price is $1,234.56""",
                height=150
            )
            
            # Character settings
            st.markdown("### Character Settings")
            
            # Create 4 columns for characters
            chars = []
            for i in range(1, 5):
                with st.expander(f"Character {i} Settings", expanded=True if i == 1 else False):
                    col_name, col_voice = st.columns(2)
                    with col_name:
                        char_name = st.text_input(f"Name CHAR{i}", value=f"CHAR{i}", key=f"char{i}_name")
                    with col_voice:
                        char_voice = st.selectbox(f"Voice CHAR{i}", voice_list, key=f"char{i}_voice")
                    
                    col_speed, col_vol, col_pitch = st.columns(3)
                    with col_speed:
                        char_speed = st.slider(f"Speed CHAR{i}", 0.7, 1.3, 0.9, 0.05, key=f"char{i}_speed")
                    with col_vol:
                        char_volume = st.slider(f"Volume CHAR{i}", 0.5, 2.0, 1.0, 0.1, key=f"char{i}_vol")
                    with col_pitch:
                        char_pitch = st.slider(f"Pitch CHAR{i}", 0.8, 1.2, 1.0, 0.05, key=f"char{i}_pitch")
                    
                    chars.append({
                        'name': char_name,
                        'voice': char_voice,
                        'speed': char_speed,
                        'volume': char_volume,
                        'pitch': char_pitch
                    })
            
            # Device & pause settings
            with st.expander("Device & Pause Settings"):
                device_char = st.radio(
                    "Processing Device",
                    ["GPU 🚀" if CUDA_AVAILABLE else "GPU (Not Available)", "CPU"],
                    index=0 if CUDA_AVAILABLE else 1,
                    key="device_char"
                )
                
                default_pause_char = st.slider("Default Pause (ms)", 0, 2000, 300, key="default_pause_char")
                dot_pause_char = st.slider("After Period (.)", 0, 3000, 600, key="dot_pause_char")
                ques_pause_char = st.slider("After Question (?)", 0, 3000, 800, key="ques_pause_char")
            
            generate_char_btn = st.button("Generate Multi-Character Audio", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Multi-Character Output")
            
            if generate_char_btn and char_input:
                with st.spinner("Generating multi-character audio..."):
                    device_str = "cuda" if "GPU" in device_char and CUDA_AVAILABLE else "cpu"
                    
                    # Prepare character settings
                    voices = {}
                    speeds = {}
                    volumes = {}
                    pitches = {}
                    
                    for char in chars:
                        if char['name'].strip():
                            voices[char['name'].strip()] = char['voice']
                            speeds[char['name'].strip()] = char['speed']
                            volumes[char['name'].strip()] = char['volume']
                            pitches[char['name'].strip()] = char['pitch']
                    
                    pause_settings = {
                        'default_pause': default_pause_char,
                        'dot_pause': dot_pause_char,
                        'ques_pause': ques_pause_char,
                        'comma_pause': default_pause_char // 2,
                        'colon_pause': default_pause_char,
                        'excl_pause': dot_pause_char,
                        'semi_pause': default_pause_char,
                        'dash_pause': default_pause_char // 2,
                        'time_colon_pause': 0
                    }
                    
                    # Generate audio
                    result, stats, subtitles = generator.generate_multi_char_audio(
                        char_input, voices, speeds, volumes, pitches, 
                        device_str, pause_settings
                    )
                    
                    if result:
                        sample_rate, audio_data = result
                        
                        # Save to file
                        output_dir = "out_multi"
                        os.makedirs(output_dir, exist_ok=True)
                        filepath = os.path.join(output_dir, "output_multi.mp3")
                        with open(filepath, "wb") as f:
                            f.write(audio_data.tobytes())
                        
                        # Save SRT file
                        srt_path = os.path.join(output_dir, "subtitles_multi.srt")
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(subtitles)
                        
                        # Display audio
                        st.audio(filepath, format="audio/mp3")
                        
                        # Display stats
                        st.text_area("Processing Stats", stats, height=150, key="char_stats")
                        
                        # Display subtitles
                        with st.expander("Subtitles (.srt format)"):
                            st.text(subtitles)
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            with open(filepath, "rb") as f:
                                st.download_button(
                                    "Download Audio",
                                    f,
                                    file_name="output_multi.mp3",
                                    mime="audio/mp3",
                                    key="dl_char_audio"
                                )
                        with col_dl2:
                            with open(srt_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "Download SRT",
                                    f,
                                    file_name="subtitles_multi.srt",
                                    mime="text/plain",
                                    key="dl_char_srt"
                                )
                    else:
                        st.error("Failed to generate multi-character audio")

if __name__ == "__main__":
    create_streamlit_app()
