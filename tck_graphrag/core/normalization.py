"""
Entity Normalizasyon Modülü

TCK entity isimlerini standart forma çevirir.
LLM çıktısında aynı entity farklı biçimlerde gelebilir:
  - "hırsızlık" / "HIRSIZLIK" / "Hırsızlık"
  - "Madde  141" / "madde 141"

Bu modül hepsini tek standart forma dönüştürür.
"""

import re
import unicodedata
from typing import Optional


# Türkçe karakter dönüşüm tabloları
_TR_LOWER_MAP = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ",
                               "abcçdefgğhıijklmnoöpqrsştüüvwxyz")

_TR_UPPER_MAP = str.maketrans("abcçdefgğhıijklmnoöpqrsştüüvwxyz",
                               "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ")

# Bilinen entity tipleri — whitelist
VALID_ENTITY_TYPES = {
    "MADDE", "SUC", "CEZA", "SURE",
    "KOSUL", "KAVRAM", "TANIM",
    "KISIM", "BOLUM"
}

# Bilinen ilişki tipleri — whitelist
VALID_RELATIONSHIP_TYPES = {
    "TANIMLAR", "CEZA_OLARAK", "SURE_OLARAK",
    "AGIRLAŞTIRIR", "HAFIFLETIR", "NITELIKLISI",
    "REFERANS_VERIR", "ICERIR", "ILGILI"
}

# Entity tipinden Neo4j label'ına dönüşüm
ENTITY_TYPE_TO_LABEL = {
    "MADDE": "Madde",
    "SUC": "Suc",
    "CEZA": "Ceza",
    "SURE": "Sure",
    "KOSUL": "Kosul",
    "KAVRAM": "Kavram",
    "TANIM": "Tanim",
    "KISIM": "Kisim",
    "BOLUM": "Bolum"
}

# UI / dokümantasyon (paper figure) — Neo4j type kodu değişmez
ENTITY_TYPE_DISPLAY = {
    "SUC": "Crime",
    "KOSUL": "Condition",
    "MADDE": "Article",
    "SURE": "Duration",
    "KAVRAM": "Concept",
    "CEZA": "Penalty",
    "TANIM": "Definition",
}


def format_entity_type_display(entity_type: str) -> str:
    """Neo4j type kodunu kullanıcıya gösterilecek etikete çevirir."""
    if not entity_type:
        return ""
    normalized = entity_type.strip().upper()
    return ENTITY_TYPE_DISPLAY.get(normalized, entity_type)


def tr_lower(text: str) -> str:
    """Türkçe-aware lowercase dönüşümü.
    
    Python'un str.lower() fonksiyonu Türkçe'de sorunlu:
      "İ".lower() → "i̇"  (yanlış)  — olması gereken: "i"
      "I".lower() → "i"   (yanlış)  — olması gereken: "ı"
    
    Bu fonksiyon Türkçe karakterleri doğru dönüştürür.
    """
    return text.translate(_TR_LOWER_MAP)


def tr_upper(text: str) -> str:
    """Türkçe-aware uppercase dönüşümü."""
    return text.translate(_TR_UPPER_MAP)


def tr_title(text: str) -> str:
    """Türkçe-aware title case dönüşümü.
    
    Her kelimenin ilk harfi büyük, geri kalanı küçük.
    "hırsızlık" → "Hırsızlık"
    "kasten öldürme" → "Kasten Öldürme"
    """
    words = text.split()
    result = []
    for word in words:
        if word:
            first = tr_upper(word[0])
            rest = tr_lower(word[1:]) if len(word) > 1 else ""
            result.append(first + rest)
    return " ".join(result)


def normalize_whitespace(text: str) -> str:
    """Boşlukları standartlaştır.
    
    - Baştaki/sondaki boşlukları kaldır
    - Ardışık boşlukları tek boşluğa çevir
    - Tab ve newline'ları boşluğa çevir
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_entity_name(name: str) -> str:
    """Entity ismini standart forma çevirir.
    
    Kurallar:
    1. Boşluk standardizasyonu
    2. Unicode normalizasyon (NFC)
    3. Şapkalı karakter standardizasyonu (â→a, î→i, û→u)
    4. Madde isimleri: "Madde X" formatı (M büyük, tek boşluk)
    5. Diğer isimler: Title Case (her kelimenin ilk harfi büyük)
    
    Örnekler:
        "hırsızlık"           → "Hırsızlık"
        "HIRSIZLIK"           → "Hırsızlık"  
        "Madde  141"          → "Madde 141"
        "madde 141"           → "Madde 141"
        " Hapis Cezası  "     → "Hapis Cezası"
        "kasten ÖLDÜRME"      → "Kasten Öldürme"
        "Zorunluluk Hâli"     → "Zorunluluk Hali"
        "Adlî Para Cezası"    → "Adli Para Cezası"
    """
    if not name:
        return name
    
    # 1. Boşluk standardizasyonu
    name = normalize_whitespace(name)
    
    # 2. Unicode normalizasyon (NFC formu — composed characters)
    name = unicodedata.normalize("NFC", name)
    
    # 3. Şapkalı karakter standardizasyonu
    #    TCK'da "hâl/hal", "adlî/adli" gibi tutarsızlıklar var
    name = name.replace("â", "a").replace("Â", "A")
    name = name.replace("î", "i").replace("Î", "İ")
    name = name.replace("û", "u").replace("Û", "U")
    
    # 4. "Madde" isimleri için özel format
    madde_match = re.match(r'[Mm][Aa][Dd][Dd][Ee]\s+(\d+)', name)
    if madde_match:
        madde_no = madde_match.group(1)
        return f"Madde {madde_no}"
    
    # 5. Tüm büyük veya tüm küçük ise → Title Case
    if name == tr_upper(name) or name == tr_lower(name):
        return tr_title(name)

    # 6. Karışık case: her kelimenin ilk harfi büyük mü kontrol et (Title Case mi?)
    #    "Adli Para Cezası" → zaten Title Case, olduğu gibi bırak
    #    "kasten ÖLDÜRME" → Title Case değil, düzelt
    words = name.split()
    is_title_case = all(
        w[0] == tr_upper(w[0]) for w in words if w and w[0].isalpha()
    )
    if is_title_case:
        return name

    return tr_title(name)


def normalize_entity_type(entity_type: str) -> str:
    """Entity tipini standartlaştır.
    
    "suc" → "SUC", "Madde" → "MADDE", "bilinmeyen" → "KAVRAM"
    """
    if not entity_type:
        return "KAVRAM"
    
    normalized = entity_type.strip().upper()
    
    if normalized in VALID_ENTITY_TYPES:
        return normalized
    
    # Bilinen olmayan tipleri KAVRAM'a düşür
    return "KAVRAM"


def normalize_relationship_type(rel_type: str) -> str:
    """İlişki tipini standartlaştır.
    
    "ceza_olarak" → "CEZA_OLARAK", "bilinmeyen" → "ILGILI"
    """
    if not rel_type:
        return "ILGILI"
    
    normalized = rel_type.strip().upper()
    
    # Türkçe karakter düzeltmeleri
    normalized = normalized.replace("Ş", "Ş").replace("Ğ", "Ğ")
    
    if normalized in VALID_RELATIONSHIP_TYPES:
        return normalized
    
    # Bilinen olmayan tipleri ILGILI'ya düşür
    return "ILGILI"


def get_neo4j_label(entity_type: str) -> str:
    """Entity tipinden Neo4j label'ı döndür.
    
    "SUC" → "Suc", "MADDE" → "Madde"
    """
    normalized_type = normalize_entity_type(entity_type)
    return ENTITY_TYPE_TO_LABEL.get(normalized_type, "Kavram")


def extract_madde_no(text: str) -> Optional[int]:
    """Metinden madde numarasını çıkar.
    
    "Madde 141" → 141
    "Madde 82" → 82
    "Hırsızlık" → None
    """
    match = re.search(r'[Mm]adde\s+(\d+)', text)
    if match:
        return int(match.group(1))
    return None
