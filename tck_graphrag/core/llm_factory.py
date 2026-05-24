"""
LLM Factory - OpenAI ve Anthropic (Claude) desteği

Bu modül, farklı LLM sağlayıcıları için tek bir arayüz sağlar.
Akademik karşılaştırma çalışmaları için model değişikliği kolaylaştırılmıştır.

Kullanım:
    from tck_graphrag.core.llm_factory import get_llm, get_embeddings
    
    # Config'den otomatik
    llm = get_llm()
    
    # Manuel override
    llm = get_llm(provider="anthropic", model="claude-sonnet-4-20250514")
"""

import os
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from tck_graphrag.core.config import get_settings

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()


# Desteklenen modeller
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]


def resolve_provider_for_model(model: str) -> Literal["openai", "anthropic"]:
    """Model adından LLM sağlayıcısını çıkarır."""
    if model in ANTHROPIC_MODELS or model.lower().startswith("claude"):
        return "anthropic"
    return "openai"


def get_llm(
    provider: Optional[Literal["openai", "anthropic"]] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs
) -> BaseChatModel:
    """
    LLM instance oluşturur.
    
    Args:
        provider: "openai" veya "anthropic". None ise config'den alınır.
        model: Model ismi. None ise config'den alınır.
        temperature: Sampling temperature (varsayılan: 0.1)
        **kwargs: Ek model parametreleri
    
    Returns:
        LangChain ChatModel instance
    
    Raises:
        ValueError: Geçersiz provider veya eksik API key
    
    Örnek:
        # Config'den
        llm = get_llm()
        
        # Manuel
        llm = get_llm(provider="anthropic", model="claude-sonnet-4-20250514")
    """
    settings = get_settings()
    
    # Provider ve model belirleme
    _provider = provider or settings.llm_provider
    _model = model or settings.llm_model
    
    if _provider == "openai":
        return _create_openai_llm(_model, temperature, **kwargs)
    elif _provider == "anthropic":
        return _create_anthropic_llm(_model, temperature, **kwargs)
    else:
        raise ValueError(f"Desteklenmeyen provider: {_provider}. 'openai' veya 'anthropic' kullanın.")


def _create_openai_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """OpenAI ChatModel oluşturur."""
    from langchain_openai import ChatOpenAI
    
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key bulunamadı! "
            ".env dosyasına OPENAI_API_KEY ekleyin veya ortam değişkeni olarak tanımlayın."
        )
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def _create_anthropic_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    """Anthropic (Claude) ChatModel oluşturur."""
    from langchain_anthropic import ChatAnthropic
    
    settings = get_settings()
    api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Anthropic API key bulunamadı! "
            ".env dosyasına ANTHROPIC_API_KEY ekleyin veya ortam değişkeni olarak tanımlayın."
        )
    
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def get_embeddings(model: str = "text-embedding-3-small") -> Embeddings:
    """
    OpenAI Embeddings instance oluşturur.
    
    Not: Anthropic'in kendi embedding modeli yok, bu yüzden her zaman OpenAI kullanılır.
    Bu, karşılaştırma çalışmalarında embedding'lerin sabit kalmasını sağlar (fair comparison).
    
    Args:
        model: Embedding model ismi (varsayılan: text-embedding-3-small)
    
    Returns:
        LangChain Embeddings instance
    """
    from langchain_openai import OpenAIEmbeddings
    
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key gerekli (embedding için).")
    
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def get_provider_info() -> dict:
    """Mevcut LLM konfigürasyonunu döndürür."""
    settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "openai_key_set": bool(settings.openai_api_key or os.getenv("OPENAI_API_KEY")),
        "anthropic_key_set": bool(settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")),
    }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Factory Test")
    print("=" * 60)
    
    info = get_provider_info()
    print(f"\nKonfigürasyon:")
    print(f"  Provider: {info['provider']}")
    print(f"  Model: {info['model']}")
    print(f"  OpenAI key: {'✓' if info['openai_key_set'] else '✗'}")
    print(f"  Anthropic key: {'✓' if info['anthropic_key_set'] else '✗'}")
    
    # Test: Config'den LLM
    try:
        llm = get_llm()
        print(f"\n✅ Config'den LLM oluşturuldu: {type(llm).__name__}")
        
        # Basit test
        response = llm.invoke("Merhaba, 2+2 kaç eder?")
        print(f"   Test yanıtı: {response.content[:100]}...")
    except Exception as e:
        print(f"\n❌ Config LLM hatası: {e}")
    
    # Test: Manuel override
    if info['anthropic_key_set']:
        try:
            claude = get_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")
            print(f"\n✅ Claude LLM oluşturuldu: {type(claude).__name__}")
        except Exception as e:
            print(f"\n❌ Claude LLM hatası: {e}")
