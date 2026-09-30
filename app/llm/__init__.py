from app.config import Settings
from app.llm.base import LLMClient, LLMError, LLMNotConfiguredError, Message
from app.llm.fake import FakeLLM
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient

__all__ = ["LLMClient", "LLMError", "LLMNotConfiguredError", "Message", "create_llm_client"]


def create_llm_client(settings: Settings) -> LLMClient:
    """Pick the provider from LLM_PROVIDER. Switching providers is one env variable."""
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    if settings.llm_provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model)
    if settings.llm_provider == "fake":
        return FakeLLM()
    raise LLMNotConfiguredError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
