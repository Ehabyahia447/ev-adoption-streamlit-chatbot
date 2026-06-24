import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError


PROJECT_CONTEXT_PATH = Path(__file__).with_name("project_context.md")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatbotSettings(BaseModel):
    provider: Literal["groq", "gemini"] = "groq"
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.3


def _read_secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name)


@st.cache_data(show_spinner=False)
def load_project_context() -> str:
    if PROJECT_CONTEXT_PATH.exists():
        return PROJECT_CONTEXT_PATH.read_text(encoding="utf-8")
    return (
        "EV adoption Streamlit project. Documentation file project_context.md "
        "was not found, so only limited project context is available."
    )


def load_chatbot_settings() -> ChatbotSettings:
    provider = (_read_secret("CHATBOT_PROVIDER") or "groq").lower()
    return ChatbotSettings(
        provider=provider,
        groq_api_key=_read_secret("GROQ_API_KEY"),
        gemini_api_key=_read_secret("GEMINI_API_KEY") or _read_secret("GOOGLE_API_KEY"),
        groq_model=_read_secret("GROQ_MODEL") or "llama-3.1-8b-instant",
        gemini_model=_read_secret("GEMINI_MODEL") or "gemini-2.0-flash",
    )


def build_system_prompt(
    project_context: str,
    dataset_shape: Optional[tuple] = None,
    feature_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    model_loaded: bool = False,
) -> str:
    feature_lines: List[str] = []
    if feature_metadata:
        for group_name, features in feature_metadata.items():
            names = ", ".join(features.keys())
            feature_lines.append(f"{group_name}: {names}")

    dataset_summary = (
        f"Current loaded dataset shape: {dataset_shape[0]} rows x {dataset_shape[1]} columns."
        if dataset_shape
        else "Dataset shape is not available in this session."
    )

    return f"""
You are the EV Adoption Smart Assistant inside a Streamlit data science project.

Your job:
- Explain the project goal, dataset, preprocessing, model, metrics, predictions, interpretation, limitations, and assumptions.
- Answer in the same language as the user when possible.
- Use simple diploma/final-project level explanations.
- Stay grounded in the project context. If a question is outside this EV adoption project, politely say it is outside the assistant scope.
- Do not invent model metrics, dataset facts, or code behavior that are not in the context.
- If the user asks how to run or configure the chatbot, explain Streamlit secrets and environment variables.

Project context:
{project_context}

Runtime context:
{dataset_summary}
Model file loaded in app: {"yes" if model_loaded else "no, app may be using fallback simulation behavior"}.
Feature groups from the app:
{chr(10).join(feature_lines)}
""".strip()


def _format_history_for_groq(system_prompt: str, history: List[dict]) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-8:])
    return messages


def _query_groq(settings: ChatbotSettings, system_prompt: str, history: List[dict]) -> str:
    if not settings.groq_api_key:
        return (
            "Groq API key is missing. Add GROQ_API_KEY to Streamlit secrets or "
            "as an environment variable, then restart the app."
        )

    response = requests.post(
        GROQ_CHAT_URL,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.groq_model,
            "messages": _format_history_for_groq(system_prompt, history),
            "temperature": settings.temperature,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _query_gemini(settings: ChatbotSettings, system_prompt: str, history: List[dict]) -> str:
    if not settings.gemini_api_key:
        return (
            "Gemini API key is missing. Add GEMINI_API_KEY or GOOGLE_API_KEY to "
            "Streamlit secrets or as an environment variable, then restart the app."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    contents = []
    for message in history[-8:]:
        contents.append(
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
        )

    response = requests.post(
        url,
        params={"key": settings.gemini_api_key},
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": settings.temperature},
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def query_llm(settings: ChatbotSettings, system_prompt: str, history: List[dict]) -> str:
    try:
        if settings.provider == "gemini":
            return _query_gemini(settings, system_prompt, history)
        return _query_groq(settings, system_prompt, history)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return f"LLM API error ({status}). Details: {detail}"
    except requests.RequestException as exc:
        return f"Connection error while calling the LLM API: {exc}"
    except (KeyError, IndexError, TypeError) as exc:
        return f"Unexpected LLM response format: {exc}"


def render_ev_chatbot(
    dataset_shape: Optional[tuple] = None,
    feature_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    model_loaded: bool = False,
) -> None:
    st.markdown('<div class="main-header">EV Adoption Smart Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask about the dataset, preprocessing, model, metrics, predictions, and limitations.</div>',
        unsafe_allow_html=True,
    )

    settings = load_chatbot_settings()
    project_context = load_project_context()
    system_prompt = build_system_prompt(project_context, dataset_shape, feature_metadata, model_loaded)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Chatbot Settings")
        st.caption(f"Provider: {settings.provider}")
        st.caption(
            f"Model: {settings.groq_model if settings.provider == 'groq' else settings.gemini_model}"
        )
        if st.button("Clear Chat Memory"):
            st.session_state.ev_chat_history = []
            st.rerun()

    if "ev_chat_history" not in st.session_state:
        st.session_state.ev_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi. I can explain this EV adoption project, including the data, "
                    "features, preprocessing, model, metrics, predictions, and limitations."
                ),
            }
        ]

    for message in st.session_state.ev_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about the EV adoption project...")
    if user_input is None:
        return

    clean_input = user_input.strip()
    if not clean_input:
        st.warning("Please write a question before sending.")
        return

    try:
        user_message = ChatMessage(role="user", content=clean_input)
    except ValidationError as exc:
        st.error(f"Invalid chat message: {exc}")
        return

    st.session_state.ev_chat_history.append(user_message.dict())
    with st.chat_message("user"):
        st.markdown(user_message.content)

    with st.chat_message("assistant"):
        with st.spinner("Asking the project assistant..."):
            answer = query_llm(settings, system_prompt, st.session_state.ev_chat_history)
            st.markdown(answer)

    st.session_state.ev_chat_history.append({"role": "assistant", "content": answer})
