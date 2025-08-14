---
title: Emotional Support Chat
emoji: 🫂
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
---

# 🫂 Emotional Support Chat

**Live demo:** [Use it on Hugging Face Spaces](https://huggingface.co/spaces/hashirehtisham/Emotional-Support-chat)

## 🚀 Features (Tabs)

- **Emotional Support Chatbot**  
  Friendly, empathetic replies for check-ins and general support.

- **Motivational Quotes**  
  Short, uplifting quotes and one-liners on demand.

- **Emotions Detector**  
  Detects tone (happy, sad, angry, neutral), identifies style (formal/casual/slang), and suggests wording improvements for the intended audience. Provides analysis only (no chatting).

- **Jokes for You**  
  Unique, non-repeating, family-friendly jokes with topic rotation (wordplay, everyday life, animals, professions, etc.).

## 🛠 Tech Stack

- **Frontend/UI:** [Gradio 5 (Blocks)](https://www.gradio.app/) with custom CSS (styled buttons, hidden footer, hover effects)  
- **LLM & Inference:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) via Hugging Face **InferenceClient**  
- **Runtime:** Python (standard library `os` for env vars)  
- **Streaming:** Token-by-token streaming for responsive outputs  
- **Config:** Uses `HF_TOKEN1` environment variable for the Hugging Face API token
