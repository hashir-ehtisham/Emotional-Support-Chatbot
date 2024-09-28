import gradio as gr
from huggingface_hub import InferenceClient

# CSS to hide footer and customize button
css = """
footer {display:none !important}
.output-markdown{display:none !important}

.gr-button-primary {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important; 
    background: none rgb(17, 20, 45) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: none !important;
}
.gr-button-primary:hover {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important;
    background: none rgb(66, 133, 244) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: rgb(0 0 0 / 23%) 0px 1px 7px 0px !important;
}
.hover\:bg-orange-50:hover {
    --tw-bg-opacity: 1 !important;
    background-color: rgb(229,225,255) !important;
}

.to-orange-200 {
    --tw-gradient-to: rgb(37 56 133 / 37%) !important;
}

.from-orange-400 {
    --tw-gradient-from: rgb(17, 20, 45) !important;
    --tw-gradient-to: rgb(255 150 51 / 0);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to) !important;
}

.group-hover\:from-orange-500 {
    --tw-gradient-from:rgb(17, 20, 45) !important; 
    --tw-gradient-to: rgb(37 56 133 / 37%);
    --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to) !important;
}

.group:hover .group-hover\:text-orange-500 {
    --tw-text-opacity: 1 !important;
    color:rgb(37 56 133 / var(--tw-text-opacity)) !important;
}
"""

# Initialize the InferenceClient for chatbot
client = InferenceClient("HuggingFaceH4/zephyr-7b-alpha")

# Define the function for chatbot response
def respond(
    message,
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

def send_message(message, history, system_message, max_tokens, temperature, top_p):
    if message:
        history.append((message, ""))
        response = respond(
            message=message,
            history=history,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = ""
        for r in response:
            response_text = r
        history[-1] = (message, response_text)
    return history, gr.update(value="")

# Description for the chatbot
description = """
Hello! I'm here to support you emotionally and answer any questions. How are you feeling today?
<div style='color: green;'>Developed by Hashir Ehtisham</div>
"""

# Motivational tagline for the new tab
motivational_tagline = """
Welcome to the Motivational Quotes tab! Let’s ignite your day with some inspiration. What do you need motivation for today?
<div style='color: green;'>Developed by Hashir Ehtisham</div>
"""

# Emotions Detector tagline for the new tab
emotions_detector_tagline = """
Know how your message sounds and how to improve the tone of the message with Emotions Detector.
<div style='color: green;'>Developed by Hashir Ehtisham</div>
"""

# Jokes tagline for the new tab
jokes_tagline = """
Ready for a good laugh? Ask me for a joke to lighten up your mood!
<div style='color: green;'>Developed by Hashir Ehtisham</div>
"""

# Define the Gradio Blocks interface
with gr.Blocks(css=css) as demo:
    with gr.Tab("Emotional Support Chatbot"):
        gr.Markdown("# Emotional Support Chatbot")
        gr.Markdown(description)
        
        system_message = gr.Textbox(value="You are a friendly Emotional Support Chatbot.", visible=False)
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear")
        
        with gr.Accordion("Additional Inputs", open=False):
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=""), chat_history

        msg.submit(respond_wrapper, [msg, chatbot, system_message, max_tokens, temperature, top_p], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    with gr.Tab("Motivational Quotes"):
        gr.Markdown("# Motivational Quotes")
        gr.Markdown(motivational_tagline)
        
        system_message_motivational = gr.Textbox(value="You are a friendly Motivational Quotes Chatbot.", visible=False)
        chatbot_motivational = gr.Chatbot()
        msg_motivational = gr.Textbox(label="Your message")
        clear_motivational = gr.Button("Clear")
        
        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_motivational = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature_motivational = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_motivational = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper_motivational(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=""), chat_history

        msg_motivational.submit(respond_wrapper_motivational, [msg_motivational, chatbot_motivational, system_message_motivational, max_tokens_motivational, temperature_motivational, top_p_motivational], [msg_motivational, chatbot_motivational])
        clear_motivational.click(lambda: None, None, chatbot_motivational, queue=False)
    
    with gr.Tab("Emotions Detector"):
        gr.Markdown("# Emotions Detector")
        gr.Markdown(emotions_detector_tagline)
        
        system_message_emotions = gr.Textbox(value="You are an Emotions Detector Chatbot. Analyze the tone of the message (happy, sad, angry, neutral) and answer back.", visible=False)
        chatbot_emotions = gr.Chatbot()
        msg_emotions = gr.Textbox(label="Your message")
        clear_emotions = gr.Button("Clear")
        
        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_emotions = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature_emotions = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_emotions = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper_emotions(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=""), chat_history

        msg_emotions.submit(respond_wrapper_emotions, [msg_emotions, chatbot_emotions, system_message_emotions, max_tokens_emotions, temperature_emotions, top_p_emotions], [msg_emotions, chatbot_emotions])
        clear_emotions.click(lambda: None, None, chatbot_emotions, queue=False)
    
    with gr.Tab("Jokes for You"):
        gr.Markdown("# Jokes for You")
        gr.Markdown(jokes_tagline)
        
        system_message_jokes = gr.Textbox(value="You are a friendly Jokes Chatbot. Provide a joke when asked.", visible=False)
        chatbot_jokes = gr.Chatbot()
        msg_jokes = gr.Textbox(label="Your message")
        clear_jokes = gr.Button("Clear")
        
        with gr.Accordion("Examples", open=False):
            gr.Examples(
                examples=[
                    ["Tell me a joke"],
                    ["Make me laugh"],
                    ["Say something funny"],
                ],
                inputs=msg_jokes,
            )

        with gr.Accordion("Additional Inputs", open=False):
            max_tokens_jokes = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature_jokes = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
            top_p_jokes = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

        def respond_wrapper_jokes(message, chat_history, system_message_val, max_tokens_val, temperature_val, top_p_val):
            chat_history, _ = send_message(
                message=message,
                history=chat_history,
                system_message=system_message_val,
                max_tokens=max_tokens_val,
                temperature=temperature_val,
                top_p=top_p_val,
            )
            return gr.update(value=""), chat_history

        msg_jokes.submit(respond_wrapper_jokes, [msg_jokes, chatbot_jokes, system_message_jokes, max_tokens_jokes, temperature_jokes, top_p_jokes], [msg_jokes, chatbot_jokes])
        clear_jokes.click(lambda: None, None, chatbot_jokes, queue=False)

# Launch the Gradio interface
demo.launch()
