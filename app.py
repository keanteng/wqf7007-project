# use the one from lecture lab class demonstration
import gradio as gr
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab') #model for sentence tokenizer

def nltk_tokenizer(text):
    tokens = word_tokenize(text)
    return '\n'.join(tokens)

# Example input sentences
examples = [
    ["Hello! How are you doing today?"],
    ["NLP is fun, isn't it?"],
    ["Dr. Smith went to Washington D.C. on Jan 5, 2024."],
    ["I can't believe it's already April!"],
    ["The chicken is ready to eat."]
]

# Gradio Interface
gr.Interface(
    fn=nltk_tokenizer,
    inputs=gr.Textbox(lines=3, placeholder="Enter some text here..."),
    outputs=gr.Textbox(label="Tokens", lines=3, interactive=False),
    title="NLTK Tokenizer",
    description="Tokenizes input text using NLTK's word_tokenize.",
    examples=examples
).launch()