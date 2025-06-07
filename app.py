
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from collections import Counter

class ClimateChangeAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sentiment_labels = {
            -1: "Anti-Climate",
            0: "Neutral",
            1: "Pro-Climate",
            2: "News"
        }
        self.load_model()

    def load_model(self):
        try:
            print("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("keanteng/bert-large-raw-climate-sentiment-wqf7007")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            print(f"Model loaded successfully on {device}!")
        except Exception as e:
            print(f"Model loading failed: {e}")

    def predict_sentiment(self, text):
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        if not text.strip():
            return "Please enter some text"

        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(outputs.logits, dim=1)

            predicted_label = prediction.item()
            if predicted_label == 0:
                sentiment_key = -1
            elif predicted_label == 1:
                sentiment_key = 0
            elif predicted_label == 2:
                sentiment_key = 1
            else:
                sentiment_key = 2

            confidence = probabilities[0][predicted_label].item()
            sentiment = self.sentiment_labels[sentiment_key]
            return f"Prediction: {sentiment}\nConfidence: {confidence:.2%}\nModel: keanteng/bert-large-raw-climate-sentiment-wqf7007\nDevice: {device}"
        except Exception as e:
            return f"Prediction failed: {str(e)}"

    def predict_batch(self, texts):
        if not texts:
            return "No texts to analyze", None
        text_list = [t.strip() for t in texts.split('\n') if t.strip()]
        if not text_list:
            return "No valid texts found", None
        results = []
        for text in text_list[:20]:
            pred_result = self.predict_sentiment(text)
            if "Prediction failed" not in pred_result and "Please enter" not in pred_result:
                lines = pred_result.split('\n')
                sentiment = lines[0].split('Prediction: ')[1]
                confidence = lines[1].split('Confidence: ')[1]
                results.append({
                    "Text": text[:80] + "..." if len(text) > 80 else text,
                    "Sentiment": sentiment,
                    "Confidence": confidence
                })
        if not results:
            return "No successful predictions", None
        sentiments = [r["Sentiment"] for r in results]
        sentiment_counts = Counter(sentiments)
        summary = f"Analysis Results:\nTotal texts: {len(results)}\nDistribution:\n"
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results)) * 100
            summary += f"- {sentiment}: {count} ({percentage:.1f}%)\n"
        df = pd.DataFrame(results)
        return summary, df

analyzer = ClimateChangeAnalyzer()

with gr.Blocks(title="Climate Change Sentiment Analysis") as demo:
    gr.Markdown("# Climate Change Sentiment Analysis\nModel: keanteng/bert-large-raw-climate-sentiment-wqf7007")
    with gr.Tabs():
        with gr.Tab("Single Text Analysis"):
            single_input = gr.Textbox(label="Enter text", lines=4)
            single_button = gr.Button("Analyze")
            single_output = gr.Markdown()
        with gr.Tab("Batch Analysis"):
            batch_input = gr.Textbox(label="Enter multiple texts (one per line)", lines=8)
            batch_button = gr.Button("Analyze All")
            batch_summary = gr.Markdown()
            batch_results = gr.Dataframe()
    single_button.click(fn=analyzer.predict_sentiment, inputs=single_input, outputs=single_output)
    batch_button.click(fn=analyzer.predict_batch, inputs=batch_input, outputs=[batch_summary, batch_results])

demo.launch()
