
from transformers import AutoTokenizer
print("Testing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer works!")

from transformers import AutoModelForSequenceClassification
print("Testing your team's model...")
model = AutoModelForSequenceClassification.from_pretrained("keanteng/bert-large-raw-climate-sentiment-wqf7007")
print("✅ Your model works!")

from transformers import AutoTokenizer
print("Testing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer works!")

from transformers import AutoModelForSequenceClassification
print("Testing your team's model...")
model = AutoModelForSequenceClassification.from_pretrained("keanteng/bert-large-raw-climate-sentiment-wqf7007")
print("✅ Your model works!")

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
        """Load the pre-trained BERT model"""
        try:
            print("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("keanteng/bert-large-raw-climate-sentiment-wqf7007")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()

            print(f"✅ Model loaded successfully on {device}!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")

    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded"

        if not text.strip():
            return "⚠️ Please enter some text"

        try:
            # Tokenize input
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )

            # Move inputs to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(outputs.logits, dim=1)

            # Map model output to sentiment labels
            predicted_label = prediction.item()
            if predicted_label == 0:
                sentiment_key = -1  # anti
            elif predicted_label == 1:
                sentiment_key = 0   # neutral
            elif predicted_label == 2:
                sentiment_key = 1   # pro
            else:
                sentiment_key = 2   # news

            confidence = probabilities[0][predicted_label].item()
            sentiment = self.sentiment_labels[sentiment_key]

            result = f"""
**🎯 Prediction:** {sentiment}
**📊 Confidence:** {confidence:.2%}
**🤖 Model:** keanteng/bert-large-raw-climate-sentiment-wqf7007
**⚡ Device:** {device}

✅ *Real prediction from your team's BERT model*
            """

            return result

        except Exception as e:
            return f"❌ Prediction failed: {str(e)}"

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if not texts:
            return "⚠️ No texts to analyze", None

        text_list = [t.strip() for t in texts.split('\n') if t.strip()]

        if not text_list:
            return "⚠️ No valid texts found", None

        results = []
        for text in text_list[:20]:  # Limit to 20 texts
            pred_result = self.predict_sentiment(text)
            if "❌" not in pred_result and "⚠️" not in pred_result:
                lines = pred_result.split('\n')
                sentiment = lines[1].split('**🎯 Prediction:** ')[1]
                confidence = lines[2].split('**📊 Confidence:** ')[1]

                results.append({
                    "Text": text[:80] + "..." if len(text) > 80 else text,
                    "Sentiment": sentiment,
                    "Confidence": confidence
                })

        if not results:
            return "❌ No successful predictions", None

        # Create summary
        sentiments = [r["Sentiment"] for r in results]
        sentiment_counts = Counter(sentiments)

        summary = f"**Analysis Results:**\n\nTotal texts: {len(results)}\n\n**Distribution:**\n"

        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results)) * 100
            summary += f"\n- **{sentiment}**: {count} ({percentage:.1f}%)"

        df = pd.DataFrame(results)
        return summary, df

# Initialize analyzer
print("🌍 Initializing Climate Change Sentiment Analyzer...")
analyzer = ClimateChangeAnalyzer()

# Create Gradio interface
with gr.Blocks(title="Climate Change Sentiment Analysis") as demo:

    gr.HTML("""
    <div style="text-align: center;">
        <h1>🌍 Climate Change Sentiment Analysis</h1>
        <p><strong>Group 4</strong></p>
        <p><i>Model: keanteng/bert-large-raw-climate-sentiment-wqf7007</i></p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Single Text Analysis"):
            single_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="e.g., Climate change is real and requires urgent action...",
                lines=4
            )
            single_button = gr.Button("🤖 Analyze Sentiment", variant="primary")
            single_output = gr.Markdown(label="Analysis Result")

            gr.Examples(
                examples=[
                    ["Climate change is the most pressing issue of our time."],
                    ["The weather has always changed naturally throughout history."],
                    ["Scientists report that global temperatures have risen significantly."],
                    ["I'm not convinced that humans are causing climate change."]
                ],
                inputs=single_input
            )

        with gr.Tab("Batch Analysis"):
            batch_input = gr.Textbox(
                label="Enter multiple texts (one per line)",
                placeholder="Climate change is real\nGlobal warming is a hoax\nRenewable energy is important",
                lines=8
            )
            batch_button = gr.Button("🤖 Analyze All Texts", variant="primary")
            gr.Examples(
                examples=[
                    ["Climate change is the most urgent crisis facing humanity today.\nThe weather has always changed naturally throughout history.\nScientists report that global temperatures have risen by 1.1°C since pre-industrial times.\nI don't believe humans are the main cause of climate change.\nRenewable energy investments reached record highs in 2023.\nGlobal warming is just a natural cycle, not caused by humans.\nThe IPCC report confirms that human activities are driving climate change.\nClimate policies will destroy our economy and jobs.\nSea levels are rising faster than previously predicted.\nThere's still debate among scientists about climate change causes."]
                ],
                inputs=batch_input
            )
            batch_summary = gr.Markdown(label="Summary")
            batch_results = gr.Dataframe(label="Detailed Results")

        with gr.Tab("About"):
            gr.Markdown("""
            ## About This System

            **Model**: `keanteng/bert-large-raw-climate-sentiment-wqf7007`

            **Sentiment Categories:**
            - **Pro-Climate**: Expresses support for climate science, policies, or actions addressing climate change.
            - **Anti-Climate**: Shows skepticism about or denial of climate change or its human causes.
            - **Neutral**: Contains a balanced tone, unclear stance, or no strong opinion on the topic.
            - **News**: Reports factual information or news updates related to climate change, without personal opinion.

            **Team:**
            - Data Preprocessing: Loong Shih-Wai
            - Modeling: Khor Kean Teng
            - Results Evaluation: Huang Lili
            - Gradio App Development: ZOU JINGYI
            - Gradio App Deployment: Xian Zhi Yi

            **SDG 13: Climate Action** 🌍
            """)

    # Connect functions
    single_button.click(
        fn=analyzer.predict_sentiment,
        inputs=single_input,
        outputs=single_output
    )

    batch_button.click(
        fn=analyzer.predict_batch,
        inputs=batch_input,
        outputs=[batch_summary, batch_results]
    )

# Launch the app
print("🚀 Starting application...")
demo.launch(share=True, debug=True)
