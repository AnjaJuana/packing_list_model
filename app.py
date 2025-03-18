from transformers import pipeline
import gradio as gr

# Load the model and create a pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-base")

# Load labels from a txt file
with open("labels.txt", "r", encoding="utf-8") as f:
    class_labels = [line.strip() for line in f if line.strip()]

# Define the Gradio interface
def classify(text):
    return classifier(text, class_labels)

demo = gr.Interface(
    fn=classify,
    inputs="text",
    outputs="json",
    title="Zero-Shot Classification",
    description="Enter a text describing your trip",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()