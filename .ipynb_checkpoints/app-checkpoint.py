import gradio as gr
from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-base")

# Define the classification function
def classify_text(text, labels):
    labels = labels.split(",")  # Convert the comma-separated string into a list
    result = classifier(text, candidate_labels=labels)
    return result

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Zero-Shot Classification")
    text_input = gr.Textbox(label="Input Text")
    label_input = gr.Textbox(label="Comma-separated Labels")
    output = gr.JSON(label="Result")
    classify_button = gr.Button("Classify")

    # Link the button to the classification function
    classify_button.click(classify_text, inputs=[text_input, label_input], outputs=output)

# Launch the Gradio interface
demo.launch()
