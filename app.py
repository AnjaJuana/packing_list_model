# Prerequisites
from transformers import pipeline
import json
import pandas as pd
import gradio as gr

# get candidate labels
with open("packing_label_structure.json", "r") as file:
    candidate_labels = json.load(file)
keys_list = list(candidate_labels.keys())

# Load test data (in list of dictionaries)
with open("test_data.json", "r") as file:
    packing_data = json.load(file)

# function and gradio app
model_name = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)
cut_off = 0.5  # used to choose which activities are relevant

def classify(#model_name, 
             trip_descr, cut_off):
    
    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=['superclass', 'pred_class'])
    for i, key in enumerate(keys_list):
        if key == 'activities':
            result = classifier(trip_descr, candidate_labels[key], multi_label=True)
            indices = [i for i, score in enumerate(result['scores']) if score > cut_off]
            classes = [result['labels'][i] for i in indices]
        else:
            result = classifier(trip_descr, candidate_labels[key])
            classes = result["labels"][0]
        df.loc[i] = [key, classes]
    return df

demo = gr.Interface(
    fn=classify,
    inputs=[
        #gr.Textbox(label="Model name", value = "facebook/bart-large-mnli"),
        gr.Textbox(label="Trip description"),
        gr.Number(label="Activity cut-off", value = 0.5),
    ],
    outputs="dataframe",
    title="Trip classification",
    description="Enter a text describing your trip",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()