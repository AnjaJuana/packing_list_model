# Prerequisites
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Set 1, 2, or 4 depending on CPU usage

from transformers import pipeline
import json
import pandas as pd
import gradio as gr

# get candidate labels
with open("packing_label_structure.json", "r") as file:
    candidate_labels = json.load(file)
keys_list = list(candidate_labels.keys())

# Load packing item data
with open("packing_templates_self_supported_offgrid_expanded.json", "r") as file:
    packing_items = json.load(file)

# function and gradio app
def classify(model_name, trip_descr, cut_off = 0.5):
    classifier = pipeline("zero-shot-classification", model=model_name)
    ## Create and fill dataframe with class predictions
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

    ## Look up and return list of items to pack based on class predictions
    # make list from dataframe column
    all_classes = [elem for x in df["pred_class"] for elem in (x if isinstance(x, list) else [x])]
    # look up packing items for each class/key
    list_of_list_of_items = [packing_items.get(k, []) for k in all_classes]
    # combine lists and remove doubble entries
    flat_unique = []
    for sublist in list_of_list_of_items:
        for item in sublist:
            if item not in flat_unique:
                flat_unique.append(item)
    # sort alphabetically to notice duplicates
    sorted_list = sorted(flat_unique)  
    return df, "\n".join(sorted_list)

demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(label="Model name", value = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"),
        gr.Textbox(label="Trip description"),
        gr.Number(label="Activity cut-off", value = 0.5),
    ],
    # outputs="dataframe",
    outputs=[gr.Dataframe(label="DataFrame"), gr.Textbox(label="List of words")],
    title="Trip classification",
    description="Enter a text describing your trip",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()