# Working with Large Language Models
## Introduction: Nikky

Nowadays many people regularly use AI chatbots. These chatbots “understand” text and talk with you like they are humans with an above average level of knowledge (…when they are not hallucinating or answering falsely ;-) ). Behind such bots and systems is usually a Large Language Model (LLM). This is an AI-model trained on enormous amount of texts, so that it can recognize the meaning and logic of language.

## Why bother to adapt models to your application: Nikky
### What are LLM’s used for in practice?

A cool characteristic of LLM’s, is that they can respond in a very logical and natural way. That is why they are currently used more and more to analyze text, but also to organize information and even to generate creative ideas. AI agents are now also gaining very much in popularity, these agents are using LLM’s and automatization to learn and automatize processes. Since we were interested in their potential on a real life application, we tested how we can use LLM’s in providing personalized packing lists to know what to take on your holiday, adventure travel and/or business trip.

## The application: Personalized Holiday packing lists

The goal of the project is to develop a framework using LLM’s that automatically generates packing lists based on a textual description of a trip. This description can be really short (“a weekend camping in the mountains”) or much more elaborate (“10 days tour through Iceland in summer with hikes, camping and remote workdays. I will also do strength training, yoga and sleep outside”). The model will for instance try to understand the type, destination and duration of the trip, the season, and potential extra activities or characteristics and will then provice a list of potentially necessary items to pack for each trip.

We tested several open source NLI en zero-shot models (such as BART and DeBERTa - more about them later) to see which one performs the best on several different indicators. We have found that the standard models and options are not always directly fittable to any situation, and that when you plan to ‘use an LLM in practice’ for your own application, you might need some extra steps to make the output more reliable, consistent and usable.

In this post, we hope to outline all the steps we took and share our search for the best approach and model. We will share the code we used and the considerations we had on the way, to provide you with the tools to start applying LLM’s for your own projects and applications.

### Different types of (open-source) LLM’s

So many different LLM’s exist, with each of them having strengths and weaknesses. Some well known models such as GPT, Claude, Gemeni, LlaMa and Mistral are used for text generation and chats/conversations. For applications such as our project, where a model should understand text and than provide a list based on categories that are taken from the provided description, classification and NLI models such as BART, DeBERTa and RoBERTa can be more suitable. These type of classification models are better in recognizing the meaning and then giving a fixed set of most probable categories as outputs.

### Output: Free text or classification?

When you use a normal text-generating LLM, there are many free parameters, and packing lists outcome can then become very chaotic and unstable over time. This is why a classification model can help in providing users with well structured packing lists with the necessary items for their trip. Of course, this choice highly depends on what you would like as a final output for your application. In case you want a conversation, the text generating models would be of better use. Since we really want to provide the user with a list of items based on categories derived from the trip description, using classification models as the base for the framework seemed to be more suitable for our application.

### Trainability of LLM’s

The existing LLM’s as mentioned in the above introduction can be either used “off-the-shelf” without modification, or you can provide them with extra information regarding your specific application. If you use an text-generating LLM, as a chatbot for your customer service, you can train the model by giving more information about your specific return policy, for instance. In case of the zero-shot models, you may specify categories specific to your application, so that you will always receive the estimates for that set of categories you’re interested in.

### Open-source platforms and models

Many powerful LLM’s are available on platforms such as  
- [Hugging Face](https://huggingface.co/)  
- [Replicate](https://replicate.com/)  
- [ModelScope](https://modelscope.cn/)  
- [OpenRouter](https://openrouter.ai/)

This makes it possible to use existing models, compare them before using on the platforms and adjust them to your own application without having to train a model from scratch.

For our project we have used the  
- [Transformers library](https://pypi.org/project/transformers/) in Python to test and combine models, and  
- [Gradio](https://www.gradio.dev/) to build a simple user interface around it.  

This means we can experiment with input and output in a visual interface, and observe the output of our models directly in an app.

## Implementation of packing list model
### Set up: Anja

Hugging Face is a company and platform for the machine learning community to collaborate on models, datasets and applications, especially in the field of natural language processing.
To be able to use the full functionality offered by Hugging Face (e.g. access to models, spaces, datasets, API access) you can create a free account on their website https://huggingface.co/.
(There is a new course at data camp, which is free for the remainder of 2025: https://huggingface.co/blog/huggingface/datacamp-ai-courses)


To develop our model, we use the Anaconda Navigator, which includes the package and environment manager conda, as well as Jupyter Notebook for writing and running Python code. You can download the Anaconda navigator from their website https://www.anaconda.com/products/navigator. (Python is installed automatically) 

Using the command line, you can create a new environment to work in and install the required packages. The following commands create a new environment called hf_env and activate it:

```bash
conda create --name hf_env
conda activate hf_env
```

Next, install the libraries used in this project and set up Jupyter Notebook.

```bash
pip install transformers torch numpy tabulate gradio pandas scikit-learn
conda install jupyter
jupyter-notebook
```
Create a new Jupyter Notebook for this project. 


### Hugging face API
Let us first try out some Hugging Face models using their API. The main advantage of using the API is that you do not need to download the models locally and all computations are handled on Hugging Face servers.

To use their API you first need to create an access token. Go to https://huggingface.co/settings/tokens and click on *+ Create new token*. Select as token type *Read* and give your token a name. 
Next, save this access token in your project folder within a .env file. Create a plain text file named .env, then add and save the following line inside it:

```text
HF_API_TOKEN=YOUR_OWN_ACCESS_TOKEN
```
, where you replace YOUR_OWN_ACCESS_TOKEN with your actual access token. 

Now it is time to start coding and try out your first zero-shot-classification model. In your Jupyter Notebook, create a code cell and enter the following Python code:

```python
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()  
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

candidate_labels = ["technology", "sports", "politics", "health"]

def query(model, input_text):
    API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
    payload = {
        "inputs": input_text,
        "parameters": {"candidate_labels": candidate_labels}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

```

In this code, we first load the necessary libraries and then the .env file. Next, we then define a set of candidate labels for our zero-shot classification model and create a query function which receives a model name and an input text and returns the model's classification. 

Trying the query function with the model "facebook/bart-large-mnli" from Hugging Face and a short input text we get the following result: 

```python
input_text = "I just bought a new laptop, and it works amazing!"
output = query("facebook/bart-large-mnli", input_text)
print(json.dumps(output, indent=4))
```

```json
[
    {
        "label": "technology",
        "score": 0.970917284488678
    },
    {
        "label": "health",
        "score": 0.014999152161180973
    },
    {
        "label": "sports",
        "score": 0.008272469975054264
    },
    {
        "label": "politics",
        "score": 0.005811101291328669
    }
]
```
The scores represent the probabilities of the text belonging to a particular class label.

This approach worked great! However, using the API the functionality is limited. We were limited to 10 candidate labels for our classification, which was not sufficient for our packing list model.


### Predefine outputs/classes: Nikky

### Model implementation: Anja

Now we load the model locally and work with additional functionality. We import the required libraries and load our class labels from a JSON file. The last code block prints out these classes, sorted into several *superclasses*. For each superclass, we will use a dedicated zero-shot classification model and therefore get a list of relevant class labels for out trip.

```python
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from transformers import pipeline

with open("packing_label_structure.json", "r") as file:
    candidate_labels = json.load(file)
keys_list = list(candidate_labels.keys())

for key in candidate_labels:
    print("\n", key, ":")
    for item in candidate_labels[key]:
        print("\t", item)
```

```text
activity_type :
	 hut trek (summer)
	 hut trek (winter)
	 camping trip (wild camping)
	 camping trip (campground)
	 ski tour / skitour
	 snowboard / splitboard trip
	 long-distance hike / thru-hike
	 digital nomad trip
	 city trip
	 road trip (car/camper)
	 festival trip
	 yoga / wellness retreat
	 micro-adventure / weekend trip
	 beach vacation
	 cultural exploration
	 nature escape

activities :
	 swimming
	 going to the beach
	 relaxing
	 sightseeing
	 biking
	 running
	 skiing
	 cross-country skiing
	 ski touring
	 hiking
	 hut-to-hut hiking
	 rock climbing
	 ice climbing
	 snowshoe hiking
	 kayaking / canoeing
	 stand-up paddleboarding (SUP)
	 snorkeling
	 scuba diving
	 surfing
	 paragliding
	 horseback riding
	 photography
	 fishing
	 rafting
	 yoga

climate_or_season :
	 cold destination / winter
	 warm destination / summer
	 variable weather / spring / autumn
	 tropical / humid
	 dry / desert-like
	 rainy climate

style_or_comfort :
	 ultralight
	 lightweight (but comfortable)
	 luxury (including evening wear)
	 minimalist

dress_code :
	 casual
	 formal (business trip)
	 conservative

accommodation :
	 indoor
	 huts with half board
	 sleeping in a tent
	 sleeping in a car

transportation :
	 own vehicle
	 no own vehicle

special_conditions :
	 off-grid / no electricity
	 self-supported (bring your own cooking gear)
	 travel with children
	 pet-friendly
	 snow and ice
	 high alpine terrain
	 snow, ice and avalanche-prone terrain
	 no special conditions to consider

trip_length_days :
	 1 day
	 2 days
	 3 days
	 4 days
	 5 days
	 6 days
	 7 days
	 7+ days
```

Next, we use the pipeline function to load the model *facebook/bart-large-mnli* from Hugging Face. After that, we pass the trip description, along with the candidate labels for the *activity_type* superclass, to the classifier and print the output as a pandas DataFrame.

```python
model_name = "facebook/bart-large-mnli"
trip_descr = "I am planning a trip to Greece with my boyfriend, where we will visit two islands. We have booked an apartment on each island for a few days and plan to spend most of our time relaxing. Our main goals are to enjoy the beach, try delicious local food, and possibly go on a hike—if it’s not too hot. We will be relying solely on public transport. We’re in our late 20s and traveling from the Netherlands."
classifier = pipeline("zero-shot-classification", model = model_name)
result = classifier(trip_descr, candidate_labels["activity_type"])

df = pd.DataFrame({
    "Label": result["labels"],
    "Score": result["scores"]
})
print(df)
```

```text
                             Label     Score
0                   beach vacation  0.376311
1   micro-adventure / weekend trip  0.350168
2                    nature escape  0.133974
3               digital nomad trip  0.031636
4             cultural exploration  0.031271
5          yoga / wellness retreat  0.012846
6                    festival trip  0.012700
7   long-distance hike / thru-hike  0.009527
8                hut trek (summer)  0.008148
9                        city trip  0.007793
10          road trip (car/camper)  0.006512
11              ski tour / skitour  0.005670
12       camping trip (campground)  0.004448
13     snowboard / splitboard trip  0.004113
14     camping trip (wild camping)  0.002714
15               hut trek (winter)  0.002170
```

The most likely activity type our model predicted is "beach vacation", which is correct! Now we will do this for every superclass and choose the most likely class label for our trip, except for the *activities* superclass. Because it is possible and likely to engaeg in more than one activity during a trip, we enable the multi_label otion within the classifier function. This means that the text can belong to more than one class. For this, each class label is evaluated independently and a probability of belonging to that class (vs not belonging) is returned. The activities that we select as our best guess are those with a probability of more than 50 percent.

```python
cut_off = 0.5
result_activ = classifier(trip_descr, candidate_labels["activities"], multi_label=True)
classes = df.loc[df["Score"] > 0.5, "Label"].tolist()

df = pd.DataFrame({
    "Label": result_activ["labels"],
    "Score": result_activ["scores"]
})
print(df)
print(classes)
```

```text
                            Label     Score
0              going to the beach  0.991486
1                        relaxing  0.977136
2                          hiking  0.942628
3                        swimming  0.219020
4                     sightseeing  0.175862
5                         running  0.098545
6               hut-to-hut hiking  0.083704
7                          biking  0.036792
8                     photography  0.036690
9                         surfing  0.030993
10  stand-up paddleboarding (SUP)  0.025300
11                     snorkeling  0.021451
12                           yoga  0.011070
13            kayaking / canoeing  0.007511
14                  rock climbing  0.006307
15                        fishing  0.003497
16                    paragliding  0.002656
17                        rafting  0.001970
18               horseback riding  0.001560
19                snowshoe hiking  0.001528
20           cross-country skiing  0.001502
21                   ice climbing  0.001434
22                         skiing  0.001169
23                   scuba diving  0.000789
24                    ski touring  0.000491
['going to the beach', 'relaxing', 'hiking']
```

We now write a function that automatically performs all predictions for each superclass based on a given trip description and try it out.

```python
def pred_trip(model_name, trip_descr, cut_off = 0.5):
    """
    Classifies trip
    
    Parameters:
    model_name: name of hugging-face model
    trip_descr: text describing the trip
    cut_off: cut_off for choosing activities

    Returns:
    pd Dataframe: with class predictions
    """
    
    classifier = pipeline("zero-shot-classification", model=model_name)
    df = pd.DataFrame(columns=['superclass', 'pred_class'])
    for i, key in enumerate(keys_list):
        print(f"\rProcessing {i + 1}/{len(keys_list)}", end="", flush=True)
        if key == 'activities':
            result = classifier(trip_descr, candidate_labels[key], multi_label=True)
            indices = [i for i, score in enumerate(result['scores']) if score > cut_off]
            classes = [result['labels'][i] for i in indices]
        else:
            result = classifier(trip_descr, candidate_labels[key])
            classes = result["labels"][0]
        df.loc[i] = [key, classes]
    return df

result = pred_trip(model_name, trip_descr, cut_off = 0.5)
print(result)
```

```text
           superclass                              pred_class
0       activity_type                          beach vacation
1          activities  [going to the beach, relaxing, hiking]
2   climate_or_season               warm destination / summer
3    style_or_comfort                              minimalist
4          dress_code                                  casual
5       accommodation                    huts with half board
6      transportation                          no own vehicle
7  special_conditions               off-grid / no electricity
8    trip_length_days                                 7+ days
```

And with that, we obtain the predicted labels for our trip description.

### Gradio App: Anja
Next, let's use the Gradio library to wrap our classification function in an interactive interface with inputs and outputs. We pass our function pred_trip, along with the input and output formats and some default values, to the gr.Interface function. 

```python
import gradio as gr

demo = gr.Interface(
    fn=pred_trip,
    inputs=[
        gr.Textbox(label="Model name", value = "facebook/bart-large-mnli"),
        gr.Textbox(label="Trip description"),
        gr.Number(label="Activity cut-off", value = 0.5),
    ],
    # outputs="dataframe",
    outputs=[gr.Dataframe(label="DataFrame")],
    title="Trip classification",
    description="Enter a text describing your trip",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()

```


![Demo of my Gradio app](./img/gradio_pred_trip.png)

The app is now ready to take your trip description and return a list of predicted class labels for your trip.


### Share your model: Anja
**Hugging Face Spaces**
A simple way to share your model with others is to use Hugging Face Spaces, where you can create a free Space that can be expanded later. Go to https://huggingface.co/spaces and click on "+ New Space", as SDK choose Gradio, as template Blank, as Space hardware choose "CPU Basic", and click on "Create Space" to create your Space.
Connected to your space is a remote git repository which is a smooth way to push your model code to the Space. Once the Space is created you will see the url of your Space and some instructions of how to set it up.

```bash
# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/spaces/<username>/<space_name>
```
As prompted, go to https://huggingface.co/settings/tokens to generate an access token. Click on *+ Create new token*, set the token type to *Write*. Give your token a name and click on *Create Token*. You will use this token as a password to push to your remote repository. 

Next, open the command line, navigate to your project folder, initialize git and connect it to the remote repository.

```bash
cd path/to/your/project
git init
git remote add origin https://huggingface.co/spaces/<username>/<space-name>
```

The Space will automatically run the model code from a file named app.py. In your project folder, create this file (e.g. on mac in command line: touch app.py) and open it. Copy all relevant code for your Gradio app into this file and save it.

```python
from transformers import pipeline
import json
import pandas as pd
import gradio as gr

with open("packing_label_structure.json", "r") as file:
    candidate_labels = json.load(file)
keys_list = list(candidate_labels.keys())

def pred_trip(model_name, trip_descr, cut_off = 0.5):
    """
    Classifies trip
    
    Parameters:
    model_name: name of hugging-face model
    trip_descr: text describing the trip
    cut_off: cut_off for choosing activities

    Returns:
    pd Dataframe: with class predictions
    """
    
    classifier = pipeline("zero-shot-classification", model=model_name)
    df = pd.DataFrame(columns=['superclass', 'pred_class'])
    for i, key in enumerate(keys_list):
        print(f"\rProcessing {i + 1}/{len(keys_list)}", end="", flush=True)
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
    fn=pred_trip,
    inputs=[
        gr.Textbox(label="Model name", value = "facebook/bart-large-mnli"),
        gr.Textbox(label="Trip description"),
        gr.Number(label="Activity cut-off", value = 0.5),
    ],
    outputs=[gr.Dataframe(label="DataFrame")],
    title="Trip classification",
    description="Enter a text describing your trip",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
```

Additionally, in your project folder, create a plain text file named requirements.txt.
This file tells Hugging Face which Python dependencies to install before running your app.
Add the following lines to your requirements.txt file and save it:

```text
transformers
torch
numpy
pandas
gradio
```

After that, add, commit, and push your changes to the remote repository.

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

Once the push is complete, go to the URL of your Space and try it out!

```text
https://huggingface.co/spaces/<username>/<space-name>
```


## Performance assessment: Anja
To evaluate the performance of different zero-shot classification models, we manually created a small test data set of 10 trip descriptions with corresponding class labels. We compared 12 of the most popular zero-shot classification Models available on Hugging Face. 

Performance was assessed in terms of accuracy (#correct classifications/#total classifications) for all superclasses, excluding the activities superclass. Since more than one type of activity can be correct for a single trip, we use the percentage of correctly identified activities (#correctly identified/#total correct activities) and the percentage of wrongly predicted activities (#falsly predicted/#total predicted activities) to asses its performance.

We then computed the average performance measures (across the test dataset) for each model and ranked them by accuracy.

```text
                                                        model  accuracy  true_ident  false_pred
0    MoritzLaurer-DeBERTa-v3-large-mnli-fever-anli-ling-wanli  0.611111    0.841667    0.546667
1                       sileod-deberta-v3-base-tasksource-nli  0.566667    0.700000    0.551667
2                MoritzLaurer-DeBERTa-v3-base-mnli-fever-anli  0.522222    0.841667    0.572381
3                 MoritzLaurer-deberta-v3-large-zeroshot-v2.0  0.500000    0.325000    0.500000
4                               valhalla-distilbart-mnli-12-1  0.500000    0.300000    0.533333
5   MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7  0.488889    0.833333    0.688373
6                          cross-encoder-nli-deberta-v3-large  0.466667    0.566667    0.541667
7                                    facebook-bart-large-mnli  0.466667    0.708333    0.400000
8                     MoritzLaurer-mDeBERTa-v3-base-mnli-xnli  0.455556    0.408333    0.481250
9                           cross-encoder-nli-deberta-v3-base  0.444444    0.533333    0.712500
10                      joeddav-bart-large-mnli-yahoo-answers  0.355556    0.650000    0.553792
11                                pongjin-roberta_with_kornli  0.233333    0.666667    0.452857
```


## Closing
* Summary
* Limitations
