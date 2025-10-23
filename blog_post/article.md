# Working with Large Language Models
## Introduction: Nikky
* What are LLMs and why are they popular
* How to use the open source platform to use LLMs for own application; example of creating a packing list.

## Why bother to adapt models to your application: Nikky
* Problems of LLMS: Hallucination and wrong outputs
* Controll outputs by using zero-shot-calssification
  * briefly mention other types of classification
* How we do it with our packing list model
  * Why not use packing items as classes
  * Use superclasses to categories trip and have packing items correspond to superclasses
  * Asses performance with small test data set
  * mention gradio app to make it user friendly and spaces to share model

## Implementation of packing list model
### Prerequisites before you start to code: Anja
* Hugging face account also use access token, mention api
Hugging face is..."The platform where the machine learning community collaborates on models, datasets, and applications."(from hugging face)
"Hugging Face is a company and open-source community that builds tools for machine learning (ML) and artificial intelligence (AI) — especially in the field of natural language processing (NLP)."(from the chate)
To be able to use the full funcitonality (e.g. models, spaces, data sets, api access) and most access you need to make a hugging face account here (https://huggingface.co/)
There is a new course at data camp free for the remainder of 2025: https://huggingface.co/blog/huggingface/datacamp-ai-courses

* Use anaconda
"Without a package and environment manager like Conda"(anaconda site)
We used the anaconda navigator as a package and environment manager and used the jupyter notebook through there. python is automatically installed. 
In the terminal activate the conda environment that you created previously by navigating (see cheat sheet here https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
View list of environments
```
conda env list
```

Activate correct environment
```
conda activate ENVIRONMENT_NAME
```

Install packages using pip
```
pip install transformers torch numpy tabulate gradio pandas sklearn
# pip install -r requirements.txt
```

Install jupyter notebook
```
conda install jupyter
```

Start up jupyter notebook
```
jupyter-notebook
```
Create a new jupyter notebook

#### Hugging face API
Let us first try out some hugging face models using their API. For this you need to make an accesstoken on the hugging face website. 
Log in > Settings (on left side) > Access Tokens (on left side) > + Create new token
Give token a name
* how to implement the token in your code
This access token now has to be saved in you projecct folder in a .env file. Create a plan text file that you call .env. Within it you write:
```
HF_API_TOKEN=YOUR_OWN_ACCESS_TOKEN
```
and save it. 
  

Now we load a zero-shot-classification model using API and male a simple classification.
```
from dotenv import load_dotenv
import os
import requests

load_dotenv()  # Load environment variables from .env file, contains personal access token (HF_API_TOKEN=your_token)
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

candidate_labels = ["technology", "sports", "politics", "health"]

def query(model, input_text):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": input_text,
        "parameters": {"candidate_labels": candidate_labels}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

input_text = "I just bought a new laptop, and it works amazing!"

output = query("facebook/bart-large-mnli", input_text)
print(output)
```

# Output
Hello, world!


### Predefine outputs/classes: Nikky

### Model implementation: Anja

### Using gradio app: Anja

### Share your model: Anja
* created space
* created github access token on huffing face and then github actions workflow
* you need to install the required dependencies in the environment where your Hugging Face Space is running.
* create requirements.txt file with necessary libraries


## Performance assessment: Anja
* Test data creation


## Closing
* Summary
* Limitations
