# Introduction
This is a simple AI server that can be used to run AI models. It is built using Flask and can be used to run any AI model that can be run on a GPU.
Which is consist of StableLM and KoBart. It can be used for these tasks
1. Text Generation for Chatting
2. Text Generation for Summarization
3. Text Generation for Sentimental Recongition

**But as you can see, StableLM is for only English, so that we should use google translation libs**

## Quick Start
#### Pre Requests
* Install Python3
* pickle file for Kobert Model


1. git clone this repository
2. make venv and activate
3. install requirements
    * **But** careful, if you want to use GPU then check the torch version that you have. Because in our case, rocm is used
4. run server
```bash
python app.py
```

## File Structure

```
.
├── Readme.md
├── app.py
├── app_openai.py
├── model_210519.pickle
├── requirements.txt
└── segment_predict.py

0 directories, 6 files

```

Import Statements: The necessary libraries and modules are imported.

Model Loading: The StableLM-Tuned-Alpha-7b language model is loaded into memory using the AutoModelForCausalLM class from the transformers library. The tokenizer is also initialized.

Helper Functions:

StopOnTokens class: Implements a stopping criteria for text generation.
chat function: Generates text based on the provided conversation history and system message using the loaded language model.
Flask Application:

                +------------------------+
                |    Import Statements    |
                +------------------------+
                            |
                            |
                            v
                +------------------------+
                |      Model Loading      |
                +------------------------+
                            |
                            |
                            v
                +------------------------+
                |    Helper Functions     |
                +------------------------+
                            |
                            |
                            v
                +------------------------+
                |    Flask Application    |
                +------------------------+
                            |
               +----------------------------+
               |                            |
               v                            v
         +--------------+           +------------------+
         |    /chatbot  |           |     /model       |
         +--------------+           +------------------+
                |                            |  
                |                            |
                |                            |
                |                            |
                v                            v
                         
               +-------------------------+
               |      Main Execution     |
               +-------------------------+


## System Architecture
![image](https://github.com/Friend-for-Modern-people/Server-AI-Flask/assets/80394866/7c86028a-d932-4535-bc80-e91c0f1211b8)

Two Service Domain
* User Domain
* AI Domain

Each service implements a single business capability within a bounded context and communicates with other services through well-defined APIs
Which means it could be extends as MSA architecture

## APIs

| Route	| Method	| Request Payload| 	Response|
-----------------|-----------------|------------------------|-----------------
| /chatbot	| POST|	start_message: string<br>msg: string	| String: The chatbot's response translated into Korean. |
/model|	POST	|userId: string<br>year: string<br>month: string<br>date: string<br>start_message: string	| JSON object with the following fields:<br>summary: string (the summary of the chat history)<br>seg: string (the sentiment analysis result)
