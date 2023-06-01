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


## Structure
