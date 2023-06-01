from threading import Thread
from flask import Flask, request
import subprocess
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from torch.nn import functional as F
from threading import Thread
import googletrans
import json

#Segment Recognition
from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer
from torch import nn
import pickle 
from torch.utils.data import Dataset
import gluonnlp as nlp
from gluonnlp import vocab as voc
import numpy as np
from transformers import BertModel


print(f"Starting to load the model to memory")
m = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-tuned-alpha-7b").half().cuda() #half()
tok = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
generator = pipeline('text-generation', model=m, tokenizer=tok, device=0)
print(f"Sucessfully loaded the model to the memory")

# start_message = """<|SYSTEM|># StableAssistant
# - StableAssistant is A helpful and harmless Open Sources, but will refuse to do anything that could be considered harmful to the user.
# - StableAssistant is able to give information.
# - StableAssistant will refuse to participate in anything that could harm a human."""

translator = googletrans.Translator()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def chat(curr_system_message, history):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = curr_system_message + \
        "".join(["".join(["<|USER|>"+item[0], "<|ASSISTANT|>"+item[1]])
                for item in history])

    # Tokenize the messages string
    model_inputs = tok([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        tok, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=400,
        do_sample=True,
        #top_p=0.95,
       # top_k=350,
        temperature=0.5,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=m.generate, kwargs=generate_kwargs)
    t.start()

    # print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return partial_text


'''
 Sentimental Recognition
 
 From HERE
'''

# device = torch.device("cuda:0")
# tokenizer = get_tokenizer()
# tokens = tokenizer.get_vocab()
# tok=tokenizer.tokenize
# vocab = nlp.vocab.BERTVocab(tokens)

# max_len = 64
# batch_size = 64
# warmup_ratio = 0.1
# num_epochs = 1  
# max_grad_norm = 1
# log_interval = 200
# learning_rate =  5e-5

# class BERTClassifier(nn.Module):
#     def __init__(self,
#                  bert,
#                  hidden_size = 768,
#                  num_classes=60,   ##클래스 수 조정##
#                  dr_rate=None,
#                  params=None):
#         super(BERTClassifier, self).__init__()
#         self.bert = bert
#         self.dr_rate = dr_rate
                 
#         self.classifier = nn.Linear(hidden_size , num_classes)
#         if dr_rate:
#             self.dropout = nn.Dropout(p=dr_rate)
    
#     def gen_attention_mask(self, token_ids, valid_length):
#         attention_mask = torch.zeros_like(token_ids)
#         for i, v in enumerate(valid_length):
#             attention_mask[i][:v] = 1
#         return attention_mask.float()

#     def forward(self, token_ids, valid_length, segment_ids):
#         attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
#         _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
#         if self.dr_rate:
#             out = self.dropout(pooler)
#         return self.classifier(out)
    
# with open('model_210519.pickle', 'rb') as f: 
#     model = pickle.load(f)

# class BERTDataset(Dataset):
#     def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
#                  pad, pair):
   
#         transform = nlp.data.BERTSentenceTransform(
#             bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
#         self.sentences = [transform([i[sent_idx]]) for i in dataset]
#         self.labels = [np.int32(i[label_idx]) for i in dataset]

#     def __getitem__(self, i):
#         return (self.sentences[i] + (self.labels[i], ))
         
#     def __len__(self):
#         return (len(self.labels))


# def predict(predict_sentence):

#   data = [predict_sentence, '0']
#   dataset_another = [data]

#   another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
#   test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
#   model.eval()

#   for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
#     token_ids = token_ids.long().to(device)
#     segment_ids = segment_ids.long().to(device)

#     valid_length= valid_length
#     label = label.long().to(device)

#     out = model(token_ids, valid_length, segment_ids)

#     test_eval=[]

#     for i in out:
#       logits=i
#       logits = logits.detach().cpu().numpy()

#       test_eval.append("E" + str(np.argmax(logits) + 10))


#   return test_eval[0]







app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>StableLM-Tuned-Alpha-7b Chat</p>"

@app.route('/chatbot', methods=['POST'])
def run_script():
    body = request.get_json()
    answer = ""
    msg_ko_en = translator.translate(body["msg"], dest='en', src='auto')
    print(msg_ko_en.text)
    for history in chat(body["start_message"],[[msg_ko_en.text, ""]]): 
        answer = history[-1][1]
    print("Assistant > " + answer)
    answer_en_ko = translator.translate(answer, dest='ko', src="en")
    return answer_en_ko.text + "\n"

@app.route('/model', methods=['POST'])
def run_script2():
    body = request.get_json()
    answer = ""
    chattingContent = get_chat(body["userId"], body["year"], body["month"], body["date"])
    
    msg_ko_en = translator.translate("다음 문단을 요약해줘 ' " +chattingContent+" '", dest='en', src='auto')
    print(msg_ko_en.text)
    for history in chat(body["start_message"],[[msg_ko_en.text, ""]]): 
        answer = history[-1][1]
    print("Assistant > " + answer)
    answer_en_ko = translator.translate(answer, dest='ko', src="en")
    
    #sentiment
    result=subprocess.run(["python", "segment_predict.py", "--content", answer_en_ko.text], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    print("Sentiment > "+ result.stdout) # 표준 출력
    
    return {"summary":answer_en_ko.text,
            "seg":result.stdout[:-1]}

def get_chat(userId, year, month, date):
    result = ""
    resultSB = ""

    try:
        url = f"https://frimo-93773-default-rtdb.firebaseio.com/{userId}/chat.json?orderBy=\"time/date\"&equalTo={date}&print=pretty"
        response = requests.get(url)
        data = json.loads(response.text)
        #print("data ", data) 
        for key in data:
            who = data[key]["who"]
    
            message = data[key]["message"]
            if data[key]["time"]["year"] != year:
                continue
            if data[key]["time"]["month"] != month:
                continue
            if who == "Me":
                resultSB += message + ' '
                
    except Exception as e:
        print(e)
        
    return resultSB

@app.route('/seg', methods=["POST"])
def run_script3():
    # sentence = request.get_json()['msg']
    # return (predict(sentence))
    body = request.get_json()
    result=subprocess.run(["python", "segment_predict.py", "--content", body["msg"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    print(result.stdout) # 표준 출력
    return result.stdout
 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=8394)
