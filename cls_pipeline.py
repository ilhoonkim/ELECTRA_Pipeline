from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch

class cls_pipeline:
    def __init__(self):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.cls_model = ElectraForSequenceClassification.from_pretrained("/home/aift-ml/workspace/lm/KoELECTRA/finetune/ckpt/koelectra-base-v3-faq2-ckpt/checkpoint-14000")
            
    def get_inputs(self, token_a, token_b):
        if token_b:
            text_concat = str(token_a) + "[SEP]" + str(token_b)
        else:
            text_concat = str(token_a)
        inputs = self.tokenizer(text_concat, return_tensors="pt") 
        return inputs
        
    def predict_label(self, inputs):
        pt = self.cls_model(**inputs)
        res = torch.softmax(pt.logits,dim=-1)
        max = int(res.argmax())
        pred = self.cls_model.config.id2label.get(max)
        ratio = float(res[0][max].detach())
        return pred, ratio
    
    def predict(self, text_a=None, text_b=None):
        inputs = self.get_inputs(text_a, text_b)
        pred, ratio = self.predict_label(inputs)
        
        return pred, ratio
        

if __name__ == '__main__':
    result = cls_pipeline.predict("안녕하세요")
    print(result)
