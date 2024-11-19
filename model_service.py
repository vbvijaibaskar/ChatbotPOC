import pickle
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

current_predicted_cause_group_label = ""
current_predicted_cause_code_label = ""

def generate_cause_text(damage_text, model_path="trained-generator-model-epoch7/checkpoint-602"):
    try:
        cause_text_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        cause_text_generator = GPT2LMHeadModel.from_pretrained(model_path)
        cause_text_tokenizer.pad_token = cause_text_tokenizer.eos_token 
        
        inputs = cause_text_tokenizer(
            damage_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        
        output = cause_text_generator.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=30,      
            num_beams=5,            
            top_k=50,               
            top_p=0.95,              
            repetition_penalty=2.0,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=cause_text_tokenizer.eos_token_id
        )
        
        cause_text = cause_text_tokenizer.decode(output[0], skip_special_tokens=True)
        split_text = cause_text.split('.')
        last_part = '.'.join(split_text[2:]).strip()
        return [cause_text, last_part]

    except Exception as e:
        print(f"Error in generating cause text: {e}")
        return None

def cause_group_code_predictor(damage_group, damage_code):
    print("in here")

    try:
        with open('cause_group_model.pkl', 'rb') as file:
            cause_group_model, one_hot_encoder_dgroup, label_encoder_dcode, one_hot_encoder_cgroup = pickle.load(file)

        with open('cause_code_model.pkl', 'rb') as file:
            cause_code_model, one_hot_encoder_dgroup, label_encoder_dcode, label_encoder_ccode = pickle.load(file)    
    except Exception as e:
        print(f"Error loading models :- {e}")        

    try:
        damage_group_encoded = one_hot_encoder_dgroup.transform([[damage_group]])
        print(f"damage_group_encoded:- {damage_group_encoded}")
        damage_code_encoded = label_encoder_dcode.transform([damage_code])
        print(f"damage_code_encoded:- {damage_code_encoded}")

        input_cause_group_features = pd.DataFrame(damage_group_encoded)
        input_cause_group_features['Damage_code'] = damage_code_encoded

        predicted_cause_group = cause_group_model.predict(input_cause_group_features)
        print(f"predicted_cause_group:- {predicted_cause_group}")
        predicted_cause_group_label_index = predicted_cause_group.argmax(axis=1).reshape(-1, 1)
        print(f"predicted_cause_group_label_index:- {predicted_cause_group_label_index}")
        predicted_cause_group_label = one_hot_encoder_cgroup.inverse_transform(predicted_cause_group)
        print(f"predicted_cause_group_label:- {predicted_cause_group_label}")
        predicted_cause_group_probabilities = cause_group_model.predict_proba(input_cause_group_features)


        input_cause_code_features = pd.DataFrame(damage_group_encoded)
        input_cause_code_features['Damage_code'] = damage_code_encoded

        predicted_cause_code = cause_code_model.predict(input_cause_code_features)
        print(f"predicted_cause_code:- {predicted_cause_code}")
        predicted_cause_code_label = label_encoder_ccode.inverse_transform(predicted_cause_code)
        print(f"predicted_cause_code_label:- {predicted_cause_code_label}")
        predicted_cause_code_probabilities = cause_code_model.predict_proba(input_cause_code_features)

        global current_predicted_cause_group_label, current_predicted_cause_code_label
        current_predicted_cause_group_label = predicted_cause_group_label
        current_predicted_cause_code_label = predicted_cause_code_label

        print(predicted_cause_group_label[0][0])
        print(predicted_cause_group_probabilities.max())
        print(predicted_cause_group_probabilities.max().item())
        print(predicted_cause_code_label[0])
        print(predicted_cause_code_probabilities.max())
        print(predicted_cause_code_probabilities.max().item())

        return predicted_cause_group_label[0][0], predicted_cause_group_probabilities.max(), predicted_cause_code_label[0], predicted_cause_code_probabilities.max()
    except IndexError as e:
        print("Error: Predictions for cause group or code are out of bounds.")
    except Exception as e:
        print(f"Error in prediction process: {e}")