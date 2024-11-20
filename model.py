import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pickle
import os 
import logging

logging.basicConfig(level=logging.DEBUG)

print("Current working directory:", os.getcwd())
data = pd.read_excel("chatbotpoc.XLSX")
one_hot_encoder_dgroup = OneHotEncoder(sparse_output=False)
one_hot_encoder_cgroup = OneHotEncoder(sparse_output=False)
label_encoder_dcode = LabelEncoder()
label_encoder_ccode = LabelEncoder()

def preprocess_data(data):
    damage_group_encoded = one_hot_encoder_dgroup.fit_transform(data['CdGpProb'].values.reshape(-1, 1))
    cause_group_encoded = one_hot_encoder_cgroup.fit_transform(data['CgpCse'].values.reshape(-1, 1))
    
    damage_code_encoded = label_encoder_dcode.fit_transform(data['Dam.'])
    cause_code_encoded = label_encoder_ccode.fit_transform(data['Cs.'])
    
    features = pd.DataFrame(damage_group_encoded)
    features['Damage_code'] = damage_code_encoded
    return features, cause_group_encoded, cause_code_encoded

features, cause_group_encoded, cause_code_encoded = preprocess_data(data)
# print("random row features:- ", features.sample(n=1))
# print("random row cause_group_encoded:- ", cause_group_encoded[20])
# print("random row cause_code_encoded:- ", cause_code_encoded[20])

X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
    features, cause_group_encoded, test_size=0.2, random_state=42
)

X_train_code, X_test_code, y_train_code, y_test_code = train_test_split(
    features, cause_code_encoded, test_size=0.2, random_state=42
)

try:
    cause_group_model = xgb.XGBClassifier()
    cause_group_model.fit(X_train_group, y_train_group)
    
    predicted_group = cause_group_model.predict(X_test_group)
    cause_group_accuracy_score = accuracy_score(y_test_group, predicted_group)
    print(f"cause group accuracy:- {cause_group_accuracy_score}")

    class_distribution_group = data['CgpCse'].value_counts().to_dict()
    print(f"class_distribution_group:- {class_distribution_group}") 
except Exception as e:
    print(f"Error in training cause group model: {e}")

try:
    cause_code_model = xgb.XGBClassifier()
    cause_code_model.fit(X_train_code, y_train_code)
    
    predicted_code = cause_code_model.predict(X_test_code)
    cause_code_accuracy_score = accuracy_score(y_test_code, predicted_code)
    print(f"cause code accuracy:- {cause_code_accuracy_score}")

    class_distribution_code = data['Cs.'].value_counts().to_dict() 
    print(f"class_distribution_code:- {class_distribution_code}")
except Exception as e:
    print(f"Error in training cause code model: {e}")

print("\n\n")
print("========================= GENERATION PHASE =============================")
print("\n\n")

if not (os.path.isdir("trained-generator-model-epoch7")):
    try:
        model_name = "dbmdz/german-gpt2"
        
        input_dataset = Dataset.from_pandas(data[['Item text', 'CaTxt']])
        print(f"type of input_dataset:- {type(input_dataset)}")
        print(f"Input dataset: {input_dataset[0]}")
        print(f"Input dataset size: {len(input_dataset)}")

        cause_text_tokenizer = GPT2Tokenizer.from_pretrained("trained-generator-model-epoch7/checkpoint-602")
        cause_text_tokenizer.eos_token_id = 50256
        cause_text_tokenizer.pad_token = cause_text_tokenizer.eos_token  
        print(f"Tokenizer vocab size: {cause_text_tokenizer.vocab_size}")
        print(f"Tokenizer EOS token ID: {cause_text_tokenizer.eos_token_id}")

        def tokenize_function(examples):
            combined_texts = [f"{damage} {cause}" for damage, cause in zip(examples["Item text"], examples["CaTxt"])]
            tokenized_output = cause_text_tokenizer(
                combined_texts,
                padding='max_length',  
                truncation=True,
                max_length=128  
            )
            tokenized_output["labels"] = tokenized_output["input_ids"].copy() 
            return tokenized_output

        print("Tokenizing dataset")
        tokenized_dataset = input_dataset.map(tokenize_function, batched=True)
        print(type(tokenized_dataset[0]['input_ids'][0])) 
        print("Number of items in tokenized dataset:", len(tokenized_dataset))
        print(type(tokenized_dataset))
        print(tokenized_dataset[0].keys())

        for i in range(3):  
            print("Input IDs Length:", len(tokenized_dataset[i]['input_ids']))
            print("Labels Length:", len(tokenized_dataset[i]['labels']))

        cause_text_generator = GPT2LMHeadModel.from_pretrained("trained-generator-model-epoch7/checkpoint-602")

        training_args = TrainingArguments(
            output_dir="./trained-generator-model-epoch8",
            num_train_epochs=2,                 
            per_device_train_batch_size=4,        
            gradient_accumulation_steps=8,        
            per_device_eval_batch_size=8,         
            logging_dir="./logs-generator-model-epoch8", 
            logging_steps=500,                    
            save_steps=1000,                                                
            evaluation_strategy="no",             
            weight_decay=0,                    
            warmup_steps=500,                     
            lr_scheduler_type="linear",           
            no_cuda=True,                         
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=cause_text_tokenizer,
            mlm=False, 
        )

        print("Training Model")
        trainer = Trainer(
            model=cause_text_generator,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            tokenizer=cause_text_tokenizer,
            data_collator=data_collator
        )

        trainer.train()


    except Exception as e:
        print(f"Error creating GPT2 model: {e}")

with open("cause_group_model.pkl", 'wb') as file:
        pickle.dump((cause_group_model, one_hot_encoder_dgroup, label_encoder_dcode, one_hot_encoder_cgroup), file)

with open("cause_code_model.pkl", 'wb') as file:
        pickle.dump((cause_code_model, one_hot_encoder_dgroup, label_encoder_dcode, label_encoder_ccode), file)
