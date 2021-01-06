#   ###################################################################
#   imports
#   ###################################################################
import torch
import pandas as pd
import numpy as np
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

#   ###################################################################
#   loading preprocessed dataset
#   ###################################################################
df = pd.read_csv("Data/artificial_ground_truth.csv",
                 sep=',',
                 # names=['_id', 'label', 'id', 'tweet_text', 'text_preprocessed', 'tokens_preprocessed'],
                 names=['_id', 'id', 'text_preprocessed', 'tweet_date', 'location',
                        'textblob_preprocessed_label', 'vader_preprocessed_label'],
                 # header=None,  # no header, alternative header = header_col
                 index_col=None,  # no index, alternative header = index_row
                 skiprows=0  # how many rows to skip / not include in read_csv
                 )

df = df.drop(['_id', 'location', 'textblob_preprocessed_label', 'tweet_date'], axis=1)
df = df.rename(columns={'vader_preprocessed_label': 'label'})
df.set_index('id', inplace=True)

print(df.head())
print(df.text_preprocessed.iloc[0])
print(df.label.value_counts())

label_dict = {
    'neutral': 0,
    'positive': 1,
    'negative': 2
}
print(len(label_dict))
#   ###################################################################
#   split train/val
#   ###################################################################
X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.2,
    random_state=42,
    stratify=df.label.values
)
print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))

#   ###################################################################
#   check if both classes have OK representation in val and add 'data_type'
#   ###################################################################
df['data_type'] = ['not_set'] * df.shape[0]
print(df.head())
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
df.groupby(['label', 'data_type']).count()
print(df.groupby(['label', 'data_type']).count())
# # # sentiment140
# #                    text_preprocessed
# # # label data_type
# # # 0     train                 676701
# # #       val                   119680
# # # 1     train                 675856
# # #       val                   119513
#                  text_preprocessed
# # # vaccine_db
# # # label data_type
# # #  0    train                  17842
# # #       val                     4460
# # #  1    train                  34973
# # #       val                     8744
# # #  2    train                   9096
# # #       val                     2274

#   ###################################################################
#   use tokenizer (BERT) -> convert text data to numerical data
#   ###################################################################
print("Running tokenizer")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True,
)
#
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].text_preprocessed.values,
    add_special_tokens=True,
    return_attention_mask=True,
    # pad_to_max_length=True,
    max_length=256,
    return_tensors='pt',
    truncation=True,
    padding=True  # padding=longest
)
#
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].text_preprocessed.values,
    add_special_tokens=True,
    return_attention_mask=True,
    # pad_to_max_length=True,
    max_length=256,
    return_tensors='pt',
    truncation=True,
    padding=True  # padding=longest
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val, labels_val)

print(len(dataset_train))
print(len(dataset_val))

#   ###################################################################
#   setting up BERT pretrained model
#   ###################################################################
#   takes time to dl! :)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)

#   ###################################################################
#   creating data loaders
#   ###################################################################
batch_size_train = 4  # 32  # 4 is small batch size, but resources here are limited! :)_
batch_size_val = 32  # validation data is a lot less, so it can handle 32

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size_train
)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=batch_size_val
)

#   ###################################################################
#   setting up Optimizer (defining learning rate)
#   ###################################################################
optimizer = AdamW(
    model.parameters(),
    lr=1e-5,  # between 2e-5 to 5e-5, based on original BERT paper
    eps=1e-8
)  # Adam Learning Rate

#   ###################################################################
#   setting up Scheduler
#   ###################################################################
epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train) * epochs
)


#   ###################################################################
#   F1
#   ###################################################################
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')  # ='weighted', in case of imbalanced


#   ###################################################################
#   Accuracy
#   ###################################################################
def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}')


#   ###################################################################
#   seed everything! >:D
#   ###################################################################
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#   ###################################################################
#   if you can 'CUDA', cud' it! Else.. cpu is also fine!
#   ###################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


#   ###################################################################
#   evaluate method
#   ###################################################################
def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


#   ###################################################################
#   training loop
#   ###################################################################
for epoch in tqdm(range(1, epochs + 1)):
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(
            {'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
    torch.save(model.state_dict(), f'Models/BERT_fine_tune_{epoch}.model')
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Trainig loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')

#   ###################################################################
#   loading and evaluating model!
#   ###################################################################
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)
pass

model.load_state_dict(
    torch.load('Models/finetuned_bert_epoch_1_gpu_trained.model',
               map_location=torch.device('cpu'))
)

_, predictions, true_vals = evaluate(dataloader_val)

accuracy_per_class(predictions, true_vals)
