import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import classification_report
from seqeval.metrics import classification_report, f1_score

LEARNING_RATE = 5e-5
EPOCHS = 1
BATCH_SIZE = 128

def train_loop(model, train_dataset, val_dataset, dict):
    

    writer = SummaryWriter()
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE, drop_last=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    count=1 
    for epoch_num in range(EPOCHS):      
        
        print("epoch ", epoch_num+1, "/", EPOCHS)
    
        model.train()     
        loss_scalar=0
        tqdm_data=tqdm(train_dataloader)
        for train_data, train_label in tqdm_data:      # iterate on batches of dataloaders
            
            tqdm_data.set_postfix(loss=loss_scalar, refresh=False)
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=train_label, return_dict=False)
            
            y_train_true=[]
            y_train_pred=[]

            for i in range(BATCH_SIZE):                # iterate on elements of a single batch (batch_size=logits.shape[0])
                
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                y_train_true.append(label_clean.tolist())  
                y_train_pred.append(predictions.tolist())
                            
            if count%10==0: 
                writer.add_scalar('batch_loss/train', loss.item(), count)

            loss_scalar=loss.item()
            loss.backward()
            optimizer.step()
            count+=1
        
        model.eval()

        y_val_true=[]
        y_val_pred=[]
        
        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=val_label, return_dict=False)

            for i in range(logits.shape[0]):

                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]
                
                predictions = logits_clean.argmax(dim=1)
                y_val_true.append(label_clean.tolist())  
                y_val_pred.append(predictions.tolist())
         
        
        true=[dict.transform_ids_to_labels(sent) for sent in y_val_true]
        pred=[dict.transform_ids_to_labels(sent) for sent in y_val_pred]
        
        score = f1_score(true, pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(true, pred))

    writer.flush()
    writer.close()

