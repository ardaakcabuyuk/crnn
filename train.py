import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

#from config import train_config as config
#from dataset import OCR_Dataset, ocr_dataset_collate_fn
#from model import CRNN
#from evaluate import evaluate

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, labels, label_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    label_lengths = torch.flatten(label_lengths)

    loss = criterion(log_probs, labels, input_lengths, label_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    epochs = train_config['epochs']
    train_batch_size = train_config['train_batch_size']
    eval_batch_size = train_config['eval_batch_size']
    lr = train_config['lr']
    show_interval = train_config['show_interval']
    valid_interval = train_config['valid_interval']
    save_interval = train_config['save_interval']
    cpu_workers = train_config['cpu_workers']
    reload_checkpoint = train_config['reload_checkpoint']
    valid_max_iter = train_config['valid_max_iter']

    img_width = common_config['img_width']
    img_height = common_config['img_height']
    data_dir = common_config['data_dir']

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Running on GPU!")
    else:
        device = torch.device('cpu')
        print("Running on CPU!")

    train_dataset = OCR_Dataset(root_dir=data_dir, mode='train',
                                     img_height=img_height, img_width=img_width)

    valid_dataset = OCR_Dataset(root_dir=data_dir, mode='val',
                                     img_height=img_height, img_width=img_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=ocr_dataset_collate_fn)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        collate_fn=ocr_dataset_collate_fn)

    num_class = len(OCR_Dataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden= common_config['map_to_seq_hidden'],
                rnn_hidden= common_config['rnn_hidden'],
                use_leaky_relu= common_config['use_leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.Adam(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in tqdm(train_loader):

            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=eval_config['decode_method'],
                                      beam_size=eval_config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % save_interval == 0:
                    prefix = 'crnn'
                    loss = evaluation['loss']
                    save_model_path = os.path.join(train_config['checkpoints_dir'],
                                                   f'{prefix}_{i:06}_loss{loss}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
