from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import common_config as config
from ocr_dataset import OCR_Dataset, ocr_dataset_collate_fn
from model import CRNN
from decoders import ctc_decode


def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    arguments = docopt(__doc__)

    images = arguments['IMAGE']
    reload_checkpoint = arguments['-m']
    batch_size = int(arguments['-s'])
    decode_method = arguments['-d']
    beam_size = int(arguments['-b'])

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    #todo: jotform dataset
    predict_dataset = JotformDataset(paths=images,
                                      img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn = jotform_dataset_collate_fn)

    num_class = len(JotformDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                use_leaky_relu=config['use_leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(crnn, predict_loader, JotformDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
