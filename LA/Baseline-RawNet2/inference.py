from model import RawNet
import argparse
import os
import yaml
import torch
from data_utils import load_sample
import tqdm
from torch.nn import functional as F
import torch
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference parser')
    parser.add_argument('--input_path', type=str,default=None, help='Change this to actual path to the audio file')
    parser.add_argument('--model_path', type=str,default=None, help='Model checkpoint')

    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    model = RawNet(parser1['model'], device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model.eval()

    input_path = args.input_path
    print(input_path)

    file_paths = os.listdir(input_path)
    final_paths = [os.path.join(input_path, i) for i in file_paths]
    result = []

    for input_path in tqdm(final_paths, desc="Processing", unit="file"):
        out_list_multi = []
        out_list_binary = []
        for m_batch in load_sample(input_path):
            m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
            logits = model(m_batch)
            multi_logits = logits
            probs = F.softmax(logits, dim=-1)
            probs_multi = F.softmax(multi_logits, dim=-1)

            out_list_multi.append(probs_multi.tolist()[0])
            out_list_binary.append(probs.tolist()[0])

        result_multi = np.average(out_list_multi, axis=0).tolist()
        result_binary = np.average(out_list_binary, axis=0).tolist()

        # print('Multi classification result : gt:{}, wavegrad:{}, diffwave:{}, parallel wave gan:{}, wavernn:{}, wavenet:{}, melgan:{}'.format(result_multi[0], result_multi[1], result_multi[2], result_multi[3], result_multi[4], result_multi[5], result_multi[6]))
        if result_binary[0] > result_binary[1]:
            result.append('FAKE') # Fake
        else:    
            result.append('REAL')

    print(result)








    
    