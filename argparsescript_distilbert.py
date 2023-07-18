import sys
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import json
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle

nsynth_dataset = '/Users/omar/Documents/THESIS_CODE/nsynth-test 3/examples_updated.json'
audio_folder = '/Users/omar/Documents/THESIS_CODE/nsynth-test 3/audio'
embedding_map_path = '/Users/omar/Documents/THESIS_CODE/nsynth-test 3/embedding_map.pkl'

#decide on model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the appropriate device
model = DistilBertModel.from_pretrained(model_name).to(device)

#preprocess
def encode_texts(texts):
    input_ids = tokenizer.batch_encode_plus(texts, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)['input_ids']
    input_ids = input_ids.to(device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
    return last_hidden_states.mean(dim=1).cpu().numpy()  # flatten the output

def text_similarity(vec1, vec2):
    return distance.cosine(vec1, vec2)

with open(nsynth_dataset) as f:
    nsynth_data = json.load(f)

if len(sys.argv) < 2:
    print('Please provide a query.')
    sys.exit()

query = sys.argv[1]

query_vec = encode_texts([query])[0]

# Check if embedding map file exists, if not, create one
if not os.path.isfile(embedding_map_path):
    embedding_map = {}
    for data_id, data in nsynth_data.items():
        combined_description = data['instrument_str'] + ' ' + data['instrument_family_str'] + ' ' + ' '.join(data['qualities_str']) + ' ' + data['audio_description']
        embedding_map[data_id] = encode_texts([combined_description])[0]
    with open(embedding_map_path, 'wb') as f:
        pickle.dump(embedding_map, f)
else:
    with open(embedding_map_path, 'rb') as f:
        embedding_map = pickle.load(f)

# compute the distances
distances = {data_id: text_similarity(query_vec, vec) for data_id, vec in embedding_map.items()}
closest_data_id = min(distances, key=distances.get)
closest_data = nsynth_data[closest_data_id]

audio_file_path = os.path.join(audio_folder, closest_data_id + '.wav')
if os.path.isfile(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    sf.write('/Users/omar/Documents/THESIS_CODE/argparse_output/closest_match.wav', y, sr)
    print('Closest match:', closest_data['note_str'], distances[closest_data_id])

    fig = plt.figure(figsize=(14, 5), facecolor='none')
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.axis('off')
    plt.savefig('/Users/omar/Documents/THESIS_CODE/argparse_output/closest_match_waveform.png', transparent=True, bbox_inches='tight', pad_inches=0)
else:
    print('Audio file not found for the closest match.')
