import numpy as np
import librosa
import os
import soundfile as sf
import gensim
import json
import pickle
from gensim.models import KeyedVectors
from scipy.signal import resample
import sys
import matplotlib.pyplot as plt

word2vec = '/Users/omar/Documents/THESIS_CODE/GoogleNews-vectors-negative300.bin'
nsynth_dataset = '/Users/omar/Documents/THESIS_CODE/nsynth-test 3/examples_updated.json'
audio_folder = '/Users/omar/Documents/THESIS_CODE/nsynth-test 3/audio'
embedding_map_file = '/Users/omar/Documents/THESIS_CODE/embedding_map.pkl'

#extract additional features
def audio_features_to_text(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    description = []
    if np.mean(mfccs) > 0:
        description.append('warm')
    else:
        description.append('cold')

    if np.mean(chroma_stft) > 0.5:
        description.append('bright')
    else:
        description.append('dull')

    if np.mean(spectral_contrast) > 20:
        description.append('rich')
    else:
        description.append('thin')

    if np.mean(tonnetz) > 0:
        description.append('harmonic')
    else:
        description.append('inharmonic')

    return ' '.join(description)


with open(nsynth_dataset) as f:
    nsynth_data = json.load(f)

model = KeyedVectors.load_word2vec_format(word2vec, binary=True)


if len(sys.argv) < 2:
    print('Please provide a query.')
    sys.exit()

#take the command line argument as string
query = sys.argv[1]
query_vec = np.mean([model[word] for word in query.split() if word in model.key_to_index], axis=0)

# Load or calculate the embedding map
try:
    with open(embedding_map_file, 'rb') as f:
        embedding_map = pickle.load(f)
except FileNotFoundError:
    embedding_map = {}
    for data_id, data in nsynth_data.items():
        combined_description = data['instrument_str'] + ' ' + data['instrument_family_str'] + ' ' + ' '.join(data['qualities_str']) + ' ' + data['audio_description']
        words = [word for word in combined_description.split() if word in model.key_to_index]
        description_vec = np.mean([model[word] for word in words], axis=0) if words else np.zeros(model.vector_size)
        embedding_map[data_id] = description_vec
    with open(embedding_map_file, 'wb') as f:
        pickle.dump(embedding_map, f)

distances = []
for data_id in nsynth_data:
    description_vec = embedding_map[data_id]
    distance = np.linalg.norm(query_vec - description_vec)
    distances.append((distance, data_id))

distances.sort(key=lambda x: x[0])

if len(distances) < 1:
    print('No matching audio files found.')
    sys.exit()

closest_data_id = distances[0][1]
closest_data = nsynth_data[closest_data_id]

audio_file_path = os.path.join(audio_folder, closest_data_id + '.wav')
if os.path.isfile(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    sf.write('/Users/omar/Documents/THESIS_CODE/argparse_output/closest_match.wav', y, sr)
    print('Closest match:', closest_data['note_str'], distances[0][0])

    fig = plt.figure(figsize=(14, 5), facecolor='none')
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.axis('off')
    plt.savefig('/Users/omar/Documents/THESIS_CODE/argparse_output/closest_match_waveform.png', transparent=True, bbox_inches='tight', pad_inches=0)
else:
    print('Audio file not found for closest match.')
