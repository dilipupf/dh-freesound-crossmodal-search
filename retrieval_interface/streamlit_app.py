import argparse
import os
from functools import partial
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torchaudio
from tqdm import tqdm
import random
import pandas as pd
from decouple import config
from dotenv import load_dotenv
import os
import boto3

from smart_open import open

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

from crossmodal_alignment.retrieval_model import (
    AudioEmbeddingTorchText,
    TransformersModel,
)

# aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
# aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# bucket_name = os.getenv('AWS_BUCKET_NAME')
# bucket_region = os.getenv('AWS_DEFAULT_REGION')
# database_name = os.getenv('DATABASE_NAME')

aws_access_key_id = st.secrets['AWS_ACCESS_KEY_ID']  
aws_secret_access_key = st.secrets['AWS_SECRET_ACCESS_KEY']
bucket_name = st.secrets['AWS_BUCKET_NAME']
bucket_region = st.secrets['AWS_DEFAULT_REGION']
database_name = st.secrets['DATABASE_NAME']

def display_horizontal_stars(num_stars):
    star_emoji = "⭐️"  # Unicode character for BLACK STAR
    horizontal_stars = "".join([star_emoji for _ in range(num_stars)])
    st.markdown(horizontal_stars)


@st.cache_resource
def load_model(ckpt: str | os.PathLike | None = None):
    if ckpt is None:
        model = TransformersModel()
    else:
        model = AudioEmbeddingTorchText.load_from_checkpoint(ckpt)
    model.train(False)
    return model


def load_audio_input(audio_path: Path, sampling_rate: int):
    # if audio_path.suffix == ".npy":
    #     return torch.from_numpy(np.load(audio_path))
    # else:
    audio, sr = torchaudio.load(audio_path)
    audio = torchaudio.functional.resample(audio, sr, sampling_rate)
    return audio.mean(0)


@st.cache_data
def build_audio_index(root_dir: Path, _audio_encoder, pattern: str = "*.wav", **kwargs):
    file_names = []
    audios = []
    limit = 10
    for index, file in enumerate(tqdm(root_dir.rglob(pattern))):
        if index > limit:
            break

        with torch.inference_mode():
            input_audio = load_audio_input(file, **kwargs)
            embedded_audio = _audio_encoder(input_audio)
        audios.append(embedded_audio)
        file_names.append(file.name)
    return torch.stack(audios), file_names



@st.cache_data
def build_audio_index_s3(bucket_name, folder_path, _audio_encoder, sampling_rate):

 
    # Initialize the S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=bucket_region)

    # List objects (files) in the specified S3 folder with the given pattern
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    

    s3_file_names = []
    audios = []

    if 'Contents' in objects:
        for obj in objects['Contents'][:20]:
            key = obj['Key']
            # print(f"Reading file: {key}")
            s3_file_path = f's3://{bucket_name}/{key}'
            # Read the audio file content directly from S3 using smart_open
            with open(s3_file_path, 'rb') as file:
                # Modify the load_audio_input function to handle S3 file path
                input_audio = load_audio_input(file, sampling_rate)
                embedded_audio = _audio_encoder(input_audio)
            audios.append(embedded_audio)
            s3_file_names.append(s3_file_path)

    return torch.stack(audios), s3_file_names




def map_file_path(
    path_to_map: Path, source_root: Path, target_root: Path, new_ext: str | None = None
):
    if path_to_map.is_relative_to(source_root):
        sub_path = path_to_map.relative_to(source_root)
    else:
        sub_path = path_to_map
    new_path = target_root / sub_path
    if new_ext:
        return new_path.with_suffix(new_ext)
    return new_path


def save_results_to_csv(results):
    df = pd.DataFrame(results, columns=['query', 'batch_index',
                      'index_of_audio_output_tensor)', 'audio_file_name', 'similarity_score_by_model', 'user_relevance_score'])

    # check if results_fgbg.csv exists and if it does then append the new results to it
    if os.path.isfile('results_fgbg_sam.csv'):
        df.to_csv('results_fgbg_sam.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results_fgbg_sam.csv', mode='w', header=True, index=False)


def main(model):
    st.title("Cross-modal Search Demo")

    query = st.text_input(
        "Enter your search query:",
        max_chars=128,
        placeholder="e.g. a dog barks and a duck quacks",
    )
    query = query.strip()

    if 'page' not in st.session_state:
        st.session_state.page = 1

    with st.sidebar:
        k = st.number_input(
            "Number of results returned", value=20, min_value=1, max_value=100
        )
        st.write(f"Number of items in dataset {len(ref_names)}")

    if query:

        with torch.inference_mode():
            embedded_query = model.get_text_embedding(query)
        similarities = torch.cosine_similarity(
            embedded_query, ref_audios)
        # print('similarities', similarities.shape)
        topk_values, topk_indices = torch.topk(similarities, k=k)
        # print('topk_values', topk_values)
        # print('topk_indices', topk_indices)

        # Set the seed for reproducibility

        random_with_seed = random.Random(123)
        # Convert indices to a list and shuffle them randomly
        shuffled_indices = topk_indices.tolist()
        random_with_seed.shuffle(shuffled_indices)

        print('shuffled_indices', shuffled_indices)
        # Retrieve the elements from the original tensor using shuffled indices
        shuffled_values = similarities[shuffled_indices]

        batch_size = 10
        num_results = len(topk_values)
        num_pages = num_results // batch_size + 1

        # Number of audio results per batch
        results = []

        # Slider for page selection
        page = st.sidebar.slider("Page", 1, num_pages, 1)
        start_index = (page - 1) * batch_size
        end_index = min(page * batch_size, num_results)
        batch_results = shuffled_values[start_index:end_index]
        batch_result_indices = shuffled_indices[start_index:end_index]

        st.header(f"{start_index+1} - {end_index } out of {k} results")

        # geenrate a unique random number for each batch

        random_number = random.randint(1000, 9999)
        # print(random_number)

        for match, idx in zip(batch_results, batch_result_indices):

            s3_audio_file_path = ref_names[idx]
            # st.write(f"{result_path}")
            # audio_file_name = str(name_to_result_mapping(result_path))
            with open(s3_audio_file_path, 'rb') as file:
                # Modify the load_audio_input function to handle S3 file path
                input_audio = file.read()
            
            st.audio(input_audio, format='audio/wav')

            audio_file_name = s3_audio_file_path.split('/')[-1]

            slider_key = f"rating_{idx}"
            # Display the radio buttons for rating

            relevance_score = st.radio(f'How relevant is the above audio? Rate it between 1 to 10:', list(
                range(1, 11)), key=slider_key, horizontal=True)

            st.columns([2, 4, 2])

            # based on the value of x, display the corresponding number of stars dynamically
            if relevance_score:
                display_horizontal_stars(relevance_score)

            # st.caption(f"Score: {match}")
            results.append(
                [query, random_number, idx, audio_file_name, round(match.item(), 4), relevance_score])

        # Add a save button
        if st.button(f"Save Results"):
            if len(results) > 0:
                save_results_to_csv(results)
                st.success("Results saved to results_fgbg_sam.csv")


        # Rerun the app to display the updated page
        if page != st.session_state.page:
            print(st.session_state.page, page)
            st.session_state.page = page


desc = """Run a retrieval interface app to test a text-to-audio search system.

The model loaded from CKPT_PATH indexes preprocessed audio files from DATA_DIR.
Query results are displayed by mapping paths of matching files to AUDIO_DIR.
"""

parser = argparse.ArgumentParser(description=desc)
# parser.add_argument(
#     "data_dir",
#     help="Root directory of all audio inputs to be indexed by the model",
#     type=Path,
# )
# parser.add_argument(
#     "audio_dir",
#     help="Root directory of all audio inputs used to display as results",
#     type=Path,
# )
parser.add_argument(
    "--ckpt_path", help="Path to a checkpoint to load the model from", type=Path
)
try:
    args = parser.parse_args()
except:
    st.code(parser.format_help())
    raise
# data_direc = Path('data/audio/dh-new_scapes')
# audio_direc = Path('data/audio/dh-new_scapes')

model = load_model(args.ckpt_path)
# ref_audios, ref_names = build_audio_index(
#     data_direc, model.get_audio_embedding, sampling_rate=model.sampling_rate
# )



# name_to_result_mapping = partial(
#     map_file_path, source_root= data_direc, target_root=audio_direc, new_ext=".wav"
# )





folder_path = 'dh-new_scapes/'

ref_audios, ref_names = build_audio_index_s3(bucket_name, folder_path, model.get_audio_embedding,  sampling_rate=model.sampling_rate)
main(model)