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

from crossmodal_alignment.retrieval_model import (
    AudioEmbeddingTorchText,
    TransformersModel,
)


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
    if audio_path.suffix == ".npy":
        return torch.from_numpy(np.load(audio_path))
    else:
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        return audio.mean(0)


@st.cache_data
def build_audio_index(root_dir: Path, _audio_encoder, pattern: str = "*.wav", **kwargs):
    file_names = []
    audios = []
    for file in tqdm(root_dir.rglob(pattern)):
        with torch.inference_mode():
            input_audio = load_audio_input(file, **kwargs)
            embedded_audio = _audio_encoder(input_audio)
        audios.append(embedded_audio)
        file_names.append(file.name)
    return torch.stack(audios), file_names


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

    # check if results.csv exists and if it does then append the new results to it
    if os.path.isfile('results.csv'):
        df.to_csv('results.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results.csv', mode='w', header=True, index=False)


def main(model, name_to_result_mapping):
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
            "Number of results returned", value=10, min_value=1, max_value=100
        )
        st.write(f"Number of items in dataset {len(ref_names)}")

    if query:

        with torch.inference_mode():
            embedded_query = model.get_text_embedding(query)
        similarities = torch.cosine_similarity(
            embedded_query, ref_audios)
        # print('similarities', similarities.shape)
        matches, match_indices = torch.topk(similarities, k=k)
        # print('matches', match_indices)
        # print('match', matches)

        batch_size = 10
        num_results = len(matches)
        num_pages = num_results // batch_size + 1

        # Number of audio results per batch
        results = []

        # Slider for page selection
        page = st.sidebar.slider("Page", 1, num_pages, 1)
        start_index = (page - 1) * batch_size
        end_index = min(page * batch_size, num_results)
        batch_results = matches[start_index:end_index]
        batch_result_indices = match_indices[start_index:end_index]

        st.header(f"{start_index+1} - {end_index } out of {k} results")
        # for batch in range(num_batches):

        #     st.subheader(f"Batch {batch+1}/{num_batches}")

        #     start_index = batch * batch_size
        #     end_index = (batch + 1) * batch_size
        #     batch_matches = matches[start_index:end_index]
        #     batch_match_indices = match_indices[start_index:end_index]

        # Shuffle the batch randomly
        # random.shuffle(batch_results)

        # geenrate a unique random number for each batch
        random_number = random.randint(1000, 9999)
        # print(random_number)

        for match, idx in zip(batch_results, batch_result_indices.tolist()):

            result_path = Path(ref_names[idx])
            st.write(f"{result_path}")
            audio_file_name = str(name_to_result_mapping(result_path))
            st.audio(audio_file_name)

            # relevance_score = st.radio('How relevant is the above audio? Rate it between 1 to 10:', list(
            #     range(1, 11)), key=slider_key)

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
                st.success("Results saved to results.csv")

        # Next and Previous buttons
        # col_prev, col_next = st.columns(2)
        # if page > 1:
        #     if col_prev.button("Previous"):
        #         print('previous button before update', page)
        #         page -= 1
        #         page = st.session_state.page
        #         print('previous button after update', page)
        # if page < num_pages:
        #     if col_next.button("Next"):
        #         print('Next button before click', page)
        #         st.session_state.page += 1
        #         page = st.session_state.page
        #         print('Next button after click', page)

        # if st.button("Next"):
        #     page += 1
        #     print('page', page)
        # if st.button("Previous"):
        #     page -= 1

        # Rerun the app to display the updated page
        if page != st.session_state.page:
            print(st.session_state.page, page)
            st.session_state.page = page


desc = """Run a retrieval interface app to test a text-to-audio search system.

The model loaded from CKPT_PATH indexes preprocessed audio files from DATA_DIR.
Query results are displayed by mapping paths of matching files to AUDIO_DIR.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "data_dir",
    help="Root directory of all audio inputs to be indexed by the model",
    type=Path,
)
parser.add_argument(
    "audio_dir",
    help="Root directory of all audio inputs used to display as results",
    type=Path,
)
parser.add_argument(
    "--ckpt_path", help="Path to a checkpoint to load the model from", type=Path
)
try:
    args = parser.parse_args()
except:
    st.code(parser.format_help())
    raise

model = load_model(args.ckpt_path)
ref_audios, ref_names = build_audio_index(
    args.data_dir, model.get_audio_embedding, sampling_rate=model.sampling_rate
)
name_to_result_mapping = partial(
    map_file_path, source_root=args.data_dir, target_root=args.audio_dir, new_ext=".wav"
)

main(model, name_to_result_mapping)
