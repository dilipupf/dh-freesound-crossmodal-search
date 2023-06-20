from crossmodal_alignment.retrieval_model import (
    AudioEmbeddingTorchText,
    TransformersModel,
)
import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchaudio
from tqdm import tqdm
import wandb
from laion_clap.training.train import evaluate
run = wandb.init(project="dh-freesound-crossmodal", entity="dilipharis")

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='runs/inference')

# start a new wandb run to track this script

# display horizatonal stars is a helper function to display a horizontal line of stars depending on the number of stars
# given as input by the user for the audio embedding retrieved


def display_horizontal_stars(num_stars):
    star_emoji = "⭐️"  # Unicode character for BLACK STAR
    horizontal_stars = "".join([star_emoji for _ in range(num_stars)])
    st.markdown(horizontal_stars)

# st.cache_data is used to cache the output of the function
# the function is only executed once and the output is cached
# load model is used to load the model from a checkpoint, if the checkpoint is None, then load the default model
# the model is set to eval mode and returned. The model is used to compute the audio embeddings


@st.cache_resource
def load_model(ckpt: str | os.PathLike | None = None):
    if ckpt is None:
        model = TransformersModel()
    else:
        model = AudioEmbeddingTorchText.load_from_checkpoint(ckpt)
    model.train(False)
    return model

# If audio path suffix is .npy, then load it as a numpy array
# else, load it as a torchaudio tensor and resample it to the sampling rate
# then take the mean of the two channels (if there are two channels)


def load_audio_input(audio_path: Path, sampling_rate: int):
    if audio_path.suffix == ".npy":
        return torch.from_numpy(np.load(audio_path))
    else:
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        return audio.mean(0)

# This function is used to build the audio index
# It takes a root directory, an audio encoder, and a pattern to match
# It returns a stack of audio embeddings and a list of file names
# The audio encoder is used to compute the embeddings
# The pattern is used to match the files in the root directory
# The stack of audio embeddings is used to build the audio index
# The list of file names is used to map the audio index to the file names


@st.cache_data
def build_audio_index(root_dir: Path, _audio_encoder, pattern: str = "*.wav", **kwargs):
    file_names = []
    audios = []
    # tqdm is used to display a progress bar
    # for each file in the root directory that matches the pattern
    # load the audio input and compute the audio embedding
    for file in tqdm(root_dir.rglob(pattern)):
        # torch.inference_mode() is used to disable gradient computation
        with torch.inference_mode():
            # load the audio input, kwargs here is used to pass the sampling rate
            input_audio = load_audio_input(file, **kwargs)
            # compute the audio embedding
            embedded_audio = _audio_encoder(input_audio)
        # append the audio embedding and the file name to the list
        audios.append(embedded_audio)
        file_names.append(file.name)
    # torch.stack is used to stack the audio embeddings into a single tensor from a list of tensors having the same shape
    return torch.stack(audios), file_names


def map_file_path(
    path_to_map: Path, source_root: Path, target_root: Path, new_ext: str | None = None
):
    # checks if path_to_map is a relative of source_root if so,
    # sets sub_path to the relative path, otherwise sets sub_path to path_to_map
    if path_to_map.is_relative_to(source_root):
        sub_path = path_to_map.relative_to(source_root)
    else:
        sub_path = path_to_map
    # new_path is the target_root joined with the sub_path
    new_path = target_root / sub_path
    # if new_ext is not None, then set the suffix of new_path to new_ext
    if new_ext:
        return new_path.with_suffix(new_ext)
    return new_path


def main(model, name_to_result_mapping):
    st.title("Cross-modal Search Demo")

    query = st.text_input(
        "Enter your search query:",
        max_chars=128,
        placeholder="e.g. a dog barks and a duck quacks",
    )
    # query strip removes leading and trailing spaces
    query = query.strip()

    # sibebar is used to display the number of results returned and update it dynamically
    with st.sidebar:
        k = st.number_input(
            "Number of results returned", value=10, min_value=1, max_value=100
        )
        st.write(f"Number of items in dataset {len(ref_names)}")

    # if query is not empty, then display the top k results
    if query:
        st.header(f"Top {k} results")
        # inference mode is used to disable gradient computation and reduce memory usage
        # it is used to wrap the computation of the query embedding
        with torch.inference_mode():
            # get_text_embedding is used to compute the query embedding
            embedded_query = model.get_text_embedding(query)

        # cosine similarity is used to compute the similarity between the query and the reference audios
        similarities = torch.cosine_similarity(embedded_query, ref_audios)

        # topk is used to get the top k similarities and their indices
        matches, match_indices = torch.topk(similarities, k=k)

        # for each match, idx in zip(matches, match_indices.tolist()) is used to iterate over the matches and their indices
        for match, idx in zip(matches, match_indices.tolist()):

            # result_path is the path of the matching audio
            result_path = Path(ref_names[idx])
            st.write(f"{result_path}")

            # name_to_result_mapping is used to map the path of the matching audio to the path of the audio in the audio directory
            st.audio(str(name_to_result_mapping(result_path)))

            # slider_key is used to uniquely identify the slider
            slider_key = f"rating_{idx}"
            # x is the value of the slider where user will rate the relevance of the matching audio
            x = st.slider(
                'How relevant is the above audio? Rate it between 1 to 5.', 1, 5, key=slider_key)
            col1 = st.columns([2, 4, 2])

            # based on the value of x, display the corresponding number of stars dynamically
            if x:
                display_horizontal_stars(x)

            # Score of the match is updated dynamically from match
            st.caption(f"Score: {match}")


desc = """Run a retrieval interface app to test a text-to-audio search system.

The model loaded from CKPT_PATH indexes preprocessed audio files from DATA_DIR.
Query results are displayed by mapping paths of matching files to AUDIO_DIR.
"""

# argparse is used to parse the command line arguments
parser = argparse.ArgumentParser(description=desc)
# add_argument is used to specify which command line arguments are expected
# data_dir is the root directory of all audio inputs to be indexed by the model
parser.add_argument(
    "data_dir",
    help="Root directory of all audio inputs to be indexed by the model",
    type=Path,
)
# audio_dir is the root directory of all audio inputs used to display as results
parser.add_argument(
    "audio_dir",
    help="Root directory of all audio inputs used to display as results",
    type=Path,
)
# ckpt_path is the path to a checkpoint to load the model from
parser.add_argument(
    "--ckpt_path", help="Path to a checkpoint to load the model from", type=Path
)
# if the command line arguments are not as expected, then display the help message
try:
    args = parser.parse_args()
except:
    st.code(parser.format_help())
    raise

# load_model is used to load the model from the checkpoint
model = load_model(args.ckpt_path)

evaluate(model, data, start_epoch, args, writer)

# build_audio_index is used to build the audio index.
# It returns the audio embeddings and the file names of the audio files
ref_audios, ref_names = build_audio_index(
    args.data_dir, model.get_audio_embedding, sampling_rate=model.sampling_rate
)
# Partial functions allow us to fix a certain number of arguments of a function and generate a new function.
# # A normal function
# def f(a, b, c, x):
#     return 1000*a + 100*b + 10*c + x

# # A partial function that calls f with
# # a as 3, b as 1 and c as 4.
# g = partial(f, 3, 1, 4)

# # Calling g()
# print(g(5))

# maps the file path to the target root
name_to_result_mapping = partial(
    map_file_path, source_root=args.data_dir, target_root=args.audio_dir, new_ext=".wav"
)

# main is used to display the UI
main(model, name_to_result_mapping)
