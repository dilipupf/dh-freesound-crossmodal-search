# main.py
import boto3
from decouple import config
import sqlite3
import torch
import pathlib as Path
import boto3
import torchaudio
from tqdm import tqdm
import numpy as np
import os
from smart_open import open

from crossmodal_alignment.retrieval_model import TransformersModel


# Load S3-related configurations from the .env file
aws_access_key_id = config('AWS_ACCESS_KEY_ID')
aws_secret_access_key = config('AWS_SECRET_ACCESS_KEY')
bucket_name = config('AWS_BUCKET_NAME')
bucket_region = config('AWS_DEFAULT_REGION')
database_name = config('DATABASE_NAME')


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
        for obj in tqdm(objects['Contents']):
            key = obj['Key']
            print(f"Reading file: {key}")
            s3_file_path = f's3://{bucket_name}/{key}'
            # Read the audio file content directly from S3 using smart_open
            with open(s3_file_path, 'rb') as file:
                # Modify the load_audio_input function to handle S3 file path
                input_audio = load_audio_input(file, sampling_rate)
                embedded_audio = _audio_encoder(input_audio)
            audios.append(embedded_audio)
            s3_file_names.append(s3_file_path)

    return torch.stack(audios), s3_file_names


def load_audio_input(audio_path: Path, sampling_rate: int):
    audio, sr = torchaudio.load(audio_path)
    audio = torchaudio.functional.resample(audio, sr, sampling_rate)
    return audio.mean(0)

# call main function
if __name__ == "__main__":

    model = TransformersModel()
    folder_path = 'dh-new_scapes/'

    ref_audios, ref_names = build_audio_index_s3(bucket_name, folder_path, model.get_audio_embedding,  sampling_rate=model.sampling_rate)
    # Create a dictionary to store the tensors
    embeddings = {
        'ref_audios': ref_audios,
        'ref_names': ref_names
    }

    # Save the tensor to a file using torch.save()
    torch.save(embeddings, 'embeddings.pth')