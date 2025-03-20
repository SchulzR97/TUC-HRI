import os
from glob import glob
from huggingface_hub import list_repo_files, login, create_commit, CommitOperationAdd
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import tarfile
import json
import shutil
from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_directory', type=str, required=True)
    parser.add_argument('--cap_mode', type=str, default='realsense', required=True)
    parser.add_argument('--cache_directory', type=str, default=None, required=False)
    parser.add_argument('--huggingface_token', type=str, default=None, required=False)
    parser.add_argument('--huggingface_repo_id', type=str, required=True)
    parser.add_argument('--upload_sequences', type=int, default=1)
    parser.add_argument('--upload_metadata', type=int, default=1)
    parser.add_argument('--include_depth_data', type=int, default=1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    phases = ['train', 'val']

    sequences_directory = Path(args.dataset_directory).joinpath('sequences', args.cap_mode)

    cache_directory = Path(args.cache_directory if args.cache_directory else 'cache')
    cache_color = cache_directory.joinpath('color')
    cache_color.mkdir(exist_ok=True, parents=True)

    if args.include_depth_data:
        cache_depth = cache_directory.joinpath('depth')
        cache_depth.mkdir(exist_ok=True, parents=True)

    token = input('Please enter your HuggingFace Token:') if not args.huggingface_token else args.huggingface_token
    login(token)

    repo_files = list_repo_files(args.huggingface_repo_id, repo_type='dataset')

    #region upload sequences
    with open(sequences_directory.joinpath('LABELS.txt'), 'r') as f:
        labels = f.read().split('\n')[:-1]

    local_sequences = sorted(glob(f'{sequences_directory}/train/*') + glob(f'{sequences_directory}/val/*'))

    SEQS_PER_PART = 1000
    part_idx = 0
    num_parts = int(np.round(len(local_sequences) / SEQS_PER_PART + 0.5))
    operations = []

    data_json = {
        'train': [],
        'val': []
    }

    prog1 = tqdm(range(num_parts), desc='Uploading parts')
    for part_idx in prog1:
        part_id = f'{part_idx:0>5}'
        prog1.set_description(part_id)

        repo_file_color = f'color/{part_id}.tar.gz'
        if args.include_depth_data:
            repo_file_depth = f'depth/{part_id}.tar.gz'
        if repo_file_color in repo_files and repo_file_depth in repo_files:
            #continue
            pass

        s_i = part_idx * SEQS_PER_PART
        e_i = (part_idx + 1) * SEQS_PER_PART if (part_idx + 1) * SEQS_PER_PART < len(local_sequences) else len(local_sequences)
        part_sequences = local_sequences[s_i:e_i]

        cache_tar_file_color = cache_color.joinpath(f'{part_id}.tar.gz')

        if args.include_depth_data:
            cache_tar_file_depth = cache_depth.joinpath(f'{part_id}.tar.gz')

        if args.upload_sequences:
            tar_color = tarfile.open(cache_tar_file_color, 'w:gz')
            if args.include_depth_data:
                tar_depth = tarfile.open(cache_tar_file_depth, 'w:gz')

        prog2 = tqdm(part_sequences, leave = False)
        for sequence in prog2:
            sequence = Path(sequence)
            prog2.set_description(sequence.name)

            phase = sequence.parent.name
            action = int(sequence.name[1:4])
            data_json[phase].append({
                'id': sequence.name,
                'action': action,
                'camera': int(sequence.name[5:8]),
                'subject': int(sequence.name[9:12]),
                'idx': int(sequence.name[15:18]),
                'label': labels[action],
                'link': f'{part_id}/{sequence.name}'
            })

            if args.upload_sequences:
                color_files = sorted(glob(f'{sequence}/*_color.jpg'))
                for color_file in color_files:
                    color_file = Path(color_file)
                    fname = color_file.name.replace('_color', '')
                    tar_color.add(color_file, f'{sequence.name}/{fname}')

                if args.include_depth_data:
                    depth_files = sorted(glob(f'{sequence}/*_depth.jpg'))
                    for depth_file in depth_files:
                        depth_file = Path(depth_file)
                        fname = depth_file.name.replace('_depth', '')
                        tar_depth.add(depth_file, f'{sequence.name}/{fname}')

        if args.upload_sequences:
            tar_color.close()

            if args.include_depth_data:
                tar_depth.close()

            operations = [CommitOperationAdd(repo_file_color, cache_tar_file_color)]
            if args.include_depth_data:
                operations.append(CommitOperationAdd(repo_file_depth, cache_tar_file_depth))

            create_commit(args.huggingface_repo_id, operations, commit_message=f'Upload part {part_id}', repo_type='dataset')
            operations = []

            os.remove(cache_tar_file_color)

            if args.include_depth_data:
                os.remove(cache_tar_file_depth)
        pass
    #endregion

    #region upload metadata
    if args.upload_metadata:
        for phase in data_json:
            with open(cache_directory.joinpath(f'{phase}.json'), 'w') as f:
                json.dump(data_json[phase], f)
            operations.append(CommitOperationAdd(f'{phase}.json', cache_directory.joinpath(f'{phase}.json')))
        
        create_commit(args.huggingface_repo_id, operations, commit_message='Add metadata', repo_type='dataset')
        operations = []
        for phase in data_json:
            if os.path.isfile(cache_directory.joinpath(f'{phase}.json')):
                os.remove(cache_directory.joinpath(f'{phase}.json'))
    #endregion
    pass