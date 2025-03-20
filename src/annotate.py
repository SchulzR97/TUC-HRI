import os
import cv2 as cv
import shutil
import numpy as np
from rsp.common import drawing
import rsp.common.color as colors
from glob import glob
import argparse

action = 0
subject = 0
main_camera = 1
cameras = [1, 3, 5]

min_seq_len = 30

def load_new_frames(dir, main_camera, num_frames = 1000000):
    new_frames = []

    for i, entry in enumerate(sorted(os.listdir(dir))):
        if os.path.isdir(f'{dir}/{entry}'):
            continue
        if entry.startswith('.') or not entry.endswith('.jpg'):
            continue
        if '_depth' in entry:
            continue

        camera = entry[1:4]
        frame = entry[5:9]

        camera = int(camera)
        frame = int(frame)

        if camera != main_camera:
            continue

        new_frames.append(entry)
        if len(new_frames) >= num_frames:
            return new_frames
    return new_frames

def get_sequence_number(dir, action, camera, subject):
    seq_number = 0
    sub_dirs = ['new', 'train', 'val']
    for sub_dir in sub_dirs:
        for entry in os.listdir(f'{dir}/{sub_dir}'):
            if not os.path.isdir(f'{dir}/{sub_dir}/{entry}'):
                continue

            a = int(entry[1:4])
            c = int(entry[5:8])
            s = int(entry[9:12])
            seq = int(entry[15:18])

            if a != action or c != camera or s != subject:
                continue
            if seq >= seq_number:
                seq_number = seq + 1
    return seq_number

def get_action_labels(dir):
    label_file = f'{dir}/LABELS.txt'
    if not os.path.isfile(label_file):
        labels = ['None', 'Waving', 'Pointing', 'Clapping', 'Follow', 'Walking', 'Stop', 'Turn', 'Jumping', 'Come here', 'Calm']
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f'{label}\n')

    with open(label_file, 'r') as f:
        labels = f.readlines()
    for i in range(len(labels)):
        labels[i] = labels[i].replace('\n', '')
    return labels

def get_subjects(dir):
    subject_file = f'{dir}/SUBJECTS.txt'

    with open(subject_file, 'r') as f:
        subjects = f.readlines()
    for i in range(len(subjects)):
        subjects[i] = subjects[i].replace('\n', '')
    return subjects

def get_args():
    parser = argparse.ArgumentParser(description='Annotate sequences')
    parser.add_argument('--dataset_directory', type=str, default=None, help='Directory of the dataset')
    parser.add_argument('--cap_mode', type=str, default='realsense', help='Capture mode')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    base_dir = '' if args.dataset_directory is None else args.dataset_directory + '/'
    base_dir += f'sequences/{args.cap_mode}'
    new_dir = f'{base_dir}/new'

    new_frames = load_new_frames(new_dir, main_camera)
    action_labels = get_action_labels(base_dir)
    subjects = get_subjects(base_dir)

    seq_start_idx = 0

    camera = 0
    seq_number = get_sequence_number(base_dir, action, camera, subject)

    i = 0
    while True:
        imgs_color = []
        imgs_depth = []

        entry = new_frames[i]

        camera = int(entry[1:4])
        frame = int(entry[5:10])

        #seq_number = get_sequence_number(base_dir, action, camera, subject)

        for c in cameras:
            fname_color = f'{new_dir}/C{c:0>3}F{frame:0>5}_color.jpg'
            fname_depth = f'{new_dir}/C{c:0>3}F{frame:0>5}_depth.jpg'

            img = cv.imread(fname_color)

            seq_name = f'A{action:0>3}C{c:0>3}S{subject:0>3}SEQ{seq_number:0>3} - {action_labels[action]}'
            img = cv.putText(img, f'{seq_name} - F{frame:0>5}', (10, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))
            img = cv.putText(img, f'seq_start_idx {seq_start_idx}', (10, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))
            img = cv.putText(img, f'seq_len {i-seq_start_idx}', (10, 70), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))
            img = cv.putText(img, f'S{subject:0>3} - {subjects[subject]}', (10, 90), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))

            if img is None:
                img = np.ones((500, 1000))
                img = cv.putText(img, f'Image {fname_color} not found.', (10, 30), fontScale=0.6, fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(0,0,0))
            
            cv.imshow(f'C{c:0>3} color', img)

        img_info = np.full((200, 300, 3), 255, dtype=np.uint8)

        info_txt = f'- ...previous frame\n' +\
                    f'+ ...next frame\n' +\
                    f'enter ...save sequence\n' +\
                    f'n ...start new sequence\n' +\
                    f'a ...select action\n' +\
                    f's ...select subject'
        img_info = drawing.add_text(img_info, info_txt, (10, 10), scale=0.5, foreground=colors.BLACK, text_thickness=1, width=img_info.shape[1], height=img_info.shape[0])
        cv.imshow('Info', img_info)

        key = cv.waitKey()

        if key != -1:
            pass
        if key == 13:   # enter
            if i - seq_start_idx >= min_seq_len:
                num_sequences = int(np.round((i + 1 - seq_start_idx) / min_seq_len - 0.5))
                seq_len = int(np.round((i - seq_start_idx) / num_sequences))

                for s in range(num_sequences):
                    s_start = seq_start_idx + s * seq_len
                    s_end = s_start + seq_len
                    if s == num_sequences - 1:
                        s_end = i
                    
                    for f in range(s_start, s_end):
                        in_frame = new_frames[f]
                        frame_number = int(in_frame[5:10])
                        for c in cameras:                            
                            seq_name = f'A{action:0>3}C{c:0>3}S{subject:0>3}SEQ{seq_number+s:0>3}'
                            if not os.path.isdir(f'{new_dir}/{seq_name}'):
                                os.mkdir(f'{new_dir}/{seq_name}')
                            in_file_color = f'{new_dir}/C{c:0>3}F{frame_number:0>5}_color.jpg'
                            in_file_depth = f'{new_dir}/C{c:0>3}F{frame_number:0>5}_depth.jpg'

                            out_file_color = f'{new_dir}/{seq_name}/C{c:0>3}F{frame_number:0>5}_color.jpg'
                            out_file_depth = f'{new_dir}/{seq_name}/C{c:0>3}F{frame_number:0>5}_depth.jpg'
                            
                            shutil.move(in_file_color, out_file_color)
                            shutil.move(in_file_depth, out_file_depth)
                            pass

                seq_start_idx = i
                pass
            seq_number = get_sequence_number(base_dir, action, camera, subject)

        # new sequence
        if key == ord('n'):
            seq_start_idx = i
            seq_number = get_sequence_number(base_dir, action, camera, subject)

        # choose action
        if key == ord('a'):
            if action < len(action_labels) - 1:
                action += 1
            else:
                action = 0
            seq_number = get_sequence_number(base_dir, action, camera, subject)

        # choose subject
        if key == ord('s'):
            if subject < len(subjects) - 1:
                subject += 1
            else:
                subject = 0
            seq_number = get_sequence_number(base_dir, action, camera, subject)
        
        # coose frame
        if key == ord('-'):
            if i > 0:
                i -= 1
            else:
                i = len(new_frames) - 1
            pass
        if key == ord('+'):
            if i < len(new_frames) - 1:
                i += 1
            else:
                i = 0
            pass
        
    pass
