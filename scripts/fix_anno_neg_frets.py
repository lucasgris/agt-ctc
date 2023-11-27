import jams
import os
import glob
from tqdm import tqdm

import interpreter as itp


def add_key_anno(jam):
    '''Add a key annotation according to the filename
    
    Parameters
    ----------
    jam : jams.JAMS
    '''
    # Build key_mode value string
    track_info_list = jam.file_metadata.title.split('_')
    key_info_list = track_info_list[1].split('-')
    tonic = key_info_list[2]
    mode = 'minor' if key_info_list[0][-1] == '2' else 'major'
    key_mode = ':'.join([tonic, mode])
    
    key_ann = jam.search(namespace='key_mode')
    if len(key_ann):
        print('key annotation already exist for file {}. len(key_ann) = {}'.format(
            os.path.basename(jam_path), len(key_ann)))
    else:
        key_anno = jams.Annotation(namespace='key_mode', duration=jam.file_metadata.duration)
        key_anno.append(time=0, duration=jam.file_metadata.duration, value=key_mode, confidence=1)
        jam.annotations.append(key_anno)

str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

def clean_bad_fret(jam):
    '''Clean the jams annotations to get rid of notes having a negative fret number.
    
    Parameters
    ----------
    jam : jams.JAMS
    
    Returns
    -------
    log : list [source, i, value, fret]
        a log of additional information
    '''
    log = []
    for source in range(6):
        anno_pair = jam.search(data_source=str(source))
        note_ann = anno_pair.search(namespace='note_midi')[0]
        pt_ann = anno_pair.search(namespace='pitch_contour')[0]
        
        per_string_output = inspect_anno_for_bad_fret(note_ann)
        bad_note_indices = [i for _, i, _, _ in per_string_output]
        clean_anno_pair(note_ann, pt_ann, bad_note_indices)
        
        log.extend(per_string_output)
    return log
    
def inspect_anno_for_bad_fret(note_anno, tolerance=0.5):
    output = []
    source = int(note_anno.annotation_metadata.data_source)
    for i, obs in enumerate(note_anno):
        if obs.value - str_midi_dict[source] + tolerance < 0:
            output.append([source, i, obs.value, obs.value - str_midi_dict[source]])
        if obs.value - str_midi_dict[source] > 20:
            output.append([source, i, obs.value, obs.value - str_midi_dict[source]])
    return output

def clean_anno_pair(note_ann, pt_ann, bad_note_indices):
    index_ann = jams.Annotation(namespace='segment_open', duration=note_ann.duration)
    note_data = note_ann.pop_data()
    pt_data = pt_ann.pop_data()
    new_idx = 0
    
    for idx, note in enumerate(note_data):
        if idx not in bad_note_indices:
            note_ann.data.add(note)
            index_ann.append(time=note.time, duration=note.duration, value=new_idx)
            new_idx += 1
            
    for obs in pt_data:
        note_idx_list = index_ann.to_samples([obs.time])[0]
        if len(note_idx_list) > 0:
            new_value = obs.value
            new_value['index'] = int(note_idx_list[0])
            pt_ann.append(time=obs.time, duration=obs.duration,
                          value=new_value)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gs_path', type=str)
    parser.add_argument('--anno_dir', type=str, default="annotation")
    args = parser.parse_args()

    gs_path = args.gs_path
    anno_dir = args.anno_dir

    print("Looking for jams at", os.path.abspath(os.path.join(gs_path, anno_dir)))
    excerpts = glob.glob(os.path.abspath(os.path.join(gs_path, anno_dir))+'/*.jams')
    print(len(excerpts))

    for jam_path in tqdm(excerpts):
        jam = jams.load(jam_path)
        add_key_anno(jam)
        clean_bad_fret(jam)
        jam.save(jam_path)