import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
#     ['ref_audio', 'ref_duration', 'ref_audio_path', 'mix_audio', 'mix_duration', 'mix_audio_path', 'target_audio', 'target_duration', 'target_audio_path']
#     print(dataset_items)

    
    ref_durations = []
    mix_durations = []
    target_durations = []
    
    ref_audio_paths = []
    mix_audio_paths = []
    target_audio_paths = []
    
    ref_lengths = []
    mix_lengths = []
    target_lengths = []
    
    target_ids = []
    
    for item in dataset_items:
        ref_durations.append(item['ref_duration'])
        mix_durations.append(item['mix_duration'])
        target_durations.append(item['target_duration'])
        
        ref_audio_paths.append(item['ref_audio_path'])
        mix_audio_paths.append(item['mix_audio_path'])
        target_audio_paths.append(item['target_audio_path'])
        
        ref_lengths.append(min(item['ref_audio'].shape[1], 48000))
        mix_lengths.append(item['mix_audio'].shape[1])
        target_lengths.append(item['target_audio'].shape[1])
        
        target_ids.append(item['target_id'])
        
        
    
    ref_audios = torch.zeros((len(ref_durations), max(ref_lengths)))
    mix_audios = torch.zeros((len(ref_durations), max(mix_lengths)))
    target_audios =torch.zeros((len(ref_durations), max(target_lengths)))
    
    for i, item in enumerate(dataset_items):
        ref_audios[i, :ref_lengths[i]] = item['ref_audio'][:, :48000]
        mix_audios[i, :mix_lengths[i]] = item['mix_audio']
        target_audios[i, :target_lengths[i]] = item['target_audio']
    
    return {
        "ref_durations" : torch.tensor(ref_durations),
        "mix_durations" : torch.tensor(mix_durations),
        "target_durations" : torch.tensor(target_durations),
        
        "ref_audio_paths" : ref_audio_paths,
        "mix_audio_paths" : mix_audio_paths,
        "target_audio_paths" : target_audio_paths,
        
        "ref_audios" : torch.tensor(ref_audios),
        "mix_audios" : torch.tensor(mix_audios),
        "target_audios" : torch.tensor(target_audios),
        
        "ref_lengths" : torch.tensor(ref_lengths),
        "mix_lengths" : torch.tensor(mix_lengths),
        "target_lengths" : torch.tensor(target_lengths),
        
        "target_ids" : target_ids
    }