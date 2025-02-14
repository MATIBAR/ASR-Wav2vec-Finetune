import sys
sys.path.append("../")
import torch
import numpy as np
from utils.feature import load_wav
from typing import Dict, List, Tuple
from functools import lru_cache
import warnings

class DefaultCollate:
    def __init__(self, processor, sr, max_audio_length: int = None) -> None:
        self.processor = processor
        self.sr = sr
        self.max_audio_length = max_audio_length
    
    def _chunk_audio(self, audio: np.ndarray, max_length: int) -> List[np.ndarray]:
        """Split long audio into chunks."""
        if len(audio) <= max_length:
            return [audio]
        
        chunks = []
        for i in range(0, len(audio), max_length):
            chunk = audio[i:i + max_length]
            if len(chunk) >= max_length // 2:  # Only keep chunks that are at least half the max length
                chunks.append(chunk)
        return chunks
    
    def __call__(self, inputs: List[Tuple]) -> Dict[str, torch.Tensor]:
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)
        
        # Handle memory-efficient audio processing
        if self.max_audio_length:
            chunked_features = []
            chunked_transcripts = []
            for feature, transcript in zip(features, transcripts):
                chunks = self._chunk_audio(feature, self.max_audio_length)
                chunked_features.extend(chunks)
                chunked_transcripts.extend([transcript] * len(chunks))
            features = chunked_features
            transcripts = chunked_transcripts
        
        try:
            # Process features with error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch = self.processor(
                    features, 
                    sampling_rate=16000, 
                    padding="longest", 
                    return_tensors="pt",
                    return_attention_mask=True
                )
            
            # Process transcripts
            with self.processor.as_target_processor():
                labels_batch = self.processor(
                    transcripts, 
                    padding="longest", 
                    return_tensors="pt"
                )
            
            # Create labels with memory-efficient masking
            batch["labels"] = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), 
                -100
            )
            
            return batch
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Return a minimal batch if possible
            return self._create_minimal_batch(features[0], transcripts[0])
    
    def _create_minimal_batch(self, feature, transcript):
        """Create a minimal batch from a single sample for error recovery."""
        try:
            batch = self.processor(
                [feature], 
                sampling_rate=16000, 
                padding=True, 
                return_tensors="pt"
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor(
                    [transcript], 
                    padding=True, 
                    return_tensors="pt"
                )
            batch["labels"] = labels_batch["input_ids"]
            return batch
        except:
            raise RuntimeError("Unable to process even a minimal batch")

class Dataset:
    def __init__(
        self, 
        data, 
        sr: int,
        preload_data: bool = False,
        transform = None,
        cache_size: int = 100,
        max_audio_length: int = None
    ):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        self.max_audio_length = max_audio_length
        self._cache_size = cache_size
        
        # Initialize LRU cache for audio loading
        if not preload_data:
            self._load_wav_cached = lru_cache(maxsize=cache_size)(self._load_wav)
    
    def _load_wav(self, path: str) -> np.ndarray:
        """Memory-efficient audio loading."""
        try:
            return load_wav(path, sr=self.sr)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
            # Return silence of 1 second as fallback
            return np.zeros(self.sr, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        item = self.data.iloc[idx]
        
        # Load audio
        if not self.preload_data:
            feature = self._load_wav_cached(item['path'])
        else:
            feature = item['transcript']
        
        # Apply transform if available
        if self.transform is not None:
            feature = self.transform(feature)
        
        # Handle long audio if needed
        if self.max_audio_length and len(feature) > self.max_audio_length:
            # Take center portion
            start = (len(feature) - self.max_audio_length) // 2
            feature = feature[start:start + self.max_audio_length]
        
        return feature, item['transcript']
    
    def cleanup_cache(self):
        """Clear the LRU cache if it exists."""
        if not self.preload_data:
            self._load_wav_cached.cache_clear()