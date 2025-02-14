import subprocess
import sys
import argparse
import json
import torch
import os
import toml
import warnings
import datetime
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import set_seed
from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2Processor
)

# Check if jiwer is installed
try:
    import jiwer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jiwer"])

def resize_lm_head(model, vocab_size):
    """Resize the model's language model head."""
    model.lm_head = torch.nn.Linear(model.config.hidden_size, vocab_size, bias=True)
    model.config.vocab_size = vocab_size

def setup_audio_processor(pretrained_path, vocab_dict, special_tokens):
    """Set up the audio processor components."""
    # Save vocabulary
    with open('vocab.json', 'w+') as f:
        json.dump(vocab_dict, f)

    # Create tokenizer and feature extractor
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", 
        **special_tokens,
        word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
    
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )

def create_dataloader(dataset, config, processor, sr, is_train=True):
    """Create optimized DataLoader for CPU processing."""
    # Create collate function
    default_collate = DefaultCollate(processor, sr)
    
    # Set appropriate dataloader config
    dataloader_config = config["train_dataset" if is_train else "val_dataset"]["dataloader"].copy()
    
    # Optimize for CPU
    dataloader_config.update({
        "num_workers": os.cpu_count() // 2,  # Use half of available CPU cores
        "pin_memory": False,  # Disable pin_memory for CPU
        "prefetch_factor": 2,  # Reduce prefetch for memory efficiency
        "persistent_workers": True  # Keep workers alive between iterations
    })
    
    return DataLoader(
        dataset=dataset,
        collate_fn=default_collate,
        **dataloader_config
    )

def main(config_path, resume=False, preload=None):
    # Load configuration
    config = toml.load(config_path)
    
    # Set up basic parameters
    pretrained_path = config["meta"]["pretrained_path"]
    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    
    # Create directories
    save_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/checkpoints')
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/log_dir')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
    config_path = os.path.join(config["meta"]["save_dir"], config["meta"]['name'], config_name)
    with open(config_path, 'w+') as f:
        toml.dump(config, f)

    # Set reproducibility
    set_seed(config["meta"]["seed"])
    
    # Initialize datasets
    config['val_dataset']['args']['sr'] = config['meta']['sr']
    config['train_dataset']['args']['sr'] = config['meta']['sr']
    config["train_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    config["val_dataset"]["args"]["special_tokens"] = config["special_tokens"]

    # Create training dataset
    train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    vocab_dict = train_base_ds.get_vocab_dict()
    train_ds = train_base_ds.get_data()
    
    # Create processor
    processor = setup_audio_processor(pretrained_path, vocab_dict, config["special_tokens"])
    
    # Create dataloaders
    train_dl = create_dataloader(train_ds, config, processor, config['meta']['sr'], is_train=True)
    
    # Create validation dataset and dataloader
    val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    val_ds = val_base_ds.get_data()
    val_dl = create_dataloader(val_ds, config, processor, config['meta']['sr'], is_train=False)

    # Initialize model
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    # Resize model head if needed
    vocab_size = len(processor.tokenizer)
    if model.config.vocab_size != vocab_size:
        print(f"Resizing lm_head from {model.config.vocab_size} to {vocab_size}")
        resize_lm_head(model, vocab_size)

    # Optimize model for CPU training
    model.freeze_feature_encoder()  # Freeze encoder to save memory
    if hasattr(torch, 'compile'):  # PyTorch 2.0+ optimization
        try:
            model = torch.compile(model)
            print("Successfully compiled model for CPU optimization")
        except Exception as e:
            print(f"Could not compile model: {e}")

    # Setup training components
    compute_metric = Metric(processor)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["optimizer"]["lr"]
    )
    
    steps_per_epoch = (len(train_dl) // gradient_accumulation_steps) + (len(train_dl) % gradient_accumulation_steps != 0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["scheduler"]["max_lr"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    print(f"Training set size: {len(train_ds)} utterances")
    print(f"Validation set size: {len(val_ds)} utterances")

    # Initialize trainer
    from cpu_trainer import CPUTrainer  # Import our CPU-optimized trainer
    trainer = CPUTrainer(
        config=config,
        resume=resume,
        preload=preload,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model=model,
        compute_metric=compute_metric,
        processor=processor,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        log_dir=log_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=False,
        max_clip_grad_norm=max_clip_grad_norm
    )

    # Start training
    print("Starting CPU training...")
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR Training Arguments')
    parser.add_argument('-c', '--config', required=True, type=str,
                      help='path to config file')
    parser.add_argument('-r', '--resume', action="store_true",
                      help='resume from latest checkpoint')
    parser.add_argument('-p', '--preload', default=None, type=str,
                      help='path to pretrained model')
    
    args = parser.parse_args()
    main(args.config, args.resume, args.preload)