from typing import Any, Dict, Union
import torch
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from logger.pbar import PBar

class CPUTrainer(BaseTrainer):
    def __init__(self, 
                config,
                resume,
                preload,
                epochs,
                steps_per_epoch,
                model,
                compute_metric,
                processor,
                train_dl,
                val_dl,
                optimizer,
                scheduler,
                save_dir,
                log_dir,
                gradient_accumulation_steps,
                use_amp,
                max_clip_grad_norm
                ):
        super().__init__(
                        config=config,
                        resume=resume, 
                        preload=preload, 
                        epochs=epochs, 
                        steps_per_epoch=steps_per_epoch,
                        model=model, 
                        processor=processor,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        optimizer=optimizer, 
                        scheduler=scheduler,
                        save_dir=save_dir, 
                        log_dir=log_dir,
                        use_amp=False,  # Force disable AMP for CPU
                        gradient_accumulation_steps=gradient_accumulation_steps
                        )
        self.compute_metric = compute_metric
        self.sr = config["meta"]["sr"]
        self.max_clip_grad_norm = max_clip_grad_norm
        self.stateful_metrics = ["train_loss", "train_lr", "train_grad_norm", "train_wer", "val_loss", "val_wer"]
        
        # Enable memory efficient options if specified in config
        if config["trainer"].get("args", {}).get("enable_memory_efficient", False):
            self.enable_memory_optimizations()
            
        # Enable gradient checkpointing if specified
        if config["trainer"].get("args", {}).get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

    def enable_memory_optimizations(self):
        """Enable various memory optimization techniques"""
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'compile'):  # PyTorch 2.0+ optimization
            self.model = torch.compile(self.model)

    def get_grad_norm(self, params) -> torch.tensor:
        """Compute grad norm given a gradient scale."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def _train_epoch(self, epoch) -> None:
        print(f"Epoch {epoch + 1}: ")
        pbar = PBar(self.steps_per_epoch, 10, stateful_metrics=self.stateful_metrics)

        if self.resume_step >= 0:
            print("*****Load previous time steps******")
            resume_pbar = tqdm(total=self.resume_step+1)

        for dl_step, batch in enumerate(self.train_dl):
            if self.resume_step >= 0:
                self.resume_step -= 1
                resume_pbar.update()
                if self.resume_step < 0:
                    resume_pbar.close()
                continue

            # Forward pass
            self.model.train()
            outputs = self.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            wer = torch.tensor(self.compute_metric(outputs.logits.detach(), batch['labels']))

            # Optimize step
            if (dl_step + 1) % self.gradient_accumulation_steps == 0 or dl_step == len(self.train_dl) - 1:
                # Compute grad norm for monitoring
                grad_norm = self.get_grad_norm(self.model.parameters())

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # Logging
                train_logs = {
                    "loss": loss * self.gradient_accumulation_steps,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "grad_norm": grad_norm,
                    "wer": wer
                }
                train_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in train_logs.items()}

                # Write train logs
                self.writer.update(self.completed_steps, 'Train', train_logs)
                pbar.update(self.pbar_step+1, "train_", train_logs)

                # Evaluation
                if (self.completed_steps+1) % self.validation_interval == 0:
                    print("\nValidation is in progress...")
                    self.model.eval()
                    val_logs = self._valid_epoch(self.completed_steps)
                    
                    # Write val logs
                    self.writer.update(self.completed_steps, 'Validation', val_logs)
                    pbar.update(self.pbar_step+1, "val_", val_logs)

                    # Save best
                    if self._is_best_epoch(val_logs['wer'], save_max_metric_score=self.save_max_metric_score):
                        self._save_checkpoint(epoch, dl_step, is_best_epoch=True)
                    else:
                        self._save_checkpoint(epoch, dl_step, is_best_epoch=False)

                self.pbar_step += 1
                self.completed_steps += 1

                # Memory management
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()

        # Reset
        self.pbar_step = 0
            
    def _valid_epoch(self, step) -> Dict[str, Union[Any, float]]:
        # Init logs
        val_logs = {
            "loss": 0,
            "wer": 0
        }

        for batch in tqdm(self.val_dl, total=len(self.val_dl)):
            with torch.no_grad():
                outputs = self.model(**batch)

            val_logs["loss"] += outputs.loss / len(self.val_dl)
            val_logs["wer"] += torch.tensor(self.compute_metric(outputs.logits, batch['labels'])) / len(self.val_dl)

        val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
        return val_logs