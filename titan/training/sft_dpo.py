import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

class SFTTrainer:
    """
    Supervised Fine-Tuning (SFT) Trainer.
    
    Bridges pre-training and RLHF by fine-tuning the base model on 
    high-quality instruction-response pairs. This teaches the model 
    the conversational format before alignment begins.
    
    Key features:
    - Prompt masking: Only compute loss on assistant responses
    - Mixed precision training with bf16
    - Gradient accumulation for effective batch sizes
    """
    
    def __init__(self, model, optimizer, scheduler, tokenizer,
                 gradient_accumulation_steps=4, max_grad_norm=1.0, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.global_step = 0

    def compute_loss(self, batch):
        """
        Computes causal LM loss with prompt masking.
        Labels where value == -100 are ignored (prompt tokens).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for causal prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        
        return loss

    def train_step(self, batch, step):
        """Single training step with gradient accumulation."""
        loss = self.compute_loss(batch) / self.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        return loss.item() * self.gradient_accumulation_steps

    def train_epoch(self, dataloader, epoch=0, log_interval=50):
        """Runs one SFT epoch."""
        self.model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            loss = self.train_step(batch, step)
            total_loss += loss
            
            if step % log_interval == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | Avg: {avg_loss:.4f}")
        
        return total_loss / len(dataloader)


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Trainer.
    
    A lightweight alternative to the full PPO pipeline that eliminates 
    the need for a separate reward model. Instead, DPO directly optimizes 
    the policy using preference pairs through a modified cross-entropy objective.
    
    Advantages over PPO:
    - No reward model needed (fewer GPUs)
    - More stable training (no RL instability)
    - Faster convergence
    
    Disadvantage:
    - Less expressive than PPO for complex reasoning tasks
    
    The DPO loss:
    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    
    Where:
    - π is the actor policy
    - π_ref is the frozen reference policy
    - y_w is the preferred (winning) response
    - y_l is the rejected (losing) response
    - β is the temperature controlling deviation from reference
    """
    
    def __init__(self, model, ref_model, optimizer, scheduler, 
                 beta=0.1, label_smoothing=0.0, device="cuda"):
        self.model = model
        self.ref_model = ref_model
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.device = device

    def _get_batch_logps(self, model, input_ids, attention_mask, labels):
        """
        Computes per-token log probabilities for the given sequences,
        then sums over the response tokens (where labels != -100).
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Log probs at each position
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        # Gather the log probs of the actual tokens
        per_token_logps = torch.gather(
            log_probs, 2, labels[:, 1:].unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)
        
        # Mask out prompt tokens and padding
        loss_mask = (labels[:, 1:] != -100).float()
        per_token_logps = per_token_logps * loss_mask
        
        # Sum log probs over response tokens
        return per_token_logps.sum(dim=-1)

    def compute_dpo_loss(self, chosen_batch, rejected_batch):
        """
        The core DPO objective.
        """
        # Actor log probs
        chosen_logps = self._get_batch_logps(
            self.model, chosen_batch["input_ids"].to(self.device),
            chosen_batch["attention_mask"].to(self.device),
            chosen_batch["labels"].to(self.device)
        )
        rejected_logps = self._get_batch_logps(
            self.model, rejected_batch["input_ids"].to(self.device),
            rejected_batch["attention_mask"].to(self.device),
            rejected_batch["labels"].to(self.device)
        )
        
        # Reference log probs (frozen)
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(
                self.ref_model, chosen_batch["input_ids"].to(self.device),
                chosen_batch["attention_mask"].to(self.device),
                chosen_batch["labels"].to(self.device)
            )
            ref_rejected_logps = self._get_batch_logps(
                self.ref_model, rejected_batch["input_ids"].to(self.device),
                rejected_batch["attention_mask"].to(self.device),
                rejected_batch["labels"].to(self.device)
            )
        
        # DPO Loss
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        logits = chosen_rewards - rejected_rewards
        
        if self.label_smoothing > 0:
            # Label smoothing for robustness
            loss = (
                -nn.functional.logsigmoid(logits) * (1 - self.label_smoothing)
                - nn.functional.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -nn.functional.logsigmoid(logits).mean()
        
        # Metrics
        accuracy = (logits > 0).float().mean()
        chosen_reward_mean = chosen_rewards.mean()
        rejected_reward_mean = rejected_rewards.mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        return loss, {
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_reward_mean.item(),
            "rejected_reward": rejected_reward_mean.item(),
            "reward_margin": reward_margin.item(),
        }

    def train_step(self, batch):
        """Single DPO optimization step."""
        loss, metrics = self.compute_dpo_loss(batch["chosen"], batch["rejected"])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item(), metrics
