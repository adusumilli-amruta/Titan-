import torch
import torch.nn as nn
from transformers import PreTrainedModel

class RewardModel(nn.Module):
    """
    Appends a linear value head to the base Titan causal language model.
    Used for RLHF preference ranking (Bradley-Terry models).
    Instead of predicting the next token, it outputs a scalar 'reward' or 'score' 
    for the sequence's desirability (e.g., how helpful/harmless it is).
    """
    def __init__(self, config, base_model: PreTrainedModel):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Scalar value head attached to the final hidden dimension
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = config.pad_token_id if hasattr(config, "pad_token_id") else 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        # Grab the last hidden state of shape: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1] 

        # Generate per-token reward expectations: [batch_size, seq_len, 1]
        values = self.v_head(last_hidden_state).squeeze(-1)

        # We only care about the reward for the final token in the response 
        # (excluding padding). We find the index of the last non-pad token.
        
        # Example mask logic: [1, 1, 1, 0, 0] -> sequence length is 3. 
        # The reward value is derived from the representation at index 2.
        last_non_pad_idx = (attention_mask.cumsum(dim=1) - 1).max(dim=1).values.long()
        
        # Gather the final aggregate scalar rewards for the batch
        batch_size = input_ids.shape[0]
        sequence_rewards = values[torch.arange(batch_size, device=values.device), last_non_pad_idx]

        return values, sequence_rewards

class RewardTrainer:
    """
    Trains the Bradley-Terry Reward Model based on pairwise preference data.
    Takes a 'chosen' response and a 'rejected' response.
    Model is penalized if the score of the rejected response > chosen response.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def compute_loss(self, chosen_inputs, rejected_inputs):
        # 1. Forward pass chosen sequences
        _, chosen_rewards = self.model(**chosen_inputs)
        
        # 2. Forward pass rejected sequences
        _, rejected_rewards = self.model(**rejected_inputs)

        # 3. Bradley-Terry Loss (log-sigmoid objective)
        # We want chosen_reward - rejected_reward to be a large positive number.
        # -log(sigmoid(positive)) approaches 0. -log(sigmoid(negative)) approaches infinity.
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # 4. Accuracy Tracking (how often the model correctly ranked chosen > rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, accuracy

    def train_step(self, batch):
        """Executes a single preference learning gradient step."""
        loss, acc = self.compute_loss(batch["chosen"], batch["rejected"])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), acc.item()
