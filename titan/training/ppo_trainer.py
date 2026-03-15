import torch
from transformers import PreTrainedModel

class PPOTrainer:
    """
    Implements Proximal Policy Optimization (PPO) for LLM Alignment.
    
    The architecture requires 4 distinct model instances in VRAM:
    1. Actor Model (The LLM being trained / generating responses)
    2. Reference Model (Frozen copy of Actor before RLHF starts - for KL penalties)
    3. Reward Model (Frozen ranking model trained previously)
    4. Value Model (Trains alongside Actor to predict expected reward, can share base model with Reward Model)
    """
    def __init__(self, actor_model, ref_model, reward_model, value_model, optimizer):
        self.actor_model = actor_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        
        self.optimizer = optimizer
        self.kl_coef = 0.1
        self.gamma = 1.0     # Discount factor (often 1.0 in RLHF text gen)
        self.lam = 0.95      # GAE parameter
        self.clip_range = 0.2
        self.vf_coef = 0.1   # Value loss coefficient

    @torch.no_grad()
    def compute_rlhf_rewards_and_advantages(self, input_ids, mask):
        """
        1. Pass generations to the frozen Reward Model to get standard scores.
        2. Calculate the KL divergence between Actor and Reference distributions.
        3. Subtract KL from base reward to force the Actor not to deviate too far.
        4. Calculate Generalized Advantage Estimation (GAE).
        """
        # Step 1: Base Reward (from the final token)
        _, final_reward = self.reward_model(input_ids, attention_mask=mask)
        
        # Step 2: Distribution differences (KL Penalties)
        actor_logits = self.actor_model(input_ids).logits
        ref_logits = self.ref_model(input_ids).logits
        
        actor_logprobs = torch.log_softmax(actor_logits, dim=-1)
        ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
        
        # Gather probabilities of the actual sequence generated
        actor_logprobs = torch.gather(actor_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = torch.gather(ref_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Simplified KL (approx): log(P_actor) - log(P_ref)
        kl_penalty = (actor_logprobs - ref_logprobs)

        # Step 3: Total Reward Vector
        # Reward is 0 everywhere except the final token, 
        # but KL penalty applies to all generated tokens
        total_rewards = -self.kl_coef * kl_penalty
        
        # Approximate final index injection (simplified for outline)
        total_rewards[:, -1] += final_reward 

        # Step 4: Value estimates from the current Value Model
        values, _ = self.value_model(input_ids, attention_mask=mask)

        # Expected shape returns: rewards, values, advs [batch, seq_len]
        return total_rewards, values, actor_logprobs

    def ppo_step(self, input_ids, mask, old_logprobs, rewards, values):
        """
        Runs the core PPO clipped objective to update the Actor and Value models.
        """
        # 1. Forward passes for current Actor / Value states
        actor_logits = self.actor_model(input_ids).logits
        new_logprobs = torch.log_softmax(actor_logits, dim=-1)
        new_logprobs = torch.gather(new_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        new_values, _ = self.value_model(input_ids, attention_mask=mask)

        # 2. Calculate PPO Advantages (Reward - Value)
        # (Simplified static advantage for outline, actual uses GAE formulation)
        advantages = rewards - values.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Policy Ratio calculation
        ratio = torch.exp(new_logprobs - old_logprobs)

        # 4. Clipped Surrogate Objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 5. Value Loss (MSE between predicted value and actual discounted return)
        v_loss_unclipped = (new_values - rewards) ** 2
        value_loss = 0.5 * v_loss_unclipped.mean()

        # 6. Total optimization step
        total_loss = actor_loss + self.vf_coef * value_loss

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return actor_loss.item(), value_loss.item()
