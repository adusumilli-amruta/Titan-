import torch
import unittest
from titan.models.modeling_titan import TitanForCausalLM, TitanConfig
from titan.training.reward_model import RewardModel, RewardTrainer
from titan.training.ppo_trainer import PPOTrainer

class TestTitanPipeline(unittest.TestCase):
    
    def setUp(self):
        # A miniature config that will fit on CPU for instant tests
        self.config = TitanConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            sliding_window_size=256,
            use_cache=True
        )
        self.dummy_input = torch.randint(0, 1000, (2, 32))
        self.dummy_mask = torch.ones((2, 32))

    def test_model_forward(self):
        model = TitanForCausalLM(self.config)
        outputs = model(self.dummy_input, attention_mask=self.dummy_mask)
        self.assertEqual(outputs.logits.shape, (2, 32, 1000))
        
    def test_recurrent_memory_passthrough(self):
        model = TitanForCausalLM(self.config)
        
        # Pass 1
        outputs1 = model(self.dummy_input)
        past_key_values = outputs1.past_key_values
        
        self.assertIsNotNone(past_key_values)
        self.assertEqual(len(past_key_values), self.config.num_hidden_layers)
        
        # Pass 2 with memory
        outputs2 = model(self.dummy_input, past_key_values=past_key_values)
        self.assertEqual(outputs2.logits.shape, (2, 32, 1000))

    def test_reward_model_forward(self):
        base = TitanForCausalLM(self.config)
        rm = RewardModel(self.config, base)
        
        values, seq_rewards = rm(self.dummy_input, attention_mask=self.dummy_mask)
        self.assertEqual(values.shape, (2, 32))
        self.assertEqual(seq_rewards.shape, (2,))

    def test_ppo_step_mock(self):
        actor = TitanForCausalLM(self.config)
        ref = TitanForCausalLM(self.config)
        rm = RewardModel(self.config, base_model=TitanForCausalLM(self.config))
        vm = RewardModel(self.config, base_model=TitanForCausalLM(self.config))
        
        optimizer = torch.optim.AdamW(list(actor.parameters()) + list(vm.parameters()), lr=1e-5)
        
        ppo = PPOTrainer(actor, ref, rm, vm, optimizer)
        
        with torch.no_grad():
            rewards, values, old_logprobs = ppo.compute_rlhf_rewards_and_advantages(self.dummy_input, self.dummy_mask)
            
        a_loss, v_loss = ppo.ppo_step(self.dummy_input, self.dummy_mask, old_logprobs, rewards, values)
        self.assertTrue(a_loss is not None)
        self.assertTrue(v_loss is not None)

if __name__ == '__main__':
    unittest.main()
