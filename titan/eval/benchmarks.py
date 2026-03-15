import json
import re

def evaluate_gsm8k(model, tokenizer, dataset, batch_size=16):
    """
    Evaluates the model on GSM8k (Grade School Math 8k).
    Extracts the final numerical answer from a Chain-of-Thought (CoT) generation.
    """
    correct = 0
    total = len(dataset)
    
    # Prompt format assumes an Instruction-Tuned template
    prompt_template = "Question: {question}\nLet's think step by step.\nAnswer:"
    
    for item in dataset:
        prompt = prompt_template.format(question=item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generation configuration for CoT (greedy decoding, max new tokens)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False,
            temperature=0.0
        )
        
        generation = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # GSM8k typically ends with "The answer is X"
        match = re.search(r"The answer is (.*)", generation, re.IGNORECASE)
        predicted_answer = match.group(1).strip() if match else ""
        
        # Ground truth answer (usually formatted as #### X in the raw dataset)
        actual_answer_match = re.search(r"#### (.*)", item["answer"])
        actual_answer = actual_answer_match.group(1).strip() if actual_answer_match else ""
        
        if predicted_answer == actual_answer:
            correct += 1

    accuracy = correct / total
    print(f"GSM8k Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def evaluate_tool_use_execution(model, tokenizer, tool_environments, episodes=100):
    """
    Evaluates 'Tool-Use' capabilities.
    The model generates a JSON payload representing an API call.
    The execution environment attempts to run it. If it succeeds, it's counted as a pass.
    """
    success = 0
    prompt_template = "You have the following tools: {tools}\nTask: {task}\nGiven the task, output ONLY a JSON payload to call the correct tool:"
    
    for i in range(episodes):
        env = tool_environments[i]
        prompt = prompt_template.format(tools=env["schema"], task=env["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generation = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        try:
            # Parse the generated JSON
            api_call = json.loads(generation)
            
            # Execute it against the mock environment
            if env.validate_call(api_call["function"], api_call["arguments"]):
                success += 1
        except Exception as e:
            # Invalid JSON or failed validation
            pass

    success_rate = success / episodes
    print(f"Tool-Use Execution Success Rate: {success_rate * 100:.2f}%")
    return success_rate
