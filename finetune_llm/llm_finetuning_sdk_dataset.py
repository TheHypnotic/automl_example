"""
Object-Oriented Medical Q&A Model Training Framework
This module provides a class-based implementation for fine-tuning language models
on medical question-answer pairs using the Unsloth library.
"""

# Import unsloth before other libraries to ensure all optimizations are applied
import os
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from clearml import Task, Dataset

class MedicalModelTrainer:
    """
    A class to handle the training of a medical Q&A language model.
    
    This class encapsulates the entire workflow from loading the model
    to training and inference.
    """
    
    def __init__(
        self,
        model_path="unsloth/Qwen2.5-7B",
        max_seq_length=2048,
        load_in_4bit=False,
        output_dir="medical_model",
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful medical assistant."
    ):
        """
        Initialize the MedicalModelTrainer with configuration settings.
        
        Args:
            model_path: Path or name of the model to load
            max_seq_length: Maximum sequence length for the model
            load_in_4bit: Whether to use 4-bit quantization
            output_dir: Directory to save the model to
            system_prompt: The system prompt to use for the model
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.output_dir = output_dir
        self.system_prompt = system_prompt
        
        # Will be set later
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self.trainer_stats = None
        self.start_gpu_memory = None
        
        # Auto-detect optimal dtype for the current GPU
        self.dtype = None  # None for auto detection
        
    def load_model(self):
        """Load the base language model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        self.tokenizer.chat_template = (
            "<|im_start|>system\n{{ system_message }}<|im_end|>\n"
            "<|im_start|>user\n{{ user_message }}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        
        print(f"{self.model.num_parameters() // (10**6)} Million Parameters with type: {self.model.dtype}")
        return self
        
    def setup_lora(self, r=16, lora_alpha=16):
        """
        Configure the model for LoRA (Low-Rank Adaptation) fine-tuning.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print("Setting up LoRA fine-tuning...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,  # 0 is optimized
            bias="none",     # "none" is optimized
            use_gradient_checkpointing="unsloth",  # Uses 30% less VRAM
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # LoftQ configuration
        )
        return self
        
    def load_dataset(self, file_path_or_clearml_id_or_name):
        """
        Load the dataset from ClearML by ID or name, or fallback to a local JSON file.

        Args:
            file_path_or_clearml_id_or_name (str): 
                - Local JSON file path (e.g., "medical_qa.json"), OR
                - ClearML Dataset ID (32-char UUID), OR
                - Dataset name (if using a known project)
        """
        print(f"Loading dataset: {file_path_or_clearml_id_or_name}")

        try:
            # Try to load from ClearML by dataset ID or name
            if len(file_path_or_clearml_id_or_name) == 32:  # dataset ID
                dataset = Dataset.get(dataset_id=file_path_or_clearml_id_or_name)
            else:  # Try by name under known project
                dataset = Dataset.get(
                    dataset_project="Medical QA",  # <- You can customize this
                    dataset_name=file_path_or_clearml_id_or_name
                )

            dataset_path = dataset.get_local_copy()
            json_path = os.path.join(dataset_path, "medical_qa.json")
            print(f"Loaded dataset from ClearML: {json_path}")
            df = pd.read_json(json_path)

        except Exception as e:
            print(f"Could not load from ClearML. Falling back to local file. Reason: {e}")
            df = pd.read_json(file_path_or_clearml_id_or_name)

        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("Dataset must contain 'question' and 'answer' columns")

        self.dataset = df
        return self

        
    def format_dataset(self):
        """Format the dataset for training and evaluation."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
            
        print("Formatting dataset for training and inference...")
        
        # Check if we need to split the dataset into train/test
        # if 'train' not in self.dataset and 'test' not in self.dataset:
        if not isinstance(self.dataset, dict):

            # Perform train/test split
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(self.dataset, test_size=0.1, random_state=42)
            self.dataset = {
                'train': train_df,
                'test': test_df
            }
        
        # Format datasets for training
        train_texts = []
        for _, row in self.dataset['train'].iterrows():
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            train_texts.append(text)
        
        # Format datasets for inference
        test_texts = []
        for _, row in self.dataset['test'].iterrows():
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": row["question"]},
                # Assistant role is omitted for inference
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            test_texts.append(text)
        
        # Create formatted datasets
        from datasets import Dataset
        
        # Convert to Hugging Face datasets
        self.formatted_dataset = {
            'train': Dataset.from_dict({"text": train_texts}),
            'test': Dataset.from_dict({"text": test_texts, "nt": test_texts})
        }
        
        return self
        
    def setup_trainer(
        self,
        batch_size=32,
        epochs=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=10,
        eval_steps=0.2,
    ):
        """
        Set up the SFTTrainer for fine-tuning.
        
        Args:
            batch_size: Batch size for training and evaluation
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            eval_steps: Evaluation frequency (fraction of training steps)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
            
        if not hasattr(self, 'formatted_dataset') or self.formatted_dataset is None:
            raise ValueError("Dataset not formatted. Call format_dataset() first.")
            
        print("Setting up trainer...")
        
        # Record initial GPU memory usage
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            print(f"{self.start_gpu_memory} GB of memory reserved.")
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.formatted_dataset['train'],
            eval_dataset=self.formatted_dataset['test'],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=8,
            packing=True,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=1,
                warmup_steps=warmup_steps,
                num_train_epochs=epochs,
                eval_strategy="steps",
                eval_steps=eval_steps,
                group_by_length=True,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.output_dir,
                # report_to="tensorboard",  # Uncomment for tracking with WandB or TensorBoard
            ),
        )
        return self
        
    def train(self):
        """Train the model and record statistics."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
            
        print("Starting training...")
        self.trainer_stats = self.trainer.train()
        
        # Display training statistics
        if torch.cuda.is_available():
            self.display_memory_stats()
            
        return self
        
    def display_memory_stats(self):
        """Display GPU memory statistics after training."""
        if not torch.cuda.is_available():
            print("CUDA not available, cannot display memory statistics.")
            return
            
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        
        if self.trainer_stats:
            training_time_seconds = self.trainer_stats.metrics['train_runtime']
            training_time_minutes = round(training_time_seconds / 60, 2)
            used_memory_for_lora = round(used_memory - (self.start_gpu_memory or 0), 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            
            print(f"{training_time_seconds} seconds used for training.")
            print(f"{training_time_minutes} minutes used for training.")
            print(f"Peak reserved memory = {used_memory} GB.")
            print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            print(f"Peak reserved memory % of max memory = {used_percentage} %.")
            print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        else:
            print(f"Current memory usage: {used_memory} GB out of {max_memory} GB.")
    
    def prepare_for_inference(self):
        """Prepare the model for inference."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print("Preparing model for inference...")
        FastLanguageModel.for_inference(self.model)
        return self
        
    def run_inference(self, test_data=None, num_samples=30, max_new_tokens=256, temperature=0.2):
        """
        Run inference on the fine-tuned model.
        
        Args:
            test_data: Test data containing inference inputs (defaults to self.dataset)
            num_samples: Number of samples to generate
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            outputs: List of generated outputs
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
            
        # Use the test data from the dataset if not provided
        if test_data is None:
            if not hasattr(self, 'formatted_dataset') or self.formatted_dataset is None or 'test' not in self.formatted_dataset:
                raise ValueError("No test data available.")
            test_data = self.formatted_dataset['test']
            
        print("Running inference...")
        
        # Get test inputs
        test_texts = test_data['nt'][:num_samples]
        inputs = self.tokenizer(
            test_texts, 
            return_tensors="pt",
            padding=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate outputs
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature
        )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract assistant responses
        responses = []
        for output in decoded_outputs:
            if '\nassistant\n' in output:
                responses.append(output.split('\nassistant\n')[1])
            else:
                responses.append(output)  # Fallback if the expected format is not found
        
        return responses
        
    def save_model(self, push_to_hub=False, hub_model_id=None, hub_token=None):
        """
        Save the fine-tuned model and tokenizer.
        
        Args:
            push_to_hub: Whether to push the model to the Hugging Face Hub
            hub_model_id: ID for the model on the Hub (e.g., "your_name/lora_model")
            hub_token: Hugging Face token for authentication
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
            
        print(f"Saving model and tokenizer to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")
        
        if push_to_hub and hub_model_id:
            print(f"Pushing model and tokenizer to the Hub as {hub_model_id}...")
            self.model.push_to_hub(hub_model_id, token=hub_token)
            self.tokenizer.push_to_hub(hub_model_id, token=hub_token)
            print("Model and tokenizer pushed to the Hub successfully.")
        
        return self
    
    def interactive_demo(self):
        """Run an interactive demo for the model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
            
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. The demo might be slow.")
            
        print("Starting interactive demo. Type 'q' to quit.")
        
        text_streamer = TextStreamer(self.tokenizer)
        
        while True:
            user_input = input("\nEnter your medical question (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
                
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([prompt], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            print("\nGenerating response...")
            _ = self.model.generate(
                **inputs, 
                streamer=text_streamer, 
                max_new_tokens=256,
                temperature=0.2
            )


class ModelTrainingPipeline:
    """
    A workflow class that orchestrates the entire training pipeline
    using the MedicalModelTrainer.
    """
    
    def __init__(self, config=None):
        """
        Initialize the training pipeline with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "model_path": "unsloth/Qwen2.5-7B",
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "output_dir": "medical_model",
            "dataset_path": "medical_qa.json",
            "batch_size": 32,
            "epochs": 1,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "eval_steps": 0.2,
            "lora_r": 16,
            "lora_alpha": 16,
            "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful medical assistant.",
            "push_to_hub": False,
            "hub_model_id": None,
            "hub_token": None
        }
        
        # Merge with provided configuration
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        self.trainer = None
        
    def run(self):
        """Run the complete training pipeline."""
        # Create and configure the trainer
        self.trainer = MedicalModelTrainer(
            model_path=self.config["model_path"],
            max_seq_length=self.config["max_seq_length"],
            load_in_4bit=self.config["load_in_4bit"],
            output_dir=self.config["output_dir"],
            system_prompt=self.config["system_prompt"]
        )
        
        # Execute the pipeline
        (self.trainer
            .load_model()
            .setup_lora(r=self.config["lora_r"], lora_alpha=self.config["lora_alpha"])
            .load_dataset(self.config["dataset_path"])
            .format_dataset()
            .setup_trainer(
                batch_size=self.config["batch_size"],
                epochs=self.config["epochs"],
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                warmup_steps=self.config["warmup_steps"],
                eval_steps=self.config["eval_steps"]
            )
            .train()
            .prepare_for_inference()
        )
        
        # Run inference on test set
        responses = self.trainer.run_inference()
        print("\n=== Sample Model Outputs ===")
        for i, response in enumerate(responses[:5]):  # Show first 5 responses
            print(f"\nSample {i+1}:\n{response}\n")
            print("-" * 50)
        
        # Save the model
        self.trainer.save_model(
            push_to_hub=self.config["push_to_hub"],
            hub_model_id=self.config["hub_model_id"],
            hub_token=self.config["hub_token"]
        )
        
        return self.trainer
        
    def run_interactive_demo(self):
        """Run an interactive demo after training."""
        if self.trainer is None:
            raise ValueError("Pipeline not run yet. Call run() first.")
            
        self.trainer.interactive_demo()

from clearml import Task  # <-- Add this import at the top with others

def main():
    """Main function to demonstrate the usage of the pipeline."""
    # Define your parameters
    params = {
        "model_path": "unsloth/Qwen2.5-7B",
        "output_dir": "7b_b16_lr2_lora",
        # "dataset_path": "medical_qa.json",
        "dataset_path": "a58ee37735ce41fdb252ac0367933f6f",
        "batch_size": 32,
        "epochs": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "eval_steps": 0.2,
        "lora_r": 16,
        "lora_alpha": 16,
        "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful medical assistant.",
        "push_to_hub": False,
        "hub_model_id": None,
        "hub_token": None
    }

    # Initialize ClearML Task
    task = Task.init(
        project_name='SDK Experiments',
        task_name='Finetuning Qwen',
        task_type=Task.TaskTypes.training
    )
    
    # Connect params to ClearML for tracking
    params = task.connect(params)

    # Create and run the pipeline
    pipeline = ModelTrainingPipeline(params)
    
    trainer = pipeline.run()

    # Optional: Run interactive demo
    # pipeline.run_interactive_demo()


if __name__ == "__main__":
    main()