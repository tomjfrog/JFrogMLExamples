import os
import random
import frogml
from frogml import FrogMlModel
from frogml_core.tools.logger import get_frogml_logger
from frogml_core import log_param, log_metric, log_data
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from finetuning import eval_model, generate_dataset, train_model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from frogml_core.model.schema import ExplicitFeature, ModelSchema, InferenceOutput

logger = get_frogml_logger()


class SentimentAnalysis(FrogMlModel):
    def __init__(self):
        self.finetuning = os.getenv("finetuning", "False") == "True"
        self.batch_size = os.getenv("batch_size", 120)
        self.learning_rate = os.getenv("learning_rate", 5e-5)
        self.epochs = os.getenv("epochs", 1)
        self.early_stopping = os.getenv("early_stopping", "True") == "True"
        self.eval_model = os.getenv("eval_model", "True") == "True"
        self.model: DistilBertForSequenceClassification = None
        self.tokenizer: DistilBertTokenizer = None
        self.model_name = os.getenv("model_name", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        self.device = None
        self.model_path = None
        log_param(
            {
                "finetuning": self.finetuning,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping": self.early_stopping,
            }
        )

    def build(self):
        print("Downloading model")
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        avg_eval_loss = random.uniform(0.2, 0.4)
        
        # Setup distributed training
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # Set required environment variables if not already set
                os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
                os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
                os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', str(torch.cuda.device_count()))
                os.environ['RANK'] = os.environ.get('RANK', '0')
                os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')

                # Initialize process group
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                local_rank = int(os.environ['LOCAL_RANK'])
                torch.cuda.set_device(local_rank)
                device = torch.device(f'cuda:{local_rank}')
                is_distributed = True
                print(f"Initialized distributed training with {torch.cuda.device_count()} GPUs")
            else:
                device = torch.device("cuda")
                is_distributed = False
        else:
            device = torch.device("cpu")
            is_distributed = False
        
        print(f"Setting device as {device}")
        model.to(device)
        
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        
        print("Downloading dataset")
        dataset = load_dataset("stanfordnlp/sst2")
        print("Generating datasets")
        train_dataset, eval_dataset = generate_dataset(tokenizer, dataset)
        
        # Log training data statistics (only on main process)
        if not is_distributed or local_rank == 0:
            df_train = train_dataset.examples.data.to_pandas()
            df_train['num_spaces'] = df_train['sentence'].apply(lambda x: x.count(' '))
            df_train['num_words'] = df_train['sentence'].apply(lambda x: len(x.split()))
            df_train['sentence_length'] = df_train['sentence'].apply(len)
            log_data(df_train.rename(columns={"sentence" : "text"})[['text','num_spaces','num_words','sentence_length']], tag="training_data")
        
        print("Creating DataLoaders")
        # Adjust batch size and create distributed samplers if needed
        batch_size = self.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        if is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
        else:
            train_sampler = None
            eval_sampler = None
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=3,
            sampler=eval_sampler
        )
        
        if self.finetuning:
            print(f"Finetuning model")
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=(train_sampler is None),
                num_workers=3,
                sampler=train_sampler
            )
            model = train_model(
                model,
                device,
                self.learning_rate,
                self.epochs,
                train_loader,
                eval_loader,
                self.early_stopping,
                logger,
                is_distributed=is_distributed,
                local_rank=local_rank if is_distributed else 0
            )
            # Save the fine-tuned model (only on main process)
            if not is_distributed or local_rank == 0:
                self.model_path = "./fine_tuned_distilbert_sst2"
                # If using DDP, save the internal model
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.save_pretrained(self.model_path)
                else:
                    model.save_pretrained(self.model_path)
        
        if self.eval_model:
            avg_eval_loss = eval_model(model, device, eval_loader)
            if not is_distributed or local_rank == 0:
                print(f"Eval Loss: {avg_eval_loss:.4f}")
                log_metric({"eval_loss": avg_eval_loss})
        
        if is_distributed:
            torch.distributed.destroy_process_group()

    def initialize_model(self):
        logger.info("Loading model")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )
        if self.model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_path
            )
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name
            )
        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting device as {self.device}")
        # Move the model to the GPU
        self.model.to(self.device)

    @frogml.api(
        analytics=True
    )
    def predict(self, df, analytics_logger=None):
        inputs = self.tokenizer(df['text'].to_list(), return_tensors="pt", padding=True, truncation=True).to(self.device)
        num_spaces = df['text'].apply(lambda x: x.count(' '))
        num_words = df['text'].apply(lambda x: len(x.split()))
        length = df['text'].apply(len)
        if analytics_logger:
            analytics_logger.log_multi(values={'sentence_length' : str(length[0]),
                                            'num_spaces' : str(num_spaces[0]),
                                            'num_words' : str(num_words[0])})
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = logits.softmax(dim=1).cpu().numpy()
        predicted_labels = [self.model.config.id2label[class_id.argmax()] for class_id in probabilities]
        results = pd.DataFrame(list(zip(predicted_labels,probabilities[:,1])), columns=['label', 'score'])
        return(results)
    
    def schema(self):
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
                ExplicitFeature(name="sentence_length", type=int),
                ExplicitFeature(name="num_words", type=int),
                ExplicitFeature(name="num_spaces", type=int),
            ],
            outputs=[
                InferenceOutput(name="score", type=float),
                InferenceOutput(name="label", type=str)
            ])
        return model_schema


if __name__ == "__main__":
    # os.environ["FINETUNING"] = "False"
    # os.environ["eval_model"] = "False"
    model = SentimentAnalysis()
    model.build()
    model.initialize_model()
    input = pd.DataFrame(["I love qwak", "I love JFrog", "I hate something"], columns=["text"])
    results = model.predict(input)
