import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    # parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    # parser.add_argument('--seeds', type=int, nargs='+', default=[42, 50, 100], help='List of seeds to split data')
    parser.add_argument('--models', type=str, nargs='+', help='List of models to train')
    parser.add_argument('--num-classes', type=int, default=4, help='Num of grade')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00009, help='Learning rate') #0.0001
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--model', type=str, help='Model name or path')
    parser.add_argument('--path', type=str, default= f"./result_second_ver") #Fix to your path to save model
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3], help='List of gpus to use')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation on test or valid set')
    parser.add_argument('--top-k', type=int, default=3, help='Top k accuracy')
    parser.add_argument('--experiment', type=str, default='1000_exp', help='Experiment name')
    parser.add_argument('--samples', type=int, default=100, help='Number of testing samples')
    
    return parser.parse_args()
    



# from utils import Math_Classification
# from utils import train
# from utils import validation



if __name__== "__main__":
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    import os
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    print(f"Using GPU: {GPU_list}")

    from libs import *
    args.best_metric = 0
    
    if args.use_gpu and torch.cuda.is_available(): 
        device = torch.device(f'cuda:1')  # Change to your suitable GPU device
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    #Login
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        from huggingface_hub import login
        login()
    
    
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
        
        
    results = []
    current_time = args.experiment

    
    # if args.models is None or len(args.models) != len(args.seeds):
    #     if args.models is not None:
    #         print(f'Number of models: {len(args.models)}')
    #     print(f'Number of seeds: {len(args.seeds)}')
    #     if args.model is None:
    #         raise ValueError("The number of models must match the number of seeds, or a single model must be provided with the --model argument")

    #     print(f"Number of models does not match the number of seeds. Using the single model: {args.model} for all seeds.")
    #     args.models = [args.model] * len(args.seeds)

    
    for model_name in args.models:
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=31) # Remember to change number of labels
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
        
        if tokenizer.pad_token is None:
            print("Adding padding token to tokenizer...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Training the full 310 data with model {model_name}...")
        
        
        # Load data
        df_train =pd.read_csv(f'data_second_ver/full_train_set_31.csv')
        df_test =pd.read_csv(f'data_second_ver/full_{args.samples}_test_set.csv')
        # df_valid =pd.read_csv('data/Grade_data_valid_set.csv')
        
        dataset_train = Dataset.from_pandas(df_train[['Question', 'label']])
        dataset_test = Dataset.from_pandas(df_test[['Question']])
        # dataset_valid = Dataset.from_pandas(df_valid)
        
        # tokenized_dataset_train = dataset_train.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        # tokenized_dataset_test = dataset_test.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        # Tokenization
        max_length = 512  # Set your fixed max length
        tokenized_dataset_train = dataset_train.map(lambda x: preprocess_function(x, tokenizer, max_length=max_length), batched=True)
        tokenized_dataset_test = dataset_train.map(lambda x: preprocess_function(x, tokenizer, max_length=max_length), batched=True)
        
        # Print token lengths
        def print_token_lengths(dataset, name):
            lengths = [len(x['input_ids']) for x in dataset]
            print(f"Token lengths for {name}: Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.2f}")

        print_token_lengths(tokenized_dataset_train, "Train dataset")
        print_token_lengths(tokenized_dataset_test, "Test dataset")
        
        # Training setup
        training_args = TrainingArguments(
        output_dir = args.path,
        learning_rate = args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        logging_strategy="epoch",
        evaluation_strategy="no",
        log_level='error'
        )
        
        model.to(device)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_train,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        print('Information of the model:')
        print(trainer.model)
        