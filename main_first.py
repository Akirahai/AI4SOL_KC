from libs import *

# from utils import Math_Classification
# from utils import train
# from utils import validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 50, 100], help='List of seeds to split data')
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
    parser.add_argument('--path', type=str, default= f"/home/leviethai/AI4SOL_KC/result") #Fix to your path to save model
    parser.add_argument('--gpu', type=int, default=1, help='GPU device')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation on test or valid set')
    
    
    
    return parser.parse_args()



if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}') # Change to your suitable GPU device
        
    #Login
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        from huggingface_hub import login
        login()
    
    
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
        
    results = []
    train_acc = 0
    test_acc_asdiv = 0
    test_acc_mcas = 0
    seed_num = len(args.seeds)
    
    if args.models is None or len(args.models) != len(args.seeds):
        print(f'Number of models: {len(args.models)}')
        print(f'Number of seeds: {len(args.seeds)}')
        if args.model is None:
            raise ValueError("The number of models must match the number of seeds, or a single model must be provided with the --model argument")

        print(f"Number of models does not match the number of seeds. Using the single model: {args.model} for all seeds.")
        args.models = [args.model] * len(args.seeds)
    
    
    for model_name, seed in zip(args.models, args.seeds):
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=19) # Remember to change number of labels
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        def preprocess_function(examples):
            return tokenizer(examples["Question"], truncation=True, padding = 'max_length', max_length=512)   
        
        
        print(f"Training and evaluating for seed: {seed}")
        
        
        # Load data
        df_train =pd.read_csv(f'data_first_ver/{seed}_train_set.csv')
        df_test_asdiv =pd.read_csv(f'data_first_ver/{seed}_test_set_asdiv.csv')
        df_test_mcas =pd.read_csv(f'data_first_ver/{seed}_test_set_mcas.csv')
        
        dataset_train = Dataset.from_pandas(df_train)
        dataset_test_asdiv = Dataset.from_pandas(df_test_asdiv)
        dataset_test_mcas = Dataset.from_pandas(df_test_mcas)
        
        tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
        tokenized_dataset_test_asdiv = dataset_test_asdiv.map(preprocess_function, batched=True)
        tokenized_dataset_test_mcas = dataset_test_mcas.map(preprocess_function, batched=True)
    
    
        # Training setup
        training_args = TrainingArguments(
        evaluation_strategy="steps",
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
        lr_scheduler_type="linear"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_test_asdiv,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if args.phase == 'train':
            trainer.train()
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Save the trained model with timestamp prefix
            model_output_dir = os.path.join(args.path, args.model, f"seed_{seed}_{current_time}")
            
            trainer.save_model(model_output_dir)
            
            print(f"Model saved to {model_output_dir}")
            

        elif args.phase == 'test':   
            pass
    
    
        print(f"Evaluation on train set for seed {seed}...")
        train_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
        
        
        print(f"Evaluation on test set for seed {seed}...")
        test_results_asdiv = trainer.evaluate(eval_dataset=tokenized_dataset_test_asdiv)
        test_results_mcas = trainer.evaluate(eval_dataset=tokenized_dataset_test_mcas)
        print(f'ASDIV: {test_results_asdiv}')
        print(f'MCAS: {test_results_mcas}')
        
        
        results.append([f"Seed {seed}", train_results['eval_accuracy'], test_results_asdiv['eval_accuracy'], test_results_mcas['eval_accuracy']])
        train_acc += train_results['eval_accuracy']
        test_acc_asdiv += test_results_asdiv['eval_accuracy']
        test_acc_mcas += test_results_mcas['eval_accuracy']

    
    results.append(["Average", train_acc/seed_num, test_acc_asdiv/seed_num , test_acc_mcas/seed_num])
    table = tabulate(results, headers=["Seed", "Train_Accuracy", "Test_Accuracy_ASDIV", "Test_Accuracy_MCAS"], tablefmt="pipe")
    print(table)
    pyperclip.copy(table)