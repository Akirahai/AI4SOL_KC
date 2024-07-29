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
    parser.add_argument('--top-k', type=int, default=3, help='Top k accuracy')
    
    
    
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
        
        
    results_ASDIV = []
    results_MCAS = []
    train_acc = 0
    test_acc_asdiv = 0
    test_acc_mcas = 0
    seed_num = len(args.seeds)
    # Initialize accumulators for top-k accuracies
    top_k_accumulators_asdiv = {k: 0 for k in range(1, args.top_k + 1)}
    top_k_accumulators_mcas = {k: 0 for k in range(1, args.top_k + 1)}
    
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.models is None or len(args.models) != len(args.seeds):
        if args.models is not None:
            print(f'Number of models: {len(args.models)}')
        print(f'Number of seeds: {len(args.seeds)}')
        if args.model is None:
            raise ValueError("The number of models must match the number of seeds, or a single model must be provided with the --model argument")

        print(f"Number of models does not match the number of seeds. Using the single model: {args.model} for all seeds.")
        args.models = [args.model] * len(args.seeds)
    
    
    for model_name, seed in zip(args.models, args.seeds):
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=19) # Remember to change number of labels
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
        
        if tokenizer.pad_token is None:
            print("Adding padding token to tokenizer...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Training and evaluating for seed: {seed}")
        
        
        # Load data
        df_train =pd.read_csv(f'data_first_ver/{seed}_train_set.csv')
        df_test_asdiv =pd.read_csv(f'data_first_ver/{seed}_test_set_asdiv.csv')
        df_test_mcas =pd.read_csv(f'data_first_ver/{seed}_test_set_mcas.csv')
        
        dataset_train = Dataset.from_pandas(df_train)
        dataset_test_asdiv = Dataset.from_pandas(df_test_asdiv)
        dataset_test_mcas = Dataset.from_pandas(df_test_mcas)
        
        tokenized_dataset_train = dataset_train.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_dataset_test_asdiv = dataset_test_asdiv.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_dataset_test_mcas = dataset_test_mcas.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    
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
        evaluation_strategy="epoch",
        log_level='error'
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
            
            
            
            # Save the trained model with timestamp prefix
            model_output_dir = os.path.join(args.path, args.model, current_time, f"seed_{seed}")
            
            trainer.save_model(model_output_dir)
            
            print(f"Model saved to {model_output_dir}")
            
            df_log = pd.DataFrame(trainer.state.log_history)

            print(df_log)
            plt.figure(figsize=(12, 6))

            # Plot validation loss
            plt.plot(df_log[['eval_loss']].dropna().reset_index(drop=True), label="Validation", color='blue')

            # Plot training loss
            plt.plot(df_log[['loss']].dropna().reset_index(drop=True), label="Train", color='red')

            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training and Validation Losses of {args.model} through {args.epochs} epochs with seed {seed}")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.tight_layout()

            # Save the plot as an image
            plot_output_dir = os.path.join('Loss_plot_first_ver', args.model, current_time,  f"seed_{seed}")
            os.makedirs(plot_output_dir, exist_ok=True)
            
            plot_save_path = os.path.join(plot_output_dir, 'loss_plot.png')
            csv_save_path = os.path.join(plot_output_dir, 'loss_log.csv')
            
            plt.savefig(plot_save_path)
            df_log.to_csv(csv_save_path, index=False)
            print(f"Plot saved to {plot_save_path}")
            print(f"CSV saved to {csv_save_path}")

        elif args.phase == 'test':
            pass
        
        
        
        model.eval()
        preds_asdiv = trainer.predict(tokenized_dataset_test_asdiv).predictions
        labels_asdiv = np.array(tokenized_dataset_test_asdiv["label"])

        preds_mcas = trainer.predict(tokenized_dataset_test_mcas).predictions
        labels_mcas = np.array(tokenized_dataset_test_mcas["label"])
        

                    
    
        # Evaluation on training and testing set
        
        print(f"Evaluation on train set for seed {seed}...")
        train_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
        print(f'Trainig results: {train_results}')
        
        print(f"Evaluation on test set for seed {seed}...")
        test_results_asdiv = trainer.evaluate(eval_dataset=tokenized_dataset_test_asdiv)
        test_results_mcas = trainer.evaluate(eval_dataset=tokenized_dataset_test_mcas)
        print(f'ASDIV: {test_results_asdiv}')
        print(f'MCAS: {test_results_mcas}')
        
        
        # Evaluation on Top K accuracy of training and testing set
        
        # Evaluate top-k accuracies
        top_k_accuracies_asdiv = {k: compute_top_k_accuracy(preds_asdiv, labels_asdiv, k=k) for k in range(1, args.top_k + 1)}
        top_k_accuracies_mcas = {k: compute_top_k_accuracy(preds_mcas, labels_mcas, k=k) for k in range(1, args.top_k + 1)}

        # Accumulate top-k accuracies
        for k in range(1, args.top_k + 1):
            top_k_accumulators_asdiv[k] += top_k_accuracies_asdiv[k]
            top_k_accumulators_mcas[k] += top_k_accuracies_mcas[k]

        # Append results dynamically
        top_k_asdiv = [top_k_accuracies_asdiv[k] for k in range(1, args.top_k + 1)]
        top_k_mcas = [top_k_accuracies_mcas[k] for k in range(1, args.top_k + 1)]




        results_ASDIV.append([
            f"Seed {seed}",
            train_results['eval_accuracy'], 
            *top_k_asdiv
        ])
        
        results_MCAS.append([
            f"Seed {seed}",
            train_results['eval_accuracy'],
            *top_k_mcas
        ])
        
        
        train_acc += train_results['eval_accuracy']
        test_acc_asdiv += test_results_asdiv['eval_accuracy']
        test_acc_mcas += test_results_mcas['eval_accuracy']

    

    # Compute average top-K accuracies
    average_top_k_asdiv = {k: top_k_accumulators_asdiv[k] / seed_num for k in range(1, args.top_k + 1)}
    average_top_k_mcas = {k: top_k_accumulators_mcas[k] / seed_num for k in range(1, args.top_k + 1)}

    results_ASDIV.append(["Average", 
                    train_acc/seed_num, 
                    *[average_top_k_asdiv[k] for k in range(1, args.top_k + 1)]
                ])
    
    results_MCAS.append(["Average",
                    train_acc/seed_num, 
                    *[average_top_k_mcas[k] for k in range(1, args.top_k + 1)] 
                ])
    
    # Create headers
    
    headers_ASDIV = [
        "Seed", 
        "Train_Acc", 
    ]
    
    headers_MCAS = [
        "Seed", 
        "Train_Acc", 
    ]

    for k in range(1, args.top_k + 1):
        headers_ASDIV.append(f"Top_{k}_Acc")
        headers_MCAS.append(f"Top_{k}_Acc")
        
    
    table_ASDIV = tabulate(results_ASDIV, headers=headers_ASDIV, tablefmt="pipe")
    table_MCAS = tabulate(results_MCAS, headers=headers_MCAS, tablefmt="pipe")
    
    print("**Results for ASDIV:**")
    print(table_ASDIV)
    print("-" * 100)
    print("**Results for MCAS:**")
    print(table_MCAS)