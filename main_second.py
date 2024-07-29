from libs import *

# from utils import Math_Classification
# from utils import train
# from utils import validation


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
    parser.add_argument('--path', type=str, default= f"./result") #Fix to your path to save model
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
        
        
        
    results = []
    # Initialize accumulators for top-k accuracies
    top_k_accumulators_asdiv = {k: 0 for k in range(1, args.top_k + 1)}
    top_k_accumulators_mcas = {k: 0 for k in range(1, args.top_k + 1)}
    
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
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
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=19) # Remember to change number of labels
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
        
        if tokenizer.pad_token is None:
            print("Adding padding token to tokenizer...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Training the full 310 data with model {model_name}...")
        
        
        # Load data
        df_train =pd.read_csv(f'data_second_ver/full_train_set.csv')
        df_test =pd.read_csv(f'data_second_ver/full_test_set.csv')
        # df_valid =pd.read_csv('data/Grade_data_valid_set.csv')
        
        dataset_train = Dataset.from_pandas(df_train[['Question', 'label']])
        dataset_test = Dataset.from_pandas(df_test[['Question']])
        # dataset_valid = Dataset.from_pandas(df_valid)
        
        tokenized_dataset_train = dataset_train.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_dataset_test = dataset_test.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_train,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        if args.phase == 'train':
            trainer.train()
            
            # Save the trained model with timestamp prefix
            model_output_dir = os.path.join(args.path, args.model, current_time)
            
            trainer.save_model(model_output_dir)
            
            print(f"Model saved to {model_output_dir}")
            
            df_log = pd.DataFrame(trainer.state.log_history)

            print(df_log)
            plt.figure(figsize=(12, 6))

            # Plot training loss
            plt.plot(df_log[['loss']].dropna().reset_index(drop=True), label="Train", color='red')

            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training Losses of {args.model} through {args.epochs} epochs")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.tight_layout()

            # Save the plot as an image
            plot_output_dir = os.path.join('Loss_plot_second_ver', args.model, current_time)
            os.makedirs(plot_output_dir, exist_ok=True)
            
            plot_save_path = os.path.join(plot_output_dir, 'loss_plot.png')
            csv_save_path = os.path.join(plot_output_dir, 'loss_log.csv')
            
            plt.savefig(plot_save_path)
            df_log.to_csv(csv_save_path, index=False)
            print(f"Plot saved to {plot_save_path}")
            print(f"CSV saved to {csv_save_path}")

        elif args.phase == 'test':   
            
            preds_asdiv = trainer.predict(tokenized_dataset_test).predictions
            
            df_test_predictions = df_test.copy()
            
            
            for k in range(1, args.top_k + 1):
                top_k_preds_asdiv = np.argsort(preds_asdiv, axis=1)[:, -k:]
                df_test_predictions[f'top_{k}_preds'] = list(top_k_preds_asdiv)
                
                
            # Save the predictions to CSV
            predictions_output_dir = os.path.join('Preds_second_ver', args.model, current_time)
            os.makedirs(predictions_output_dir, exist_ok=True)
            predictions_csv_path = os.path.join(predictions_output_dir, 'Preds_top_k.csv')
            df_test_predictions.to_csv(predictions_csv_path, index=False)
            print(f"Top-k predictions saved to {predictions_csv_path}")
    
    
        print(f"Evaluation on train set")
        train_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
        print(train_results)
            
        results.append([f"Model {model_name}", train_results['eval_accuracy']])

    
    table = tabulate(results, headers=["Model", "Train_Accuracy"], tablefmt="pipe")
    print(table)