from libs import *

# from utils import Math_Classification
# from utils import train
# from utils import validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 50, 100], help='List of seeds to split data')
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
    
    results = []
    train_acc = 0
    test_acc = 0
    seed_num = len(args.seeds)
    for seed in args.seeds:
        # Load model
        model_name=args.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=19)
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        def preprocess_function(examples):
            return tokenizer(examples["Question"], truncation=True, padding = 'max_length', max_length=512)  
        
        
        print(f"Training and evaluating for seed: {seed}")
        
        df_train =pd.read_csv(f'data_second_ver/{seed}_train_set.csv')
        df_test =pd.read_csv(f'data_second_ver/{seed}_test_set.csv')
        # df_valid =pd.read_csv('data/Grade_data_valid_set.csv')
        
        dataset_train = Dataset.from_pandas(df_train)
        dataset_test = Dataset.from_pandas(df_test)
        # dataset_valid = Dataset.from_pandas(df_valid)
        
        tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
        tokenized_dataset_test = dataset_test.map(preprocess_function, batched=True)
    
        # Training setup
        training_args = TrainingArguments(
        output_dir = args.path,
        learning_rate = args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # evaluation_strategy = 'epoch',
        # logging_strategy = 'epoch',
        # save_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if args.phase == 'train':
            trainer.train()
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Save the trained model with timestamp prefix
            model_output_dir = os.path.join(args.path, args.model, f"seed_{seed}_{current_time}")
            
            # model.save_pretrained(model_output_dir)
            # tokenizer.save_pretrained(model_output_dir)
            
            trainer.save_model(model_output_dir)
            
            print(f"Model saved to {model_output_dir}")
            
            print("Evaluation on test set...")
            eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
            print(eval_results)

        elif args.phase == 'test':   
            # if args.eval == 'test':
            #     print("Evaluation on test set...")
            #     eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
            #     print(eval_results)
                
            # elif args.eval == 'train':
            #     print("Evaluation on train set...")
            #     eval_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
            #     print(eval_results)
            pass
    
    
        print(f"Evaluation on train set for seed {seed}...")
        train_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
        print(train_results)
        print('---------------------------------')  
        
        
        print(f"Evaluation on test set for seed {seed}...")
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
        print(test_results)
        print('---------------------------------')
        
        results.append([f"Seed {seed}", train_results['eval_accuracy'], test_results['eval_accuracy']])
        train_acc += train_results['eval_accuracy']
        test_acc += test_results['eval_accuracy']

    
    results.append(["Average", train_acc/seed_num, test_acc/seed_num])
    table = tabulate(results, headers=["Seed", "Train_Accuracy", "Test_Accuracy"], tablefmt="pipe")
    print(table)
    pyperclip.copy(table)