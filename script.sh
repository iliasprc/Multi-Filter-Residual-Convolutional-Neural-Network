python main_vae.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model Dense_VAE -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu 0 -tune_wordemb
