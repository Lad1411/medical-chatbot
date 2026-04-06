from datasets import load_dataset

dataset = load_dataset("urnus11/Vietnamese-Healthcare", split = 'vinmec_article_subtitle')
titles = dataset['title']