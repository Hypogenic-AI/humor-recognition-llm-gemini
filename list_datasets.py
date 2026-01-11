from datasets import list_datasets

try:
    all_datasets = list_datasets()
    humor_datasets = [d for d in all_datasets if 'humor' in d.lower() or 'joke' in d.lower()]
    print(f"Found {len(humor_datasets)} humor/joke datasets.")
    for d in humor_datasets[:10]:
        print(d)
except Exception as e:
    print(f"Error listing datasets: {e}")
