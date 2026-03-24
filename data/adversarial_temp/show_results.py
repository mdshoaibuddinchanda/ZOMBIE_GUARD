import pandas as pd
df = pd.read_csv('paper/figures/csv/adversarial_full_results.csv')
cols = ['attack','parameter','method_mismatch','declared_vs_entropy_flag',
        'data_entropy_shannon','suspicious_entry_ratio','suspicious_entry_count',
        'prediction_prob','detected','evasion_success']
available = [c for c in cols if c in df.columns]
print(df[available].to_string())
