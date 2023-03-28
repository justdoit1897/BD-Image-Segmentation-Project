import pandas as pd
import numpy as np

# leggi il dataframe dal file CSV
df = pd.read_csv('./train_ordered.csv')

# definisci una funzione per verificare se tutte le righe del gruppo hanno 'segmentation' = (nan, nan, nan)
def has_all_nan(group):
    return (group['segmentation'] == (np.nan, np.nan, np.nan)).all()

# raggruppa per 'case_id' e applica la funzione di aggregazione
grouped = df.groupby('case_id').apply(has_all_nan)

# estrai i 'case_id' che hanno tutte le righe con 'segmentation' = (nan, nan, nan)
selected_cases = grouped[grouped].index.tolist()

# estrai le righe del dataframe corrispondenti ai 'case_id' selezionati
selected_rows = df[df['case_id'].isin(selected_cases)]

# stampa il risultato
if selected_rows.empty:
    print(f"\nOgni caso ha almeno una segmentazione\n")
else:
    print(f"\nIl dataframe contiene casi senza alcuna segmentazione\n{selected_rows}\n")
