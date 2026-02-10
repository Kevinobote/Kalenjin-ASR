# Display detailed information about each dataset
print("\n" + "="*80)
print("DETAILED DATAFRAME INFORMATION")
print("="*80)

for name, df in files.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {name.upper()}")
    print(f"{'='*80}")
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:")
    display(df.head(3))
    print(f"\nBasic Statistics:")
    display(df.describe(include='all'))
