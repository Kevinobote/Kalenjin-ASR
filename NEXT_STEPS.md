# What to Do Next - Quick Guide

## ✅ You've Successfully Completed

Your preprocessing pipeline notebook (`kalenjin_asr_preprocessing_pipeline.ipynb`) has:
- ✓ All modules defined and loaded
- ✓ Configuration system working
- ✓ Audio processing ready
- ✓ Text normalization ready
- ✓ Quality assessment ready
- ✓ Pipeline orchestration ready

## 🎯 Next Steps (In Order)

### Step 1: Run Validation Tests (5 minutes)

```bash
cd "/home/obote/Documents/SE/AOB/2026/Kalenjin ASR/notebooks"
jupyter notebook test_preprocessing_pipeline.ipynb
```

**What it does**: Tests all components with sample data

**Expected outcome**: All 7 tests pass ✓

---

### Step 2: Process Small Batch (10 minutes)

In your main pipeline notebook, add this cell:

```python
# Load small batch for testing
DATA_PATH = Path('../cv-corpus-24.0-2025-12-05-kln/cv-corpus-24.0-2025-12-05/kln')
train_df = pd.read_csv(DATA_PATH / 'train.tsv', sep='\t', nrows=100)

# Prepare data pairs
data_pairs = []
for idx, row in train_df.iterrows():
    audio_path = DATA_PATH / 'clips' / row['path']
    if audio_path.exists():
        data_pairs.append((str(audio_path), row['sentence']))

print(f"Processing {len(data_pairs)} samples...")

# Initialize pipeline
config = PreprocessingConfig(
    audio=AudioConfig(),
    text=TextConfig(),
    processing=ProcessingConfig()
)
pipeline = PreprocessingPipeline(config)

# Process
results = pipeline.run_pipeline(data_pairs)

print(f"\nResults:")
print(f"  Total: {results['total_samples']}")
print(f"  Valid: {results['valid_samples']}")
print(f"  Success rate: {results['success_rate']:.1%}")
```

**Expected**: 70-90% success rate

---

### Step 3: Inspect Quality (5 minutes)

```python
# Analyze results
analyzer = PerformanceAnalyzer()
analysis = analyzer.analyze_processing_performance(results)

# Print summary
print(json.dumps(analysis['dataset_summary'], indent=2))

# Visualize
analyzer.create_quality_visualizations(results['processed_data'])
```

**Check**:
- Duration distribution looks reasonable
- SNR values are mostly > 10dB
- Text normalization worked correctly

---

### Step 4: Process Full Dataset (30-60 minutes)

```python
# Process all splits
splits = ['train', 'dev', 'test']
all_results = {}

for split in splits:
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print('='*60)
    
    # Load split
    df = pd.read_csv(DATA_PATH / f'{split}.tsv', sep='\t')
    
    # Prepare pairs
    pairs = []
    for idx, row in df.iterrows():
        audio_path = DATA_PATH / 'clips' / row['path']
        if audio_path.exists():
            pairs.append((str(audio_path), row['sentence']))
    
    # Process
    results = pipeline.run_pipeline(pairs)
    all_results[split] = results
    
    print(f"\n{split} complete:")
    print(f"  Valid: {results['valid_samples']}/{results['total_samples']}")
    print(f"  Success: {results['success_rate']:.1%}")
```

---

### Step 5: Export Datasets (10 minutes)

```python
# Create dataset splits
structurer = DatasetStructurer(config.processing)
dataset_dict = structurer.create_huggingface_dataset({
    'train': all_results['train']['processed_data'],
    'validation': all_results['dev']['processed_data'],
    'test': all_results['test']['processed_data']
})

# Export
exporter = DataExporter(config.processing)
output_dir = Path('../processed_data')

# HuggingFace format
exporter.export_huggingface(dataset_dict, output_dir)

# JSON format
exporter.export_json(
    all_results['train']['processed_data'],
    output_dir / 'train.json'
)

# CSV format
exporter.export_csv(
    all_results['train']['processed_data'],
    output_dir / 'train.csv'
)

print(f"\n✓ Data exported to {output_dir}")
```

---

### Step 6: Generate Report (5 minutes)

```python
# Generate comprehensive report
analyzer = PerformanceAnalyzer()

for split, results in all_results.items():
    analysis = analyzer.analyze_processing_performance(results)
    analyzer.generate_report(
        analysis,
        output_dir / f'{split}_report.txt'
    )

print("✓ Reports generated")
```

---

## 📊 Expected Results

### Dataset Statistics (Approximate)

| Split | Original | Valid | Success Rate |
|-------|----------|-------|--------------|
| Train | 11,065 | ~9,000 | 80-85% |
| Dev | 6,412 | ~5,200 | 80-85% |
| Test | 5,685 | ~4,600 | 80-85% |

### Quality Metrics (Expected Ranges)

- **Duration**: 1-15 seconds (mean ~5s)
- **SNR**: 15-35 dB (mean ~25dB)
- **Words/sample**: 3-20 words (mean ~8)
- **Total audio**: ~20-25 hours (after filtering)

---

## 🔍 Quality Checks

### After Processing, Verify:

1. **Audio Quality**
   - [ ] All files are 16kHz mono
   - [ ] No clipping (peak < 1.0)
   - [ ] Silence trimmed
   - [ ] Duration in valid range

2. **Text Quality**
   - [ ] All lowercase
   - [ ] Apostrophes preserved (ng')
   - [ ] No punctuation
   - [ ] No numbers
   - [ ] Valid Kalenjin characters only

3. **Dataset Balance**
   - [ ] Train/dev/test splits maintained
   - [ ] Duration distribution reasonable
   - [ ] No data leakage

---

## 🚨 Troubleshooting

### If success rate < 70%:

1. **Check audio files**: Are they accessible?
2. **Adjust thresholds**: Lower SNR requirement
3. **Check text**: Are there encoding issues?

### If processing is slow:

1. **Reduce n_jobs**: Lower parallel workers
2. **Process in batches**: Split into smaller chunks
3. **Check disk I/O**: Slow storage?

### If memory issues:

1. **Reduce batch_size**: Process fewer at once
2. **Clear cache**: Delete intermediate files
3. **Process splits separately**: One at a time

---

## 📁 Output Structure

After completion, you'll have:

```
processed_data/
├── huggingface_dataset/
│   ├── train/
│   ├── validation/
│   └── test/
├── train.json
├── train.csv
├── train_report.txt
├── dev_report.txt
└── test_report.txt
```

---

## ✅ Success Indicators

You're ready for model training when:

1. ✓ Validation tests all pass
2. ✓ Small batch processes successfully
3. ✓ Quality metrics look reasonable
4. ✓ Full dataset processed
5. ✓ HuggingFace dataset created
6. ✓ Reports generated
7. ✓ ~20+ hours of clean audio

---

## 🎯 Final Checklist

- [ ] Run `test_preprocessing_pipeline.ipynb`
- [ ] Process 100-sample test batch
- [ ] Inspect quality metrics
- [ ] Adjust thresholds if needed
- [ ] Process full train split
- [ ] Process dev split
- [ ] Process test split
- [ ] Export all formats
- [ ] Generate reports
- [ ] Verify output files
- [ ] Ready for model training! 🚀

---

## 📞 Quick Commands

```bash
# Start Jupyter
jupyter notebook

# Check disk space
df -h

# Monitor processing
tail -f preprocessing.log

# Count processed files
ls processed_data/huggingface_dataset/train/ | wc -l
```

---

**Current Status**: Pipeline ready ✓  
**Next Action**: Run test_preprocessing_pipeline.ipynb  
**Time Estimate**: 1-2 hours for full processing
