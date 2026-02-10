#!/usr/bin/env python3
"""
Notebook Section Builder for Kalenjin ASR Preprocessing Pipeline
Helps generate and insert sections into the Jupyter notebook systematically.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

class NotebookSectionBuilder:
    """Helper class to build and insert notebook sections."""
    
    def __init__(self, notebook_path: str):
        self.notebook_path = Path(notebook_path)
        self.notebook = self._load_notebook()
        
    def _load_notebook(self) -> Dict[str, Any]:
        """Load the notebook JSON."""
        with open(self.notebook_path, 'r') as f:
            return json.load(f)
    
    def _save_notebook(self):
        """Save the notebook JSON."""
        with open(self.notebook_path, 'w') as f:
            json.dump(self.notebook, f, indent=1)
    
    def add_section(self, section_num: int, title: str, markdown_content: str, code_content: str):
        """Add a complete section with markdown and code cells."""
        
        # Create markdown cell
        markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": markdown_content.split('\n')
        }
        
        # Create code cell
        code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_content.split('\n')
        }
        
        # Insert cells at the end
        self.notebook["cells"].extend([markdown_cell, code_cell])
        self._save_notebook()
        print(f"✓ Added Section {section_num}: {title}")

def get_section_4_content():
    """Generate Section 4: Text Normalization Module content."""
    
    markdown_content = """---
## 4. Text Normalization Module <a id='section4'></a>

### Linguistic Theory for Kalenjin

Kalenjin text normalization requires understanding of:
- **Orthographic Variations**: Multiple spelling conventions exist
- **Tone Marking**: Optional tone diacritics in some texts
- **Morphophonemic Processes**: Sound changes at morpheme boundaries
- **Code-switching**: Mixed Kalenjin-English utterances

### Normalization Pipeline
1. **Unicode Normalization**: NFKC form for consistent representation
2. **Case Normalization**: Lowercase conversion with exceptions
3. **Punctuation Handling**: Remove while preserving apostrophes
4. **Orthographic Standardization**: Consistent spelling rules
5. **Quality Filtering**: Remove low-quality transcriptions"""

    code_content = """class TextNormalizer:
    \"\"\"Advanced text normalization for Kalenjin ASR preprocessing.\"\"\"
    
    def __init__(self, config: TextConfig):
        self.config = config
        self.stats = defaultdict(int)
        
        # Kalenjin-specific orthographic mappings
        self.orthographic_rules = {
            'ch': 'c',  # Standardize consonant clusters
            'ng\'': 'ng\'',  # Preserve important apostrophe
            'kh': 'k',  # Simplify aspirated consonants
        }
        
        # Common abbreviations and expansions
        self.expansions = {
            'n': 'na',  # Common conjunction
            'k': 'ko',  # Common preposition
        }
        
        logger.info("TextNormalizer initialized for Kalenjin")
    
    def normalize_unicode(self, text: str) -> str:
        \"\"\"Normalize Unicode representation.\"\"\"
        # NFKC normalization for consistent representation
        normalized = unicodedata.normalize('NFKC', text)
        
        # Remove zero-width characters
        normalized = re.sub(r'[\u200b-\u200f\ufeff]', '', normalized)
        
        return normalized
    
    def normalize_case(self, text: str) -> str:
        \"\"\"Apply case normalization with language-specific rules.\"\"\"
        if self.config.lowercase:
            return text.lower()
        return text
    
    def clean_punctuation(self, text: str) -> str:
        \"\"\"Remove punctuation while preserving important markers.\"\"\"
        if not self.config.remove_punctuation:
            return text
        
        # Preserve apostrophes if configured
        if self.config.preserve_apostrophes:
            # Replace other punctuation but keep apostrophes
            text = re.sub(r"[^\w\s']", ' ', text)
        else:
            # Remove all punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    def apply_orthographic_rules(self, text: str) -> str:
        \"\"\"Apply Kalenjin-specific orthographic standardization.\"\"\"
        if not self.config.standardize_orthography:
            return text
        
        # Apply orthographic mappings
        for old, new in self.orthographic_rules.items():
            text = text.replace(old, new)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        \"\"\"Expand common abbreviations.\"\"\"
        words = text.split()
        expanded_words = []
        
        for word in words:
            if word in self.expansions:
                expanded_words.append(self.expansions[word])
                self.stats['abbreviations_expanded'] += 1
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def filter_characters(self, text: str) -> str:
        \"\"\"Filter to allowed character set.\"\"\"
        # Create allowed character set
        allowed = set(self.config.allowed_chars)
        if self.config.preserve_apostrophes:
            allowed.add("'")
        
        # Filter characters
        filtered_chars = [c for c in text if c in allowed]
        filtered_text = ''.join(filtered_chars)
        
        # Normalize whitespace
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        
        return filtered_text
    
    def detect_repetitions(self, text: str) -> int:
        \"\"\"Detect excessive character repetitions.\"\"\"
        max_repetition = 0
        current_char = ''
        current_count = 0
        
        for char in text:
            if char == current_char:
                current_count += 1
                max_repetition = max(max_repetition, current_count)
            else:
                current_char = char
                current_count = 1
        
        return max_repetition
    
    def calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        \"\"\"Calculate comprehensive text quality metrics.\"\"\"
        words = text.split()
        
        metrics = {
            'char_count': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'max_repetition': self.detect_repetitions(text),
            'unique_chars': len(set(text.lower())),
            'whitespace_ratio': text.count(' ') / len(text) if text else 0,
        }
        
        return metrics
    
    def is_valid_text(self, text: str, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        \"\"\"Validate text quality against configured thresholds.\"\"\"
        issues = []
        
        # Word count checks
        if metrics['word_count'] < self.config.min_words:
            issues.append(f"Too few words: {metrics['word_count']} < {self.config.min_words}")
        
        if metrics['word_count'] > self.config.max_words:
            issues.append(f"Too many words: {metrics['word_count']} > {self.config.max_words}")
        
        # Character repetition check
        if metrics['max_repetition'] > self.config.max_char_repetition:
            issues.append(f"Excessive repetition: {metrics['max_repetition']} > {self.config.max_char_repetition}")
        
        # Empty text check
        if not text.strip():
            issues.append("Empty text after normalization")
        
        return len(issues) == 0, issues
    
    def normalize_text(self, text: str) -> Dict[str, Any]:
        \"\"\"Complete text normalization pipeline.\"\"\"
        if not isinstance(text, str):
            return {
                'original': text,
                'normalized': '',
                'is_valid': False,
                'issues': ['Input is not a string'],
                'metrics': {}
            }
        
        original_text = text
        
        try:
            # Normalization pipeline
            text = self.normalize_unicode(text)
            text = self.normalize_case(text)
            text = self.clean_punctuation(text)
            text = self.apply_orthographic_rules(text)
            text = self.expand_abbreviations(text)
            text = self.filter_characters(text)
            
            # Calculate metrics
            metrics = self.calculate_text_metrics(text)
            
            # Validate
            is_valid, issues = self.is_valid_text(text, metrics)
            
            # Update statistics
            self.stats['total_processed'] += 1
            if is_valid:
                self.stats['valid_texts'] += 1
            else:
                self.stats['invalid_texts'] += 1
            
            return {
                'original': original_text,
                'normalized': text,
                'is_valid': is_valid,
                'issues': issues,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error normalizing text '{text[:50]}...': {e}")
            return {
                'original': original_text,
                'normalized': '',
                'is_valid': False,
                'issues': [f'Processing error: {str(e)}'],
                'metrics': {}
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        \"\"\"Get text processing statistics.\"\"\"
        total = self.stats['total_processed']
        if total == 0:
            return {'message': 'No texts processed yet'}
        
        return {
            'total_processed': total,
            'valid_texts': self.stats['valid_texts'],
            'invalid_texts': self.stats['invalid_texts'],
            'validity_rate': self.stats['valid_texts'] / total,
            'abbreviations_expanded': self.stats['abbreviations_expanded']
        }

print('✓ TextNormalizer class defined')
print('✓ Supports Kalenjin-specific orthographic rules')
print('✓ Includes comprehensive text quality validation')"""

    return markdown_content, code_content

if __name__ == "__main__":
    # Quick add Section 4
    notebook_path = "/home/obote/Documents/SE/AOB/2026/Kalenjin ASR/notebooks/kalenjin_asr_preprocessing_pipeline.ipynb"
    builder = NotebookSectionBuilder(notebook_path)
    
    markdown, code = get_section_4_content()
    builder.add_section(4, "Text Normalization Module", markdown, code)
    
    print("✓ Section 4 added to notebook")