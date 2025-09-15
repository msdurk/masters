
import csv, os, re
from typing import List
import pandas as pd

def read_simple_csv(path: str) -> pd.DataFrame:
    # standard CSV with headers "text" and optionally "label"
    return pd.read_csv(path)

def read_problem_llm_phishing(path: str) -> pd.DataFrame:
    # Lines are: <text with many commas>,<label>
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # 'text,label\n'
        for line in f:
            line = line.rstrip('\n')
            if not line: 
                continue
            # split by last comma
            if ',' not in line:
                continue
            text, label = line.rsplit(',', 1)
            texts.append(text)
            try:
                labels.append(int(label))
            except:
                # try to extract final integer
                m = re.search(r'(\d+)\s*$', label)
                labels.append(int(m.group(1)) if m else 1)
    return pd.DataFrame({'text': texts, 'label': labels})

def unify_frames(frames: List[pd.DataFrame], text_col='text', label_col='label') -> pd.DataFrame:
    out = []
    for df in frames:
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if text_col not in df.columns:
            # try to create from subject+body
            subj = df['subject'] if 'subject' in df.columns else ''
            body = df['body'] if 'body' in df.columns else ''
            text = (subj.fillna('') + ' ' + body.fillna('')).astype(str)
        else:
            text = df[text_col].astype(str)
        if label_col not in df.columns:
            # assume legit files are 1, phishing files are 1 in those names; caller will set afterward
            lab = pd.Series([None]*len(text))
        else:
            lab = df[label_col]
        out.append(pd.DataFrame({'text': text, 'label': lab}))
    return pd.concat(out, ignore_index=True)
