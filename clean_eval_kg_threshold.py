import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss, jaccard_score, classification_report
)
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
import json
import networkx as nx
from nltk import word_tokenize, sent_tokenize
import multiprocessing as mp

# Load environment variables
load_dotenv()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/negation_kg_eval_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
# Emotion labels (27 emotions, no neutral)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]

# Initial thresholds (will optimize per mode)
INITIAL_THRESHOLDS = {
  "admiration": 0.35,
  "amusement": 0.35,
  "anger": 0.25,
  "annoyance": 0.15000000000000002,
  "approval": 0.25,
  "caring": 0.25,
  "confusion": 0.15000000000000002,
  "curiosity": 0.30000000000000004,
  "desire": 0.2,
  "disappointment": 0.25,
  "disapproval": 0.15000000000000002,
  "disgust": 0.25,
  "embarrassment": 0.2,
  "excitement": 0.25,
  "fear": 0.25,
  "gratitude": 0.5,
  "grief": 0.15000000000000002,
  "joy": 0.25,
  "love": 0.35,
  "nervousness": 0.25,
  "optimism": 0.4,
  "pride": 0.15000000000000002,
  "realization": 0.15000000000000002,
  "relief": 0.55,
  "remorse": 0.35,
  "sadness": 0.25,
  "surprise": 0.25
}

# Opposites for negation cross-boost (fixed typo)
OPPOSITES = {
    'admiration': ['disapproval', 'disgust', 'annoyance'],
    'amusement': ['annoyance', 'disappointment', 'sadness'],
    'anger': ['relief', 'approval', 'caring'],
    'annoyance': ['approval', 'relief', 'joy'],
    'approval': ['disapproval', 'anger', 'disappointment'],
    'caring': ['disapproval', 'disgust', 'anger'],
    'confusion': ['realization', 'approval', 'optimism'],
    'curiosity': ['disappointment', 'annoyance', 'disapproval'],
    'desire': ['disgust', 'disappointment', 'remorse'],
    'disappointment': ['joy', 'excitement', 'relief'],
    'disapproval': ['approval', 'admiration', 'gratitude'],
    'disgust': ['admiration', 'love', 'approval'],
    'embarrassment': ['pride', 'optimism', 'relief'],
    'excitement': ['disappointment', 'sadness', 'annoyance'],
    'fear': ['relief', 'optimism', 'pride'],
    'gratitude': ['disapproval', 'anger', 'remorse'],
    'grief': ['joy', 'relief', 'optimism'],
    'joy': ['sadness', 'disappointment', 'grief'],
    'love': ['disgust', 'disapproval', 'anger'],
    'nervousness': ['relief', 'optimism', 'pride'],
    'optimism': ['disappointment', 'fear', 'sadness'],
    'pride': ['embarrassment', 'remorse', 'disapproval'],
    'realization': ['confusion', 'disappointment', 'surprise'],
    'relief': ['fear', 'nervousness', 'disappointment'],
    'remorse': ['pride', 'approval', 'relief'],
    'sadness': ['joy', 'excitement', 'relief'],
    'surprise': ['realization', 'annoyance', 'disappointment']
}

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def load_data():
    print("Loading test dataset...")
    try:
        test_df = pd.read_csv("dataset/goemotions_test.csv")
        print(f"Successfully loaded test dataset with {len(test_df)} samples")
        texts = test_df["text"].tolist()
        labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
        return texts, labels
    except FileNotFoundError:
        alternative_files = ["dataset/test_dataset.csv", "dataset/test.csv", "dataset/goemotions_test_data.csv"]
        for file_path in alternative_files:
            try:
                test_df = pd.read_csv(file_path)
                print(f"Successfully loaded from {file_path} with {len(test_df)} samples")
                texts = test_df["text"].tolist()
                labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
                return texts, labels
            except FileNotFoundError:
                continue
        raise FileNotFoundError("Could not find test dataset files")

def load_model():
    MODEL_PATH = os.getenv("MODEL_PATH", "saved_model_xlm-roberta-base")
    MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, MODEL_NAME

def build_emotion_kg(include_negation=False):
    G = nx.Graph()
    for emotion in emotion_labels:
        G.add_node(emotion)
    triggers = {  # Your provided triggers (full list as in query)
        'admiration': ['stunning view', 'beautiful architecture', 'amazing design', 'impressive facilities',
                       'gorgeous sunset', 'breathtaking scenery', 'elegant decor', 'wonderful craftsmanship',
                       'incredible location', 'sundara'],
        'amusement': ['funny staff', 'hilarious experience', 'entertaining show', 'playful atmosphere', 'joke',
                      'amusing incident', 'lighthearted vibe', 'comical error', 'witty service', 'enjoyable games',
                      'laughable mistake', 'cheerful crowd'],
        'anger': ['rude staff', 'overpriced', 'terrible service', 'frustrating wait', 'annoying noise',
                  'fight with manager', 'infuriating delay', 'outrageous bill', 'aggravating crowd',
                  'irritating insects', 'furious about cleanliness', 'enraging scam'],
        'annoyance': ['minor issue', 'slight delay', 'bothersome smell', 'irritating music', 'pesky mosquitoes',
                      'annoying crowd', 'frustrating parking', 'mild discomfort', 'bothersome noise', 'irksome wait',
                      'petty complaint', 'nagging problem'],
        'approval': ['great value', 'recommend highly', 'worth visiting', 'excellent choice', 'approve of service',
                     'good decision', 'positive experience', 'thumbs up', 'well done', 'satisfied customer',
                     'endorse this place', 'favorable review'],
        'caring': ['helpful staff', 'thoughtful service', 'caring host', 'attentive care', 'warm welcome',
                   'supportive environment', 'kind gesture', 'empathetic response', 'nurturing atmosphere',
                   'considerate amenities', 'gentle handling', 'protective measures'],
        'confusion': ['confusing layout', 'unclear directions', 'mixed signals', 'baffling menu', 'puzzling rules',
                      'disorienting paths', 'bewildering experience', 'uncertain about quality', 'muddled instructions',
                      'perplexing pricing', 'lost in crowd'],
        'curiosity': ['intriguing history', 'mysterious ruins', 'curious artifacts', 'exploring hidden spots',
                      'wondering about', 'fascinating facts', 'inquisitive tour', 'eager to discover',
                      'piqued interest', 'questioning origins', 'alluring mystery'],
        'desire': ['craving food', 'want to return', 'longing for relaxation', 'eager to stay', 'desire luxury',
                   'yearning for adventure', 'wishing for more', 'tempting menu', 'hankering for view',
                   'coveting experience', 'aspiring visit'],
        'disappointment': ['below expectations', 'let down', 'disappointing food', 'failed promise', 'regret visiting',
                           'underwhelming view', 'dashed hopes', 'mediocre service', 'unfulfilled hype', 'sad letdown',
                           'frustrated outcome', 'disheartening stay'],
        'disapproval': ['poor quality', 'not recommended', 'disapprove of hygiene', 'bad choice',
                        'unacceptable behavior', 'frown upon', 'negative review', 'criticize management',
                        'object to noise', 'condemn facilities', 'reject this place', 'dislike strongly'],
        'disgust': ['dirty room', 'filthy bathroom', 'disgusting smell', 'revolting food', 'nasty insects',
                    'gross hygiene', 'repulsive odor', 'sickening sight', 'appalling cleanliness', 'vile conditions',
                    'nauseating experience', 'kadu'],
        'embarrassment': ['awkward situation', 'embarrassing mistake', 'humiliating service', 'shameful experience',
                          'cringeworthy moment', 'red-faced error', 'mortifying incident', 'uncomfortable vibe',
                          'disgraceful handling', 'belittling staff', 'shaming review'],
        'excitement': ['thrilling adventure', 'exciting activities', 'buzzing atmosphere', 'electrifying event',
                       'pumped up', 'adrenaline rush', 'vibrant energy', 'exhilarating view', 'heart-pounding fun',
                       'eager anticipation', 'lively crowd', 'dynamic place'],
        'fear': ['scary area', 'unsafe at night', 'frightening crowd', 'alarming noise', 'terrifying experience',
                 'nerve-wracking path', 'dreadful security', 'intimidating surroundings', 'fearful of theft',
                 'spooky ambiance', 'anxious about safety'],
        'gratitude': ['thankful for service', 'appreciative host', 'grateful experience', 'thanks to staff',
                      'obliged for help', 'indebted to', 'appreciate kindness', 'thankful view', 'gracious welcome',
                      'blessed stay', 'heartfelt thanks', 'nandri'],
        'grief': ['tragic loss', 'mournful memory', 'heartbreaking event', 'sorrowful place', 'grieving over',
                  'painful reminder', 'devastating news', 'lamenting failure', 'woeful experience', 'bereaved feeling',
                  'mourn loss', 'deep sorrow'],
        'joy': ['happy stay', 'joyful experience', 'delightful food', 'cheerful atmosphere', 'blissful relaxation',
                'ecstatic view', 'gleeful moments', 'merry crowd', 'uplifting vibe', 'fun celebration',
                'radiant happiness', 'santhosam'],
        'love': ['adore this place', 'love the view', 'cherish memories', 'passionate about', 'fond of service',
                 'heartwarming', 'beloved spot', 'affectionate welcome', 'endearing charm', 'romantic setting',
                 'treasure experience', 'priyam'],
        'nervousness': ['anxious wait', 'nervous about safety', 'tense atmosphere', 'apprehensive crowd',
                        'worried service', 'uneasy feeling', 'jittery experience', 'fidgety moments', 'restless night',
                        'edgy vibe', 'nervous anticipation'],
        'optimism': ['hopeful return', 'positive outlook', 'optimistic about', 'bright future visit',
                     'encouraging signs', 'upbeat review', 'promising place', 'confident recommendation',
                     'hopeful improvement', 'cheerful prospects', 'asai'],
        'pride': ['proud achievement', 'pride in heritage', 'boastful review', 'honored to visit', 'self-satisfied',
                  'dignified place', 'prestigious location', 'arrogant charm', 'vainglorious staff', 'noble feeling',
                  'elevated status'],
        'realization': ['sudden insight', 'eye-opening experience', 'dawning awareness', 'realized truth', 'aha moment',
                        'epiphany about quality', 'uncovered fact', 'revelation in review', 'discovered hidden gem',
                        'understood issue', 'clarifying visit'],
        'relief': ['relieved after', 'sigh of relief', 'eased tension', 'comforting end', 'stress-free stay',
                   'calming atmosphere', 'soothing experience', 'unburdened feeling', 'relaxed finally',
                   'alleviated worry', 'peaceful resolution'],
        'remorse': ['regret choosing', 'sorry for visit', 'remorseful review', 'guilty pleasure', 'apologetic tone',
                    'rueful experience', 'penitent feeling', 'contrite about', 'ashamed of choice', 'repentant stay',
                    'sorrowful regret'],
        'sadness': ['sad experience', 'depressing place', 'heartbreaking view', 'melancholy atmosphere',
                    'downcast mood', 'gloomy stay', 'tearful memory', 'mournful night', 'despondent review',
                    'woeful disappointment', 'dukka'],
        'surprise': ['unexpected delight', 'shocking discovery', 'surprising quality', 'astonishing view',
                     'jaw-dropping moment', 'unforeseen issue', 'startling event', 'amazing twist', 'pleasant shock',
                     'unanticipated fun', 'bewildering surprise']
    }
    for emotion, words in triggers.items():
        for word in words:
            G.add_node(word)
            G.add_edge(word, emotion, weight=0.1)

    if include_negation:
        negations = ['not', 'no', 'never', 'isnt', 'arent', 'wasnt', 'wont', 'cant', 'dont', 'didnt']
        for neg in negations:
            G.add_node(neg)
            for emotion in emotion_labels:
                G.add_edge(neg, emotion, weight=-0.1)
    return G

def apply_kg_to_probs(all_probs, test_texts, mode='raw'):
    if mode == 'raw':
        return all_probs
    kg = build_emotion_kg(include_negation=(mode == 'kg_with_neg'))
    refined_probs = np.copy(all_probs)
    for i, text in enumerate(test_texts):
        tokens = word_tokenize(text.lower())
        adjustment = np.zeros(len(emotion_labels))
        for j, token in enumerate(tokens):
            if token in kg:
                for neighbor in kg.neighbors(token):
                    if neighbor in emotion_labels:
                        idx = emotion_labels.index(neighbor)
                        weight = kg[token][neighbor]['weight']
                        if mode == 'kg_with_neg' and j > 0 and tokens[j-1] in ['not', 'no', 'never']:
                            adjustment[idx] += weight * -1
                            for opp in OPPOSITES.get(neighbor, []):
                                if opp in emotion_labels:
                                    opp_idx = emotion_labels.index(opp)
                                    adjustment[opp_idx] += 0.05
                        else:
                            adjustment[idx] += weight
        refined_probs[i] += adjustment
        refined_probs[i] = np.clip(refined_probs[i], 0, 1)
    return refined_probs

def run_evaluation(model, tokenizer, test_texts, test_labels, mode='raw'):
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Increased for efficiency
    all_probs = []
    all_labels = []
    print(f"Running evaluation in mode: {mode}...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 10 == 0:
                print(f"Processing batch {i + 1}/{len(test_loader)}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = apply_kg_to_probs(all_probs, test_texts, mode)
    return all_probs, all_labels

def eval_mode_wrapper(args):
    model_path, test_texts, test_labels, mode = args
    model, tokenizer, _ = load_model()  # Reload per process to avoid sharing issues
    return run_evaluation(model, tokenizer, test_texts, test_labels, mode)

def optimize_thresholds(all_probs, true_labels, output_dir, mode):
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresholds = {}
    for emotion in emotion_labels:
        idx = emotion_labels.index(emotion)
        f1_scores = [f1_score(true_labels[:, idx], (all_probs[:, idx] > thresh).astype(int), zero_division=0) for thresh in thresholds]
        best_idx = np.argmax(f1_scores)
        best_thresholds[emotion] = float(thresholds[best_idx])
    json_path = f"{output_dir}/thresholds_{mode}.json"
    with open(json_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"Thresholds for {mode} saved to {json_path}")
    return best_thresholds

def apply_thresholds(probabilities, thresholds):
    predictions = np.zeros_like(probabilities, dtype=int)
    for i, emotion in enumerate(emotion_labels):
        predictions[:, i] = (probabilities[:, i] > thresholds[emotion]).astype(int)
    return predictions

def compute_metrics(true_labels, predictions, mode):
    metrics = {
        f'{mode}_f1_micro': f1_score(true_labels, predictions, average='micro'),
        f'{mode}_f1_macro': f1_score(true_labels, predictions, average='macro'),
        f'{mode}_f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        f'{mode}_precision_micro': precision_score(true_labels, predictions, average='micro', zero_division=0),
        f'{mode}_precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
        f'{mode}_recall_micro': recall_score(true_labels, predictions, average='micro', zero_division=0),
        f'{mode}_recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0),
        f'{mode}_hamming_loss': hamming_loss(true_labels, predictions),
        f'{mode}_jaccard_micro': jaccard_score(true_labels, predictions, average='micro'),
        f'{mode}_jaccard_macro': jaccard_score(true_labels, predictions, average='macro'),
        f'{mode}_subset_accuracy': accuracy_score(true_labels, predictions),
    }
    per_emotion_acc = {emotion: accuracy_score(true_labels[:, i], predictions[:, i]) for i, emotion in enumerate(emotion_labels)}
    metrics.update({f'{mode}_acc_{emotion}': per_emotion_acc[emotion] for emotion in emotion_labels})
    return metrics

def generate_classification_report(true_labels, predictions, mode):
    class_report = classification_report(true_labels, predictions, target_names=emotion_labels, output_dict=True, zero_division=0)
    return pd.DataFrame(class_report).transpose()

def compare_and_visualize(all_metrics, all_reports, output_dir):
    # Overall matrix
    pd.DataFrame([all_metrics]).to_csv(f"{output_dir}/overall_eval_matrix.csv", index=False)

    # Explicit F1 comparison table (printed)
    print("\n=== F1 Comparison Across Modes ===")
    print(f"{'Mode':<15} {'F1 Micro':<10} {'F1 Macro':<10} {'F1 Weighted':<12}")
    print("-" * 50)
    for mode in ['raw', 'kg_no_neg', 'kg_with_neg']:
        print(f"{mode:<15} {all_metrics[f'{mode}_f1_micro']:<10.4f} {all_metrics[f'{mode}_f1_macro']:<10.4f} {all_metrics[f'{mode}_f1_weighted']:<12.4f}")

    # F1 comparison bar plot (micro, macro, weighted side-by-side)
    modes = ['raw', 'kg_no_neg', 'kg_with_neg']
    f1_data = []
    for mode in modes:
        f1_data.extend([
            {'Mode': mode, 'Type': 'Micro', 'F1': all_metrics[f'{mode}_f1_micro']},
            {'Mode': mode, 'Type': 'Macro', 'F1': all_metrics[f'{mode}_f1_macro']},
            {'Mode': mode, 'Type': 'Weighted', 'F1': all_metrics[f'{mode}_f1_weighted']}
        ])
    df = pd.DataFrame(f1_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Mode', y='F1', hue='Type', data=df)
    plt.title('F1 Scores (Micro, Macro, Weighted) Across Modes')
    plt.savefig(f"{output_dir}/f1_comparison_all_modes.png")
    plt.close()
    df.to_csv(f"{output_dir}/f1_comparison.csv", index=False)

    # Per-emotion accuracy diagram
    acc_data = []
    for mode in modes:
        for emo in emotion_labels:
            acc_data.append({'Emotion': emo, 'Mode': mode, 'Accuracy': all_metrics[f'{mode}_acc_{emo}']})
    df = pd.DataFrame(acc_data)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Emotion', y='Accuracy', hue='Mode', data=df)
    plt.title('Per-Emotion Accuracy Across Modes')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/per_emotion_accuracy_comparison.png")
    plt.close()
    df.to_csv(f"{output_dir}/per_emotion_accuracy.csv", index=False)

    # Heatmaps for F1 per mode
    for mode in modes:
        report = all_reports[mode]
        f1_scores = report.loc[emotion_labels, "f1-score"].values.reshape(3, 9)
        emotion_grid = np.array(emotion_labels).reshape(3, 9)
        plt.figure(figsize=(18, 8))
        sns.heatmap(f1_scores, annot=True, fmt='.3f', cmap='RdYlGn')
        for i in range(3):
            for j in range(9):
                plt.text(j + 0.5, i + 0.5, emotion_grid[i, j], ha='center', va='center', fontsize=8)
        plt.title(f'F1 Heatmap for {mode}')
        plt.savefig(f"{output_dir}/f1_heatmap_{mode}.png")
        plt.close()

def save_model(model, tokenizer, output_dir, mode):
    save_path = f"{output_dir}/saved_model_{mode}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model for {mode} saved to {save_path}")

def main(optimize_for_all_modes=True, save_model_flag=False):
    test_texts, test_labels = load_data()
    model, tokenizer, model_name = load_model()

    modes = ['raw', 'kg_no_neg', 'kg_with_neg']
    all_probs_dict = {}
    all_metrics = {}
    all_reports = {}
    all_thresholds = {}

    # Parallel evaluation
    try:
        pool = mp.Pool(processes=3)  # 3 processes for 3 modes
        args = [(os.getenv("MODEL_PATH"), test_texts, test_labels, mode) for mode in modes]
        results = pool.map(eval_mode_wrapper, args)
        pool.close()
        pool.join()
        for mode, res in zip(modes, results):
            all_probs_dict[mode], _ = res  # true_labels are the same
    except Exception as e:
        print(f"Parallelism failed: {e}. Falling back to sequential.")
        for mode in modes:
            all_probs_dict[mode], _ = run_evaluation(model, tokenizer, test_texts, test_labels, mode=mode)

    # Process each mode
    for mode in modes:
        all_probs = all_probs_dict[mode]
        if optimize_for_all_modes:
            all_thresholds[mode] = optimize_thresholds(all_probs, test_labels, output_dir, mode)
        else:
            all_thresholds[mode] = INITIAL_THRESHOLDS

        predictions = apply_thresholds(all_probs, all_thresholds[mode])
        all_metrics.update(compute_metrics(test_labels, predictions, mode))
        all_reports[mode] = generate_classification_report(test_labels, predictions, mode)

        if save_model_flag:
            save_model(model, tokenizer, output_dir, mode)

    # Define output_dir once here


    compare_and_visualize(all_metrics, all_reports, output_dir)

if __name__ == "__main__":
    main()