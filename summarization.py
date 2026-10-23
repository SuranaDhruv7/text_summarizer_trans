from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf

from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from transformer import Transformer


# Configuration
FOLDER = Path(__file__).resolve().parent
CSV_FILE = FOLDER / "transformer_summarization_dataset.csv"
TOKENIZER_PREFIX = FOLDER / "news_tokenizer"
WEIGHTS_FILE = FOLDER / "summarizer.weights.h5"

VOCAB_SIZE = 1000
ARTICLE_LENGTH = 80
SUMMARY_LENGTH = 40
BATCH_SIZE = 8
EPOCHS = 40
BLEU_SAMPLE_SIZE = 50

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

np.random.seed(42)
tf.random.set_seed(42)


# Load and split data
if not CSV_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_FILE}")

data = pd.read_csv(CSV_FILE)

if not {"paragraph", "summary"}.issubset(data.columns):
    raise ValueError("CSV needs 'paragraph' and 'summary' columns.")

data = data[["paragraph", "summary"]].dropna().copy()
data["paragraph"] = data["paragraph"].astype(str).str.strip()
data["summary"] = data["summary"].astype(str).str.strip()
data = data[
    (data["paragraph"] != "") &
    (data["summary"] != "")
].drop_duplicates().reset_index(drop=True)

if len(data) < 10:
    raise ValueError("At least 10 valid examples are required.")

train_val, test_data = train_test_split(
    data, test_size=0.20, random_state=42
)
train_data, val_data = train_test_split(
    train_val, test_size=0.10, random_state=42
)

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print(
    f"Training: {len(train_data)} | "
    f"Validation: {len(val_data)} | "
    f"Test: {len(test_data)}"
)


# Train SentencePiece tokenizer
corpus_file = FOLDER / "tokenizer_corpus.txt"

with open(corpus_file, "w", encoding="utf-8") as file:
    for text in train_data["paragraph"]:
        file.write(text.replace("\n", " ") + "\n")
    for text in train_data["summary"]:
        file.write(text.replace("\n", " ") + "\n")

spm.SentencePieceTrainer.train(
    input=str(corpus_file),
    model_prefix=str(TOKENIZER_PREFIX),
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    pad_id=PAD_ID,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    hard_vocab_limit=False
)

tokenizer = spm.SentencePieceProcessor(
    model_file=str(FOLDER / "news_tokenizer.model")
)
vocab_size = tokenizer.get_piece_size()


# Prepare data
def prepare_data(frame):
    enc_inputs, dec_inputs, targets = [], [], []

    for _, row in frame.iterrows():
        source = tokenizer.encode(
            row["paragraph"], out_type=int
        )[:ARTICLE_LENGTH]

        summary = tokenizer.encode(
            row["summary"], out_type=int
        )[:SUMMARY_LENGTH - 1]

        decoder_input = [BOS_ID] + summary
        target = summary + [EOS_ID]

        source += [PAD_ID] * (ARTICLE_LENGTH - len(source))
        decoder_input += [PAD_ID] * (
            SUMMARY_LENGTH - len(decoder_input)
        )
        target += [PAD_ID] * (SUMMARY_LENGTH - len(target))

        enc_inputs.append(source)
        dec_inputs.append(decoder_input)
        targets.append(target)

    return (
        np.asarray(enc_inputs, dtype=np.int32),
        np.asarray(dec_inputs, dtype=np.int32),
        np.asarray(targets, dtype=np.int32)
    )


X_train_enc, X_train_dec, y_train = prepare_data(train_data)
X_val_enc, X_val_dec, y_val = prepare_data(val_data)


def make_dataset(enc, dec, targets, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        ((enc, dec), targets)
    )
    if shuffle:
        ds = ds.shuffle(len(enc), seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_dataset = make_dataset(
    X_train_enc, X_train_dec, y_train, shuffle=True
)
val_dataset = make_dataset(X_val_enc, X_val_dec, y_val)


# Build Transformer
model = Transformer(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512,
    max_source_length=ARTICLE_LENGTH,
    max_target_length=SUMMARY_LENGTH,
    pad_id=PAD_ID
)


# Loss and optimizer
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)


def masked_loss(y_true, y_pred):
    loss = loss_function(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), loss.dtype)
    return tf.reduce_sum(loss * mask) / (
        tf.reduce_sum(mask) + 1e-9
    )


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.0
    ),
    loss=masked_loss
)


# Train
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nTraining Transformer...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save_weights(WEIGHTS_FILE)
print("Model weights saved.")


# Find valid first-word tokens
# SentencePiece uses the marker ▁ to indicate a word boundary.
first_word_ids = [
    i for i in range(vocab_size)
    if tokenizer.id_to_piece(i).startswith("▁")
    and tokenizer.id_to_piece(i)[1:2].isupper()
    and i not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
]


# Generate summary
def summarize(paragraph):
    paragraph = str(paragraph).strip()
    if not paragraph:
        return ""

    source = tokenizer.encode(
        paragraph, out_type=int
    )[:ARTICLE_LENGTH]
    source += [PAD_ID] * (ARTICLE_LENGTH - len(source))

    encoder_input = tf.constant([source], dtype=tf.int32)
    generated_ids = [BOS_ID]

    for step in range(SUMMARY_LENGTH - 1):
        decoder_input = tf.constant(
            [generated_ids], dtype=tf.int32
        )

        predictions = model(
            [encoder_input, decoder_input],
            training=False
        )
        logits = predictions[0, -1, :]

        # Block special tokens.
        logits = tf.tensor_scatter_nd_update(
            logits,
            indices=[[PAD_ID], [UNK_ID], [BOS_ID]],
            updates=tf.constant(
                [-1e9, -1e9, -1e9],
                dtype=logits.dtype
            )
        )

        # At the start, prefer a complete capitalized word or
        # a capitalized word-initial subword, not a continuation.
        if step == 0 and first_word_ids:
            allowed = tf.reduce_max(
                tf.one_hot(
                    first_word_ids,
                    depth=vocab_size,
                    on_value=0.0,
                    off_value=-1e9,
                    dtype=logits.dtype
                ),
                axis=0
            )
            constrained_logits = logits + allowed

            # Use the constraint only if it leaves valid candidates.
            if bool(tf.reduce_any(constrained_logits > -1e8).numpy()):
                logits = constrained_logits

        next_token = int(tf.argmax(logits).numpy())

        if next_token == EOS_ID:
            break

        generated_ids.append(next_token)

    summary_ids = [
        token for token in generated_ids[1:]
        if token not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
    ]

    result = tokenizer.decode(summary_ids).strip()

    # Normalize the first character's capitalization without
    # inventing a missing letter or word.
    if result:
        result = result[0].upper() + result[1:]

    return result


# Faster BLEU evaluation on a reproducible test sample
def calculate_bleu(frame):
    sample = frame.sample(
        n=min(BLEU_SAMPLE_SIZE, len(frame)),
        random_state=42
    )

    references, hypotheses = [], []
    smoothing = SmoothingFunction().method1

    print(f"\nCalculating BLEU on {len(sample)} test examples...")

    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        reference = str(row["summary"]).split()
        generated = summarize(row["paragraph"]).split()

        references.append([reference])
        hypotheses.append(generated)

        if index % 10 == 0 or index == len(sample):
            print(f"Progress: {index}/{len(sample)}", flush=True)

    scores = {
        "BLEU-1": corpus_bleu(
            references, hypotheses,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing
        ),
        "BLEU-2": corpus_bleu(
            references, hypotheses,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        ),
        "BLEU-3": corpus_bleu(
            references, hypotheses,
            weights=(1/3, 1/3, 1/3, 0),
            smoothing_function=smoothing
        ),
        "BLEU-4": corpus_bleu(
            references, hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )
    }

    print("\nBLEU SCORES (sample estimate)")
    for name, score in scores.items():
        print(f"{name}: {score:.4f} ({score * 100:.2f}%)")

    return scores


bleu_scores = calculate_bleu(test_data)


# Diagnostic example
sample_row = test_data.iloc[0]
print("\nDIAGNOSTIC EXAMPLE")
print("Paragraph:", sample_row["paragraph"])
print("Reference:", sample_row["summary"])
print("Generated:", summarize(sample_row["paragraph"]))


# Interactive summarization
print("\nEnter a paragraph to summarize.")
print("Type END on a new line to generate the summary.")
print("Type EXIT on a new line to stop.")

while True:
    lines = []
    should_exit = False

    while True:
        line = input()

        if line.strip().upper() == "EXIT":
            should_exit = True
            break

        if line.strip().upper() == "END":
            break

        lines.append(line)

    if should_exit:
        print("\nProgram stopped. Goodbye!")
        break

    paragraph = " ".join(lines).strip()

    if not paragraph:
        print("Please enter a paragraph.")
        continue

    print("\nGenerated Summary:")
    print(summarize(paragraph))
    print("\nEnter another paragraph, or type EXIT to quit.")