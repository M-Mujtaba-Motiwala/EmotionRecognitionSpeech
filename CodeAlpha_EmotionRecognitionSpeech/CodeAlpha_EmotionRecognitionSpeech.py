from IPython.display import display
import os, re, glob, warnings, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa, librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")
SR = 16000

EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

rows = []
for path in sorted(glob.glob("data/RAVDESS_Actors_*/Actor_*/*.wav")):
    p = os.path.basename(path)[:-4].split("-")
    if len(p) != 7 or p[0] != "03" or p[1] != "01":
        continue
    rows.append({"path": path, "emotion": EMOTIONS[p[2]], "intensity": "normal" if p[3] == "01" else "strong",
                 "statement": p[4], "actor": int(p[6]), "gender": "male" if int(p[6]) % 2 else "female"})
meta = pd.DataFrame(rows)
print(len(meta), "clips |", meta["actor"].nunique(), "actors")
display(pd.crosstab(meta["emotion"], meta["actor"], margins=True))
meta.head()

demo_emotions = ["neutral", "happy", "sad", "angry"]
fig, axes = plt.subplots(3, 4, figsize=(18, 9))
for j, emo in enumerate(demo_emotions):
    r = meta[(meta.emotion == emo) & (meta.actor == 1) & (meta.statement == "01")].iloc[0]
    y, _ = librosa.load(r.path, sr=SR)
    librosa.display.waveshow(y, sr=SR, ax=axes[0, j], color="#4C72B0"); axes[0, j].set_title(f"{emo} — waveform")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64), ref=np.max)
    librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="mel", ax=axes[1, j], cmap="magma"); axes[1, j].set_title("mel spectrogram")
    M = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    librosa.display.specshow(M, sr=SR, x_axis="time", ax=axes[2, j], cmap="coolwarm"); axes[2, j].set_title("40 MFCCs")
plt.tight_layout(); plt.show()

N_MFCC, HOP, T_FRAMES = 40, 256, 200

def frame_features(y: np.ndarray) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=30)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=HOP)
    d1, d2 = librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=HOP)
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=512, hop_length=HOP) / (SR / 2)
    return np.vstack([mfcc, d1, d2, rms, zcr, cen]).T.astype("float32")

def pad_trunc(F: np.ndarray, T=T_FRAMES) -> np.ndarray:
    if len(F) >= T:
        return F[:T]
    return np.vstack([F, np.zeros((T - len(F), F.shape[1]), dtype="float32")])

def utterance_stats(F: np.ndarray) -> np.ndarray:
    return np.concatenate([F.mean(axis=0), F.std(axis=0)])

def augment(y: np.ndarray, kind: str, rng) -> np.ndarray:
    if kind == "noise":
        return y + rng.normal(0, 0.005 + 0.01 * rng.random(), size=y.shape).astype(y.dtype) * max(np.abs(y).max(), 1e-3) * 5
    if kind == "pitch":
        return librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.choice([-2, -1, 1, 2]))
    if kind == "stretch":
        return librosa.effects.time_stretch(y, rate=rng.choice([0.9, 1.1]))
    raise ValueError(kind)

rng = np.random.default_rng(SEED)
t0 = time.time()
seq_orig, seq_aug, aug_owner = [], [], []
for i, path in enumerate(meta["path"]):
    y, _ = librosa.load(path, sr=SR)
    seq_orig.append(pad_trunc(frame_features(y)))
    for kind in rng.choice(["noise", "pitch", "stretch"], size=2, replace=False):
        seq_aug.append(pad_trunc(frame_features(augment(y, kind, rng)))); aug_owner.append(i)

X_orig = np.stack(seq_orig); X_aug = np.stack(seq_aug); aug_owner = np.array(aug_owner)
le = LabelEncoder().fit(meta["emotion"])
y_all = le.transform(meta["emotion"]); CLASSES = list(le.classes_)
actors = meta["actor"].values
print(f"original {X_orig.shape}, augmented {X_aug.shape}   ({time.time() - t0:.0f}s)")
print("classes:", CLASSES)

ALL_ACTORS = sorted(meta["actor"].unique())
FOLDS = [tuple(ALL_ACTORS[i:i + 2]) for i in range(0, len(ALL_ACTORS) - len(ALL_ACTORS) % 2, 2)]
print("Held-out actors per fold:", FOLDS)

def fold_indices(test_actors):
    test_mask = np.isin(actors, test_actors)
    tr = np.where(~test_mask)[0]; te = np.where(test_mask)[0]
    aug_idx = np.where(~np.isin(actors[aug_owner], test_actors))[0]
    return tr, te, aug_idx

def stats_matrix(X):
    out = []
    for seq in X:
        valid = np.abs(seq).sum(axis=1) > 0
        out.append(utterance_stats(seq[valid] if valid.any() else seq))
    return np.stack(out)

S_orig, S_aug = stats_matrix(X_orig), stats_matrix(X_aug)

baseline_rows, baseline_pred = [], {"SVM (RBF)": np.zeros(len(y_all), int), "Random Forest": np.zeros(len(y_all), int)}
for fold, test_actors in enumerate(FOLDS, 1):
    tr, te, aug_idx = fold_indices(test_actors)
    Xtr = np.vstack([S_orig[tr], S_aug[aug_idx]]); ytr = np.concatenate([y_all[tr], y_all[aug_owner[aug_idx]]])
    sc = StandardScaler().fit(Xtr)
    for name, clf in {"SVM (RBF)": SVC(C=10, gamma="scale", random_state=SEED),
                      "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)}.items():
        clf.fit(sc.transform(Xtr), ytr)
        p = clf.predict(sc.transform(S_orig[te])); baseline_pred[name][te] = p
        baseline_rows.append({"fold": fold, "model": name, "accuracy": accuracy_score(y_all[te], p),
                              "macro_F1": f1_score(y_all[te], p, average="macro")})
baseline_df = pd.DataFrame(baseline_rows)
baseline_df.pivot(index="model", columns="fold", values="accuracy").assign(mean=lambda d: d.mean(axis=1)).round(3)

n_feat = X_orig.shape[2]
NUM_CLASSES = len(CLASSES)

def build_cnn():
    inp = keras.Input(shape=(T_FRAMES, n_feat))
    x = inp
    for f in (32, 64, 64):
        x = layers.Conv1D(f, 5, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x); x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.4)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn1d")

def build_bilstm():
    inp = keras.Input(shape=(T_FRAMES, n_feat))
    x = layers.SpatialDropout1D(0.1)(inp)
    x = layers.Conv1D(48, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True, dropout=0.2, recurrent_dropout=0.0))(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(48, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="bilstm")

BUILDERS = {"CNN": build_cnn, "BiLSTM": build_bilstm}
BUILDERS["CNN"]().summary()

N_CEPSTRAL = 3 * N_MFCC

def cepstral_mean_subtraction(X):
    X = X.copy()
    valid = (np.abs(X).sum(axis=2) > 0)
    for i in range(len(X)):
        v = valid[i]
        X[i, v, :N_CEPSTRAL] -= X[i, v, :N_CEPSTRAL].mean(axis=0)
    return X * valid[..., None]

EPOCHS = 40

def fit_normaliser(X_train_raw):
    Z = cepstral_mean_subtraction(X_train_raw)
    return Z.reshape(-1, n_feat).mean(axis=0), Z.reshape(-1, n_feat).std(axis=0) + 1e-6

def apply_normaliser(X_raw, mu, sd):
    Z = cepstral_mean_subtraction(X_raw)
    return ((Z - mu) / sd) * (np.abs(X_raw).sum(axis=2, keepdims=True) > 0)

def compile_model(model_name):
    model = BUILDERS[model_name]()
    if model_name == "CNN":
        opt = keras.optimizers.Adam(1e-3)
    else:
        opt = keras.optimizers.AdamW(keras.optimizers.schedules.CosineDecay(2e-3, decay_steps=EPOCHS * 30, alpha=0.05), weight_decay=1e-3)
    model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

def train_fold(model_name, test_actors, epochs=EPOCHS, verbose=0):
    tr, te, aug_idx = fold_indices(test_actors)
    Xtr_raw = np.concatenate([X_orig[tr], X_aug[aug_idx]]); ytr = np.concatenate([y_all[tr], y_all[aug_owner[aug_idx]]])
    mu, sd = fit_normaliser(Xtr_raw)
    model = compile_model(model_name)
    hist = model.fit(apply_normaliser(Xtr_raw, mu, sd), ytr, epochs=epochs, batch_size=32, verbose=verbose)
    proba = model.predict(apply_normaliser(X_orig[te], mu, sd), verbose=0)
    return te, proba, hist.history, epochs

deep_pred = {m: np.zeros(len(y_all), int) for m in BUILDERS}
deep_hist, deep_rows = {m: [] for m in BUILDERS}, []
for model_name in BUILDERS:
    for fold, test_actors in enumerate(FOLDS, 1):
        t0 = time.time()
        te, proba, hist, n_ep = train_fold(model_name, test_actors)
        pred = proba.argmax(axis=1); deep_pred[model_name][te] = pred; deep_hist[model_name].append(hist)
        acc = accuracy_score(y_all[te], pred)
        deep_rows.append({"fold": fold, "model": model_name, "accuracy": acc, "macro_F1": f1_score(y_all[te], pred, average="macro")})
        print(f"{model_name:7s} fold {fold} (held-out actors {test_actors}): accuracy {acc:.3f}  [{n_ep} epochs, {time.time() - t0:.0f}s]")
deep_df = pd.DataFrame(deep_rows)

all_df = pd.concat([baseline_df, deep_df])
summary = all_df.groupby("model").agg(accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"), macro_F1=("macro_F1", "mean")).round(3)
summary = summary.sort_values("accuracy", ascending=False)
print(f"Chance level with 8 classes = {1 / NUM_CLASSES:.3f}")
display(summary)

fig, ax = plt.subplots(figsize=(9, 4.2))
sns.barplot(data=all_df, x="model", y="accuracy", order=summary.index, errorbar="sd", color="#4C72B0", ax=ax, capsize=.15)
sns.stripplot(data=all_df, x="model", y="accuracy", order=summary.index, color="k", size=6, ax=ax)
ax.axhline(1 / NUM_CLASSES, ls="--", c="#C44E52"); ax.text(3.4, 1 / NUM_CLASSES + .01, "chance", color="#C44E52", ha="right")
ax.set_ylim(0, 1); ax.set_title("Speaker-independent accuracy (dots = individual folds)"); plt.tight_layout(); plt.show()

best_model = summary.index[0]
all_preds = {**baseline_pred, **deep_pred}
print("Best model:", best_model, "\n")
print(classification_report(y_all, all_preds[best_model], target_names=CLASSES, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(y_all, all_preds[best_model], display_labels=CLASSES, cmap="Blues", colorbar=False, ax=axes[0], xticks_rotation=45)
axes[0].set_title(f"{best_model}: confusion matrix (all held-out predictions)")
ConfusionMatrixDisplay.from_predictions(y_all, all_preds[best_model], display_labels=CLASSES, cmap="Blues", colorbar=False,
                                        normalize="true", values_format=".0%", ax=axes[1], xticks_rotation=45)
axes[1].set_title("Row-normalised (recall per emotion)")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=False)
for r, model_name in enumerate(BUILDERS):
    for c, h in enumerate(deep_hist[model_name]):
        axes[r, c].plot(h["accuracy"], label="train accuracy (augmented)", color="#4C72B0")
        axes[r, c].plot(h["loss"], label="train loss", color="#C44E52")
        axes[r, c].set_title(f"{model_name} — fold {c + 1}"); axes[r, c].set_xlabel("epoch")
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()

deep_best = max(BUILDERS, key=lambda m: summary.loc[m, "accuracy"])
print("Deep architecture used for the final model:", deep_best)

Xfull = np.concatenate([X_orig, X_aug]); yfull = np.concatenate([y_all, y_all[aug_owner]])
MU, SD = fit_normaliser(Xfull)
_prep_final = lambda X: apply_normaliser(X, MU, SD)

final_model = compile_model(deep_best)
final_model.fit(_prep_final(Xfull), yfull, epochs=EPOCHS, batch_size=32, verbose=0)

def predict_emotion(path, show=True):
    y, _ = librosa.load(path, sr=SR)
    x = pad_trunc(frame_features(y))[None]
    p = final_model.predict(_prep_final(x), verbose=0)[0]
    result = dict(sorted(zip(CLASSES, p.round(3)), key=lambda kv: -kv[1]))
    if show:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(list(result.keys())[::-1], list(result.values())[::-1], color="#4C72B0"); ax.set_xlim(0, 1)
        ax.set_title(f"Predicted emotion: {next(iter(result))}"); plt.tight_layout(); plt.show()
    return result

sample = meta.sample(1, random_state=7).iloc[0]
print("True emotion:", sample.emotion, "| file:", os.path.basename(sample.path))
print(predict_emotion(sample.path))

os.makedirs("outputs", exist_ok=True)
final_model.save("outputs/emotion_" + deep_best.lower() + ".keras")
np.savez("outputs/normalisation.npz", mu=MU, sd=SD, classes=np.array(CLASSES))
print("Saved model + normalisation statistics to ./outputs")
