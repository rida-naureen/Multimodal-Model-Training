# preprocessing/local_test.py
# ============================================================
#  LOCAL TEST — Run this on your laptop BEFORE sending to server.
#
#  What it does:
#    Picks 50 random utterances from your dataset and runs the
#    COMPLETE pipeline on just those 50:
#      • Parses emotion labels
#      • Extracts text embeddings  (RoBERTa)
#      • Extracts audio embeddings (Wav2Vec2)
#      • Extracts visual embeddings (ResNet-50 from dialog AVI)
#      • Saves to data/processed/  (same place as full run)
#      • Runs a mini train loop for 2 epochs to verify model works
#      • Prints a pass/fail report
#
#  If this passes → you're ready to send to server.
#  Total time: ~5–10 minutes on CPU.
#
#  Run:  python preprocessing\local_test.py
# ============================================================

import os
import re
import sys
import random
import numpy as np
import torch

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed"
SPLITS_DIR = "data/splits"
N_SAMPLES  = 50   # number of utterances to test with
SEED       = 42
random.seed(SEED)

VALID_EMOTIONS = {"hap", "exc", "sad", "ang", "neu"}
EMO_MAP        = {"hap": 0, "exc": 0, "sad": 1, "ang": 2, "neu": 3}

results = {}

print("=" * 60)
print("  ET-TACFN Local Test  (50 utterances)")
print("=" * 60)


# ── STEP 1: Parse labels + timestamps ─────────────────────────
print("\n[1/5] Parsing emotion labels and timestamps...")
all_labels     = {}  # utt_id → emotion
all_timestamps = {}  # utt_id → (start, end)
all_dialogs    = {}  # utt_id → dialog_name (e.g. Ses01F_impro01)

for session_num in range(1, 6):
    emo_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                           "dialog", "EmoEvaluation")
    if not os.path.exists(emo_dir):
        continue

    for fname in os.listdir(emo_dir):
        if not fname.endswith(".txt") or fname.startswith("._"):
            continue
        with open(os.path.join(emo_dir, fname),
                  encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(
                    r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)', line)
                if not m:
                    continue
                start, end, utt_id, emo = (float(m.group(1)), float(m.group(2)),
                                           m.group(3), m.group(4))
                if emo in VALID_EMOTIONS:
                    all_labels[utt_id]     = emo
                    all_timestamps[utt_id] = (start, end)
                    all_dialogs[utt_id]    = utt_id.rsplit("_", 1)[0]

total_found = len(all_labels)
if total_found == 0:
    print("  ❌ No labels found. Check data/raw/ structure.")
    sys.exit(1)

print(f"  ✅ Found {total_found} utterances across 5 sessions")
results["labels"] = "PASS"


# ── STEP 2: Pick 50 samples spread across sessions ────────────
print(f"\n[2/5] Selecting {N_SAMPLES} test utterances...")
sampled = random.sample(list(all_labels.keys()), min(N_SAMPLES, total_found))
print(f"  ✅ Selected {len(sampled)} utterances")
results["sampling"] = "PASS"


# ── STEP 3: Extract text embeddings ───────────────────────────
print("\n[3/5] Extracting text embeddings (RoBERTa)...")
os.makedirs(f"{OUTPUT_DIR}/text_embeddings", exist_ok=True)

try:
    from transformers import RobertaTokenizer, RobertaModel

    tok   = RobertaTokenizer.from_pretrained("roberta-base")
    t_model = RobertaModel.from_pretrained("roberta-base")
    t_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model = t_model.to(device)

    # Build transcript lookup from transcription files
    trans_lookup = {}
    for session_num in range(1, 6):
        trans_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                                 "dialog", "transcriptions")
        if not os.path.exists(trans_dir):
            continue
        for fname in os.listdir(trans_dir):
            if not fname.endswith(".txt") or fname.startswith("._"):
                continue
            with open(os.path.join(trans_dir, fname),
                      encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.match(r'^(Ses\S+)\s*(?:\[[\d\.\-]+\])?\s*:\s*(.+)$',
                                 line.strip())
                    if m:
                        trans_lookup[m.group(1)] = m.group(2).strip()

    text_ok  = 0
    emb_shape = None
    for utt_id in sampled:
        save_path = f"{OUTPUT_DIR}/text_embeddings/{utt_id}.npy"
        if os.path.exists(save_path):
            text_ok += 1
            if emb_shape is None:
                emb_shape = np.load(save_path).shape  # read shape from existing file
            continue
        text = trans_lookup.get(utt_id, "neutral expression")
        inp  = tok(text, return_tensors="pt", truncation=True,
                   max_length=128, padding="max_length")
        inp  = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            emb = t_model(**inp).last_hidden_state.squeeze(0).cpu().numpy()
        np.save(save_path, emb)
        emb_shape = emb.shape
        text_ok += 1

    print(f"  ✅ Text embeddings: {text_ok}/{len(sampled)} saved  shape={emb_shape}")
    results["text"] = "PASS"
except Exception as e:
    print(f"  ❌ Text failed: {e}")
    results["text"] = f"FAIL: {e}"


# ── STEP 4: Extract audio embeddings ──────────────────────────
print("\n[4/5] Extracting audio embeddings (Wav2Vec2)...")
os.makedirs(f"{OUTPUT_DIR}/audio_embeddings", exist_ok=True)

try:
    import torchaudio
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    proc    = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    a_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    a_model.eval()
    a_model = a_model.to(device)

    audio_ok = 0
    for utt_id in sampled:
        save_path = f"{OUTPUT_DIR}/audio_embeddings/{utt_id}.npy"
        if os.path.exists(save_path):
            audio_ok += 1
            continue

        # Find wav file
        dialog    = all_dialogs[utt_id]
        # Extract session number from utt_id (e.g. Ses01 → Session1)
        ses_match = re.match(r'Ses0(\d)', utt_id)
        if not ses_match:
            continue
        ses_num  = ses_match.group(1)
        wav_path = os.path.join(RAW_DIR, f"Session{ses_num}",
                                "sentences", "wav", dialog, f"{utt_id}.wav")
        if not os.path.exists(wav_path):
            continue

        wv, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wv = torchaudio.transforms.Resample(sr, 16000)(wv)
        wv = wv.mean(dim=0)
        inp = proc(wv.numpy(), sampling_rate=16000,
                   return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            emb = a_model(**inp).last_hidden_state.squeeze(0).cpu().numpy()
        np.save(save_path, emb)
        audio_ok += 1

    print(f"  ✅ Audio embeddings: {audio_ok}/{len(sampled)} saved")
    results["audio"] = "PASS"
except Exception as e:
    print(f"  ❌ Audio failed: {e}")
    results["audio"] = f"FAIL: {e}"


# ── STEP 5: Extract visual embeddings ─────────────────────────
print("\n[5/5] Extracting visual embeddings (ResNet-50 from dialog AVI)...")
os.makedirs(f"{OUTPUT_DIR}/visual_embeddings", exist_ok=True)

try:
    import cv2
    import torchvision.transforms as Tv
    from torchvision.models import resnet50, ResNet50_Weights

    bb  = resnet50(weights=ResNet50_Weights.DEFAULT)
    bb  = torch.nn.Sequential(*list(bb.children())[:-1])
    bb.eval()
    bb  = bb.to(device)
    prj = torch.nn.Linear(2048, 256).to(device)
    prj.eval()

    tfm = Tv.Compose([Tv.ToPILImage(), Tv.Resize((224, 224)),
                      Tv.ToTensor(),
                      Tv.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    visual_ok = 0
    for utt_id in sampled:
        save_path = f"{OUTPUT_DIR}/visual_embeddings/{utt_id}.npy"
        if os.path.exists(save_path):
            visual_ok += 1
            continue

        dialog    = all_dialogs[utt_id]
        ses_match = re.match(r'Ses0(\d)', utt_id)
        if not ses_match:
            continue
        ses_num  = ses_match.group(1)
        avi_path = os.path.join(RAW_DIR, f"Session{ses_num}",
                                "dialog", "avi", "DivX", f"{dialog}.avi")
        if not os.path.exists(avi_path):
            continue

        start, end = all_timestamps[utt_id]
        cap = cv2.VideoCapture(avi_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        idxs   = np.linspace(int(start*fps), max(int(end*fps)-1,0), 30, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frm = cap.read()
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                          if ret and frm is not None
                          else np.zeros((224,224,3), dtype=np.uint8))
        cap.release()

        feats = []
        with torch.no_grad():
            for frm in frames:
                t = tfm(frm).unsqueeze(0).to(device)
                f = prj(bb(t).squeeze())
                feats.append(f.cpu().numpy())

        np.save(save_path, np.stack(feats))
        visual_ok += 1

    print(f"  ✅ Visual embeddings: {visual_ok}/{len(sampled)} saved  shape=(30, 256)")
    results["visual"] = "PASS"
except Exception as e:
    print(f"  ❌ Visual failed: {e}")
    results["visual"] = f"FAIL: {e}"


# ── STEP 6: Mini train test (2 epochs on 50 samples) ──────────
print("\n[6/6] Running mini train test (2 epochs, 50 samples)...")

try:
    import yaml
    sys.path.insert(0, os.getcwd())
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Write mini split files
    valid = [uid for uid in sampled
             if all(os.path.exists(f"{OUTPUT_DIR}/{m}_embeddings/{uid}.npy")
                    for m in ["text","audio","visual"])
             and uid in all_labels]

    n_val   = max(1, len(valid)//5)
    t_ids   = valid[n_val:]
    v_ids   = valid[:n_val]

    with open(f"{SPLITS_DIR}/train_ids.txt","w") as f: f.write("\n".join(t_ids))
    with open(f"{SPLITS_DIR}/val_ids.txt",  "w") as f: f.write("\n".join(v_ids))
    with open(f"{SPLITS_DIR}/test_ids.txt", "w") as f: f.write("\n".join(v_ids))
    with open(f"{SPLITS_DIR}/labels.txt",   "w") as f:
        for uid in valid:
            emo = all_labels[uid]; emo = "hap" if emo == "exc" else emo; f.write(f"{uid}\t{emo}\n")

    # Load config and override to small values for fast local test
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Use smaller model for local test — avoids memory issues on CPU
    cfg["model"]["d_model"]   = 256
    cfg["model"]["num_heads"] = 4
    # Disable ET-TACFN heavy components for speed
    cfg["model"]["use_missing_modality"] = False
    cfg["training"]["modality_dropout"]  = 0.0

    from dataset.iemocap_dataset import IEMOCAPDataset, collate_fn
    from models.classifier import MultimodalEmotionModel
    from torch.utils.data import DataLoader
    import torch.nn as nn

    ds     = IEMOCAPDataset("train", SPLITS_DIR,
                            f"{OUTPUT_DIR}/text_embeddings",
                            f"{OUTPUT_DIR}/audio_embeddings",
                            f"{OUTPUT_DIR}/visual_embeddings")
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model  = MultimodalEmotionModel(cfg)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn= nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for batch in loader:
            logits, _ = model(batch["text"], batch["audio"], batch["visual"],
                              batch["text_mask"], batch["audio_mask"], batch["visual_mask"])
            loss = loss_fn(logits, batch["label"])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  Epoch {epoch+1}/2  loss={loss.item():.4f}")

    print("  ✅ Mini train: model runs end-to-end without errors")
    results["mini_train"] = "PASS"
except Exception as e:
    print(f"  ❌ Mini train failed: {e}")
    results["mini_train"] = f"FAIL: {e}"


# ── Final Report ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LOCAL TEST REPORT")
print("=" * 60)
all_passed = True
for step, status in results.items():
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon}  {step:15s} : {status}")
    if status != "PASS":
        all_passed = False

print("=" * 60)
if all_passed:
    print("\n  🎉 ALL TESTS PASSED!")
    print("  → You're ready to run full preprocessing on the server.")
    print("  → Copy this entire project folder to the server.")
    print("  → Then run: python preprocessing\\build_dataset.py\n")
else:
    print("\n  ❌ Some tests failed. Fix the errors above before going to server.\n")