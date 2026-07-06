"""
DROP-IN REPLACEMENT for the influence + main-loop sections of the
original script. Place after the `make_resnet50_ntk`, `finetune`,
and `test_accuracy` definitions, replacing everything from
"compute_sketch_matrix" through the JSON save.

Key change:
    Old: gradient sketched at y_i (training true class)
    New: gradient sketched at y_test (test class) for each test point

Result:
    Different-class examples appear in the ranking with meaningful
    magnitude. Top harmful = different-class examples whose gradients
    align with the test gradient at the test class.
"""

import os, math, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Setup: shared parameter subsample mask and JL projection
# ------------------------------------------------------------------ #
def init_sketch_basis(model):
    """
    One-time setup: random parameter subsample mask + JL matrix.
    Reused across all test points to keep sketches comparable.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    set_seed()
    P_eff = int(min(1.0, TARGET_PARAMS / P) * P)
    perm  = torch.randperm(P)[:P_eff].sort()[0]
    Omega = torch.randn(P_eff, SKETCH_DIM,
                        device=device) / math.sqrt(SKETCH_DIM)
    print(f"  [NTK basis] params={P:,} subsampled={P_eff:,} "
          f"s={SKETCH_DIM}")
    return params, perm, Omega

def sketch_at_class(model, pixels_1, params, perm, Omega, target_class):
    """
    Single-image JL sketch of grad(f(x)[target_class]) w.r.t. params.
    """
    model.zero_grad()
    logits = model(pixels_1)
    logits[0, target_class].backward()
    g = torch.cat([p.grad.detach().view(-1) for p in params])
    return (g[perm] @ Omega).cpu()

# ------------------------------------------------------------------ #
#  Compute training sketches AT A GIVEN TEST CLASS
# ------------------------------------------------------------------ #
def compute_train_sketches_at_class(model, target_class,
                                    params, perm, Omega):
    """
    Sketch every training example's gradient at logit[target_class].
    Returns G_t of shape (n_train, s).
    """
    model.eval()
    loader = DataLoader(MASTER_DS, batch_size=1,
                        shuffle=False, num_workers=2)
    sketches = []
    for pixels, _ in tqdm(loader,
                          desc=f"  Sketching @ class {target_class:>2}",
                          leave=False):
        pixels = pixels.to(device)
        sketches.append(sketch_at_class(model, pixels,
                                         params, perm, Omega,
                                         target_class))
        if len(sketches) % 500 == 0:
            torch.cuda.empty_cache()
    return torch.stack(sketches)

# ------------------------------------------------------------------ #
#  MAIN PIPELINE (replaces the previous influence section)
# ------------------------------------------------------------------ #
# Assumes `model` is already fine-tuned and `tr_labels`, `te_pixels`,
# `te_labels`, `MASTER_DS` are already in scope.

print("\n" + "#"*60)
print("#  INIT SKETCH BASIS (shared across test points)")
print("#"*60)
params, perm, Omega = init_sketch_basis(model)

# Cache training sketches per class — different test points may share a
# class, in which case we avoid recomputing.
sketch_cache = {}

def get_train_sketches_for_class(c):
    if c not in sketch_cache:
        sketch_cache[c] = compute_train_sketches_at_class(
            model, c, params, perm, Omega
        )
    return sketch_cache[c]

print("\n" + "#"*60)
print("#  PER-TEST-POINT INFLUENCE (test-class-aligned)")
print("#"*60)

ys_train_tensor = tr_labels.clone()    # shape (n_train,)
results = []

for test_idx in TEST_INDICES:
    test_pixels = te_pixels[test_idx]
    test_label  = int(te_labels[test_idx].item())
    print(f"\n  Test idx {test_idx}  "
          f"(class: {CLASS_NAMES[test_label]})")

    # 1) Training sketches at the TEST CLASS
    G_t = get_train_sketches_for_class(test_label)

    # 2) Test sketch at its true class
    sk_test = sketch_at_class(model,
                              test_pixels.unsqueeze(0).to(device),
                              params, perm, Omega,
                              test_label)

    # 3) Signed indicator alpha:
    #    +1 if training label matches test class, -1 otherwise.
    indicator = torch.where(
        ys_train_tensor == test_label,
        torch.tensor( 1.0),
        torch.tensor(-1.0),
    )                                                       # (n_train,)

    # 4) Influence = kernel-similarity × indicator
    kernel_sim = G_t @ sk_test                              # (n_train,)
    infl       = kernel_sim * indicator                     # (n_train,)

    # 5) Top-K helpful (largest positive) and top-K harmful (most neg.)
    helpful = torch.argsort(infl, descending=True)[:TOP_K].tolist()
    harmful = torch.argsort(infl, descending=False)[:TOP_K].tolist()
    helpful_scores = [float(infl[i].item()) for i in helpful]
    harmful_scores = [float(infl[i].item()) for i in harmful]

    print(f"    Top-{TOP_K} helpful (expect: same-class, "
          f"visually similar):")
    for r, (i, s) in enumerate(zip(helpful, helpful_scores)):
        same = "★" if int(tr_labels[i]) == test_label else " "
        print(f"      #{r+1} {same} idx={i:>4}  score={s:+.4f}  "
              f"class={CLASS_NAMES[int(tr_labels[i])]}")
    print(f"    Top-{TOP_K} harmful (expect: different-class):")
    for r, (i, s) in enumerate(zip(harmful, harmful_scores)):
        same = "★" if int(tr_labels[i]) == test_label else " "
        print(f"      #{r+1} {same} idx={i:>4}  score={s:+.4f}  "
              f"class={CLASS_NAMES[int(tr_labels[i])]}")

    # 6) Save figure (use the existing plot_influence_panel from
    #    the original script — no changes needed)
    fig_path = os.path.join(OUT_DIR,
                            f"resnet50_pets_test{test_idx:04d}.png")
    plot_influence_panel(
        test_idx, test_pixels, test_label,
        helpful, helpful_scores,
        harmful, harmful_scores,
        fig_path
    )
    print(f"    Figure -> {fig_path}")

    results.append({
        "test_idx"        : test_idx,
        "test_label"      : test_label,
        "test_class"      : CLASS_NAMES[test_label],
        "helpful_indices" : helpful,
        "helpful_scores"  : helpful_scores,
        "helpful_classes" : [CLASS_NAMES[int(tr_labels[i])]
                             for i in helpful],
        "harmful_indices" : harmful,
        "harmful_scores"  : harmful_scores,
        "harmful_classes" : [CLASS_NAMES[int(tr_labels[i])]
                             for i in harmful],
        "figure_path"     : fig_path,
    })

# ------------------------------------------------------------------ #
#  SAVE JSON SUMMARY
# ------------------------------------------------------------------ #
summary = {
    "model"             : "resnet50",
    "dataset"           : "oxford-iiit-pet",
    "formulation"       : "test-class-aligned NTK-JL with signed indicator",
    "seed"              : SEED,
    "epochs"            : EPOCHS,
    "sketch_dim"        : SKETCH_DIM,
    "target_params"     : TARGET_PARAMS,
    "lambda_reg"        : LAMBDA_REG,
    "top_k"             : TOP_K,
    "num_train"         : int(len(MASTER_DS)),
    "num_classes"       : NUM_CLASSES,
    "test_indices"      : TEST_INDICES,
    "results"           : results,
}
json_path = os.path.join(OUT_DIR, "resnet50_pets_influence_v2.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary JSON -> {json_path}")
print("\n" + "="*60)
print("DONE")
print("="*60)
