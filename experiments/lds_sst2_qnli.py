# ================================================================
# Scaled LDS harness (frozen regime) -- SST-2 + QNLI
# Methods: NTK-JL(datamodel), NTK-JL(decomp), RPS, TRAK(M=1), TRAK(M=5)
# Sweeps:  (Tier1) main table + subset-fraction alpha in {0.25,0.5,0.75}
#          (Tier2) projection dim k in {64,128,256,512,full}
#
# CHANGE vs previous version:
#   - t_datamodel now uses the MARGIN gradient on the train side, so the
#     datamodel readout is the output-function readout that equals single-model
#     TRAK exactly (NTK-JL(datamodel) == TRAK(M=1) up to projection seed).
#   - The LOGIT-gradient datamodel (the weaker variant for the appendix
#     logit-vs-margin comparison) is kept as a separate function
#     t_datamodel_logit, reported in both sweeps under the key
#     "NTK-JL(datamodel-logit)".
#
# Frozen backbone -> "model" = linear head on cached CLS features; subset-trained
# model = head refit (cheap) so M and the sweeps are nearly free. Same subsets/models
# shared across all methods within a condition; only the attribution differs.
# Output function = binary margin (logit_correct - logit_other). LDS = Spearman over
# subsets between actual margin and predicted = sum of tau over subset, averaged over
# test points. Everything is cached features + linear algebra: ~minutes per dataset.
# ================================================================
import os, json, math, random
import numpy as np
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
try:
    from scipy.stats import spearmanr
    def spearman(a,b):
        r=spearmanr(a,b).correlation; return float(r) if r==r else 0.0
except Exception:
    def spearman(a,b):
        ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
        ra-=ra.mean(); rb-=rb.mean(); d=math.sqrt((ra**2).sum()*(rb**2).sum())
        return float((ra*rb).sum()/d) if d>0 else 0.0
try:
    from IPython.display import display, Image as IPImage; _HAVE_IPY=True
except Exception: _HAVE_IPY=False

# ---------------- CONFIG ----------------
SEED=42
N_TRAIN=4000; N_TEST=200; M_SUBSETS=500
ALPHAS=[0.25,0.50,0.75]; ALPHA_MAIN=0.50
PROJ_MAIN=512; PROJ_SWEEP=[64,128,256,512,"full"]
ENS_MAX=5            # main bar needs TRAK up to M=5 (no dedicated ensemble-size sweep)
LAMBDA=1e-2; FIT_STEPS=200; FIT_LR=5e-2; FIT_WD=1e-3
METHODS=["NTK-JL(datamodel)","NTK-JL(decomp)","RPS","TRAK(M=1)","TRAK(M=5)"]
DATASETS=[
    {"name":"SST-2","hf":("nyu-mll/glue","sst2"),"backbone":"distilbert-base-uncased","pair":False,"max_len":64},
    {"name":"QNLI","hf":("nyu-mll/glue","qnli"),"backbone":"bert-base-uncased","pair":True, "max_len":128},
]
OUT_DIR="/mnt/user-data/outputs"; os.makedirs(OUT_DIR,exist_ok=True)
OUT_JSON=os.path.join(OUT_DIR,"lds_scaled_sst2_qnli_data.json")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device} | scaled LDS | datasets {[d['name'] for d in DATASETS]} | N={N_TRAIN} T={N_TEST} M={M_SUBSETS}")
def set_seed(s=SEED):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)
set_seed()

# ---------------- generic frozen-CLS feature extraction ----------------
def extract(cfg, texts_a, texts_b=None):
    tok=AutoTokenizer.from_pretrained(cfg["backbone"]); mdl=AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()
    out=[]; bs=128
    for s in tqdm(range(0,len(texts_a),bs),desc=f"  {cfg['name']} CLS",leave=False):
        a=texts_a[s:s+bs]; b=texts_b[s:s+bs] if texts_b is not None else None
        enc=tok(a,b,truncation=True,padding="max_length",max_length=cfg["max_len"],return_tensors="pt") if b is not None \
            else tok(a,truncation=True,padding="max_length",max_length=cfg["max_len"],return_tensors="pt")
        with torch.no_grad():
            h=mdl(input_ids=enc["input_ids"].to(device),attention_mask=enc["attention_mask"].to(device)).last_hidden_state[:,0,:]
        out.append(h.float().cpu())
    del mdl; torch.cuda.empty_cache()
    return torch.cat(out)

def load_dataset_feats(cfg):
    raw=load_dataset(*cfg["hf"]); tr,va=raw["train"],raw["validation"]
    tr_sel=random.Random(SEED).sample(range(len(tr)),N_TRAIN)
    by={0:[],1:[]}
    for i in range(len(va)): by[int(va[i]["label"])].append(i)
    te_sel=sorted(by[0][:N_TEST//2]+by[1][:N_TEST-N_TEST//2])
    if cfg["pair"]:
        a_tr=[tr[i]["question"] for i in tr_sel]; b_tr=[tr[i]["sentence"] for i in tr_sel]
        a_te=[va[i]["question"] for i in te_sel]; b_te=[va[i]["sentence"] for i in te_sel]
    else:
        a_tr=[tr[i]["sentence"] for i in tr_sel]; b_tr=None
        a_te=[va[i]["sentence"] for i in te_sel]; b_te=None
    TR=extract(cfg,a_tr,b_tr).to(device); TE=extract(cfg,a_te,b_te).to(device)
    ytr=torch.tensor([int(tr[i]["label"]) for i in tr_sel],device=device)
    yte=torch.tensor([int(va[i]["label"]) for i in te_sel],device=device)
    return TR,TE,ytr,yte,te_sel

# ---------------- head fit + margins ----------------
def fit_head(X,y,seed):
    set_seed(seed)
    W=torch.zeros(2,X.shape[1],device=device,requires_grad=True); b=torch.zeros(2,device=device,requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=FIT_LR,weight_decay=FIT_WD)
    for _ in range(FIT_STEPS):
        loss=F.cross_entropy(X@W.t()+b,y); opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(),b.detach()
def margins(W,b,X,yl):
    lg=X@W.t()+b; ar=torch.arange(lg.shape[0],device=device); return (lg[ar,yl]-lg[ar,1-yl])

# ---------------- gradient builders + projection ----------------
def block_grad(Faug,labels,margin):
    n,d=Faug.shape; G=torch.zeros(n,2,d,device=device); idx=torch.arange(n,device=device); G[idx,labels,:]=Faug
    if margin: G[idx,1-labels,:]=-Faug
    return G.reshape(n,2*d)
def block_grad_class(Faug,c):
    n,d=Faug.shape; G=torch.zeros(n,2,d,device=device); G[:,c,:]=Faug; return G.reshape(n,2*d)
def proj(k,din,seed):
    if k=="full" or k>=din: return torch.eye(din,device=device)
    g=torch.Generator(device="cpu"); g.manual_seed(seed)
    return (torch.randn(din,k,generator=g)/math.sqrt(k)).to(device)

# ---------------- attributions (return tau [T,N] numpy) ----------------
# Datamodel readout with the OUTPUT-FUNCTION (margin) gradient on the train side.
# This is the readout that equals single-model TRAK exactly (same margin grads on
# both sides, same Q=1-p weight, same kernel inverse) -> NTK-JL(datamodel)==TRAK(M=1).
def t_datamodel(Ftr,ytr,Fte,yte,ptr,P,lam):
    Gtr=block_grad(Ftr,ytr,True)@P; k=Gtr.shape[1]
    Kinv=torch.linalg.inv(Gtr.t()@Gtr+lam*torch.eye(k,device=device))
    Gte=block_grad(Fte,yte,True)@P; N=Ftr.shape[0]; r=1.0-ptr[torch.arange(N,device=device),ytr]
    return ((Gte@Kinv@Gtr.t())*r[None,:]).cpu().numpy()

# Datamodel readout with the LOGIT-at-assigned-class gradient on the train side.
# Weaker instance of the same form (matches margin on SST-2, lags on QNLI); used
# only for the appendix logit-vs-margin comparison.
def t_datamodel_logit(Ftr,ytr,Fte,yte,ptr,P,lam):
    Gtr=block_grad(Ftr,ytr,False)@P; k=Gtr.shape[1]
    Kinv=torch.linalg.inv(Gtr.t()@Gtr+lam*torch.eye(k,device=device))
    Gte=block_grad(Fte,yte,True)@P; N=Ftr.shape[0]; r=1.0-ptr[torch.arange(N,device=device),ytr]
    return ((Gte@Kinv@Gtr.t())*r[None,:]).cpu().numpy()

def t_decomp(Ftr,ytr,Fte,yte,P,lam):
    G=block_grad(Ftr,ytr,False)@P; k=G.shape[1]; Y=F.one_hot(ytr,2).float()
    v=torch.cholesky_solve(G.t()@Y,torch.linalg.cholesky(G.t()@G+lam*torch.eye(k,device=device))); alpha=(1/lam)*(Y-G@v)
    G0=block_grad_class(Fte,0)@P; G1=block_grad_class(Fte,1)@P
    i0=(G@G0.t())*alpha[:,0:1]; i1=(G@G1.t())*alpha[:,1:2]; T=Fte.shape[0]; N=Ftr.shape[0]
    tau=torch.zeros(T,N,device=device)
    for t in range(T): ct=int(yte[t]); tau[t]=(i0[:,t]-i1[:,t]) if ct==0 else (i1[:,t]-i0[:,t])
    return tau.cpu().numpy()
def t_trak(Ftr,ytr,Fte,yte,Plist,plist,lam):
    T=Fte.shape[0]; N=Ftr.shape[0]; acc=torch.zeros(T,N,device=device)
    for P,p in zip(Plist,plist):
        Phi=block_grad(Ftr,ytr,True)@P; Phite=block_grad(Fte,yte,True)@P; k=Phi.shape[1]
        Q=1.0-p[torch.arange(N,device=device),ytr]; Kinv=torch.linalg.inv(Phi.t()@Phi+lam*torch.eye(k,device=device))
        acc+=(Phite@Kinv@Phi.t())*Q[None,:]
    return (acc/len(Plist)).cpu().numpy()
def t_rps(Ftr,ytr,Fte,yte,ptr):
    Y=F.one_hot(ytr,2).float(); arps=Y-ptr; Kte=Ftr@Fte.t(); T=Fte.shape[0]; N=Ftr.shape[0]
    tau=torch.zeros(T,N,device=device)
    for t in range(T): ct=int(yte[t]); tau[t]=(arps[:,ct]-arps[:,1-ct])*Kte[:,t]
    return tau.cpu().numpy()

def lds(tau,PhiS,actual):
    pred=PhiS@tau.T; T=actual.shape[1]
    v=np.array([spearman(actual[:,t],pred[:,t]) for t in range(T)])
    return float(np.mean(v)), float(np.std(v)/math.sqrt(len(v)))

# ---------------- per-dataset run ----------------
ALL={}
for cfg in DATASETS:
    name=cfg["name"]; print("\n"+"#"*66+f"\n#  {name}\n"+"#"*66)
    TR,TE,ytr,yte,te_sel=load_dataset_feats(cfg)
    N,D=TR.shape; T=TE.shape[0]; din=2*(D+1)
    Ftr=torch.cat([TR,torch.ones(N,1,device=device)],1); Fte=torch.cat([TE,torch.ones(T,1,device=device)],1)
    W0,b0=fit_head(TR,ytr,SEED); p_tr=torch.softmax(TR@W0.t()+b0,1)
    acc=(TR@W0.t()+b0).argmax(1).eq(ytr).float().mean().item(); print(f"  frozen-head train acc {100*acc:.1f}%  (params {din}, proj_main {PROJ_MAIN})")
    # ensemble heads + projections (for TRAK M=1..5; member 0 = full head)
    ens_P=[proj(PROJ_MAIN,din,SEED+100+e) for e in range(ENS_MAX)]
    ens_p=[p_tr]+[torch.softmax(TR@fit_head(TR,ytr,SEED+200+e)[0].t()+fit_head(TR,ytr,SEED+200+e)[1],1) for e in range(1,ENS_MAX)]

    # ground truth per alpha (resume per dataset) -- unchanged; attribution-only edits don't touch GT
    CK=os.path.join(OUT_DIR,f"lds_scaled_{name}_ckpt.json")
    GT={}
    prev={}
    if os.path.isfile(CK):
        try:
            pv=json.load(open(CK))
            if pv.get("te_sel")==te_sel and pv.get("N")==N and pv.get("M")==M_SUBSETS: prev=pv.get("actual",{})
        except Exception: pass
    for a in ALPHAS:
        rng=np.random.RandomState(SEED+int(1000*a)); aN=int(a*N)
        subs=[np.sort(rng.choice(N,aN,replace=False)) for _ in range(M_SUBSETS)]
        PhiS=np.zeros((M_SUBSETS,N),np.float32)
        for j,s in enumerate(subs): PhiS[j,s]=1.0
        key=f"{a:.2f}"; actual=np.array(prev[key],float) if key in prev else np.full((M_SUBSETS,T),np.nan)
        if np.isnan(actual).any():
            for j in tqdm(range(M_SUBSETS),desc=f"  GT alpha={a}",leave=False):
                if not np.isnan(actual[j,0]): continue
                s=torch.as_tensor(subs[j],device=device); Wj,bj=fit_head(TR[s],ytr[s],SEED+5000+j)
                actual[j]=margins(Wj,bj,TE,yte).cpu().numpy()
            prev[key]=np.where(np.isnan(actual),None,actual).tolist()
            json.dump({"te_sel":te_sel,"N":N,"M":M_SUBSETS,"actual":prev},open(CK,"w"))
        GT[a]=(PhiS,actual)

    res={"train_acc":acc,"alpha_sweep":{},"proj_sweep":{},"main":{}}
    # --- alpha sweep (5 headline methods + logit datamodel for the appendix) ---
    P512=proj(PROJ_MAIN,din,SEED)
    for a in ALPHAS:
        PhiS,actual=GT[a]
        taus={"NTK-JL(datamodel)":t_datamodel(Ftr,ytr,Fte,yte,p_tr,P512,LAMBDA),
              "NTK-JL(decomp)":t_decomp(Ftr,ytr,Fte,yte,P512,LAMBDA),
              "RPS":t_rps(Ftr,ytr,Fte,yte,p_tr),
              "TRAK(M=1)":t_trak(Ftr,ytr,Fte,yte,ens_P[:1],ens_p[:1],LAMBDA),
              "TRAK(M=5)":t_trak(Ftr,ytr,Fte,yte,ens_P[:5],ens_p[:5],LAMBDA),
              "NTK-JL(datamodel-logit)":t_datamodel_logit(Ftr,ytr,Fte,yte,p_tr,P512,LAMBDA)}
        res["alpha_sweep"][f"{a:.2f}"]={m:lds(taus[m],PhiS,actual) for m in taus}
    res["main"]={m:res["alpha_sweep"][f"{ALPHA_MAIN:.2f}"][m] for m in METHODS}  # headline bar = 5 methods only
    # --- projection sweep (datamodel margin + logit, decomp, TRAK M=1) at alpha_main ---
    PhiS,actual=GT[ALPHA_MAIN]
    for k in PROJ_SWEEP:
        Pk=proj(k,din,SEED); kn="full" if k=="full" else str(k)
        res["proj_sweep"][kn]={
            "NTK-JL(datamodel)":lds(t_datamodel(Ftr,ytr,Fte,yte,p_tr,Pk,LAMBDA),PhiS,actual),
            "NTK-JL(datamodel-logit)":lds(t_datamodel_logit(Ftr,ytr,Fte,yte,p_tr,Pk,LAMBDA),PhiS,actual),
            "NTK-JL(decomp)":lds(t_decomp(Ftr,ytr,Fte,yte,Pk,LAMBDA),PhiS,actual),
            "TRAK(M=1)":lds(t_trak(Ftr,ytr,Fte,yte,[proj(k,din,SEED+100)],ens_p[:1],LAMBDA),PhiS,actual)}
    ALL[name]=res

    # ---- print tables ----
    print(f"\n  [{name}] LDS main (alpha={ALPHA_MAIN}, proj={PROJ_MAIN}):")
    for m in METHODS: mm,se=res["main"][m]; print(f"    {m:<18} {mm:+.4f} +/- {se:.4f}")
    dl=res["alpha_sweep"][f"{ALPHA_MAIN:.2f}"]["NTK-JL(datamodel-logit)"]
    print(f"    {'NTK-JL(dm-logit)':<18} {dl[0]:+.4f} +/- {dl[1]:.4f}   (appendix logit variant)")
    print(f"  [{name}] alpha sweep (dm-margin | dm-logit | TRAK(M=1) | RPS):")
    for a in ALPHAS:
        d=res["alpha_sweep"][f"{a:.2f}"]
        print(f"    a={a:.2f}  dm {d['NTK-JL(datamodel)'][0]:+.3f}  dm-logit {d['NTK-JL(datamodel-logit)'][0]:+.3f}  trak1 {d['TRAK(M=1)'][0]:+.3f}  rps {d['RPS'][0]:+.3f}")
    print(f"  [{name}] proj sweep (dm-margin): "+"  ".join(f"{kn}:{res['proj_sweep'][kn]['NTK-JL(datamodel)'][0]:.3f}" for kn in res["proj_sweep"]))
    print(f"  [{name}] proj sweep (dm-logit):  "+"  ".join(f"{kn}:{res['proj_sweep'][kn]['NTK-JL(datamodel-logit)'][0]:.3f}" for kn in res["proj_sweep"]))

json.dump(ALL,open(OUT_JSON,"w"),indent=2); print(f"\nALL numbers -> {OUT_JSON}")

# ---------------- figures ----------------
COL={"NTK-JL(datamodel)":"#2ca02c","NTK-JL(decomp)":"#98df8a","RPS":"#1f77b4","TRAK(M=1)":"#ff7f0e","TRAK(M=5)":"#d62728"}
# main grouped bar across datasets (5 headline methods; datamodel is now the margin readout = TRAK M=1)
fig,ax=plt.subplots(figsize=(11,6),dpi=150); names=list(ALL.keys()); x=np.arange(len(METHODS)); w=0.38
for di,nm in enumerate(names):
    means=[ALL[nm]["main"][m][0] for m in METHODS]; ses=[ALL[nm]["main"][m][1] for m in METHODS]
    ax.bar(x+(di-0.5)*w,means,w,yerr=ses,capsize=3,label=nm,alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(METHODS,rotation=20,ha="right",fontsize=9); ax.axhline(0,color="gray",lw=1)
ax.set_ylabel("mean LDS",fontsize=12,fontweight="bold"); ax.set_title(f"LDS by method (frozen, alpha={ALPHA_MAIN}, M={M_SUBSETS})",fontsize=12,fontweight="bold")
ax.legend(fontsize=11); ax.grid(True,axis="y",alpha=0.3,ls="--")
p_main=os.path.join(OUT_DIR,"lds_scaled_main_bar.png"); plt.tight_layout(); plt.savefig(p_main,dpi=150,bbox_inches="tight"); print(f"plot -> {p_main}")
if _HAVE_IPY:
    try: display(IPImage(filename=p_main))
    except Exception: pass

# per-dataset 1x2: alpha sweep | projection sweep (now also plots dm-logit)
COL2={**COL,"NTK-JL(datamodel-logit)":"#8c564b"}
for nm in names:
    R=ALL[nm]; fig,ax=plt.subplots(1,2,figsize=(12,5),dpi=150)
    for m in METHODS+["NTK-JL(datamodel-logit)"]:
        ys=[R["alpha_sweep"][f"{a:.2f}"][m][0] for a in ALPHAS]
        ax[0].plot(ALPHAS,ys,marker="o",lw=2.5,label=m,color=COL2[m])
    ax[0].set_xlabel("subset fraction alpha",fontweight="bold"); ax[0].set_ylabel("mean LDS",fontweight="bold")
    ax[0].set_title(f"{nm}: alpha sweep"); ax[0].axhline(0,color="gray",lw=1); ax[0].grid(True,alpha=0.3,ls="--"); ax[0].legend(fontsize=7)
    ks=list(R["proj_sweep"].keys()); xk=range(len(ks))
    for m in ["NTK-JL(datamodel)","NTK-JL(datamodel-logit)","TRAK(M=1)","NTK-JL(decomp)"]:
        ax[1].plot(list(xk),[R["proj_sweep"][k][m][0] for k in ks],marker="s",lw=2.5,label=m,color=COL2[m])
    ax[1].set_xticks(list(xk)); ax[1].set_xticklabels(ks); ax[1].set_xlabel("projection dim k",fontweight="bold")
    ax[1].set_ylabel("mean LDS",fontweight="bold"); ax[1].set_title(f"{nm}: projection sweep"); ax[1].grid(True,alpha=0.3,ls="--"); ax[1].legend(fontsize=7)
    pp=os.path.join(OUT_DIR,f"lds_scaled_{nm}_sweeps.png"); plt.tight_layout(); plt.savefig(pp,dpi=150,bbox_inches="tight"); print(f"plot -> {pp}")
    if _HAVE_IPY:
        try: display(IPImage(filename=pp))
        except Exception: pass
print("\nDone. Expect per dataset: datamodel(margin) ~ TRAK(M=1); dm-logit lags on QNLI; decomp ~ 0; RPS low; LDS rising with k.")
