"""Diagnostic (2026-11-17): does the GEOMETRY of embedding noise decide whether it changes edges?

Same total noise energy (sigma = 0.3 per coordinate on average) added to the standardised
tangent embeddings of the train graph (validation protocol: validation edges vs negatives),
in three shapes: isotropic over all 128 dims, inside a fixed 8-dim subspace, or along each
node's radial direction only. Reports the per-pair probability std across K = 10 draws,
the share of pairs with p_e strictly in (0, 1) at the 5%-FP tau, and the hit rate.
Needs artifacts/diagnostics/sample_diversity/encoder_context.pt (diagnose_sample_diversity.py).

Run from the repository root: python scripts/diagnose_noise_geometry.py
"""
import sys, tomllib, numpy as np, torch
sys.path.insert(0,'.')
from scripts.sweep_conditional import _setup
from stability.hypdiff_models import StabilityLPModel, StabilityDiffusion
from stability.model_training import build_model_args, normalized_adjacency
from stability.validation_recovery import calibrate_tau
torch.set_num_threads(2)
raw=tomllib.load(open(sys.argv[1] if len(sys.argv) > 1 else 'configs/cora.toml','rb'))
config, graph, split = _setup(raw)
args,_=build_model_args(graph, config, torch.device('cpu'))
saved=torch.load('artifacts/diagnostics/sample_diversity/encoder_context.pt',weights_only=False)
enc=StabilityLPModel(args); enc.load_state_dict(saved['encoder']); enc.eval()
den=StabilityDiffusion(args,dim=args.dim,device='cpu')
lab,cen=saved['labels'],saved['centers']; proto=cen[lab]
A=normalized_adjacency(graph.node_count, split.train,'cpu')
pairs=torch.as_tensor(np.r_[split.val, split.val_negatives]); y=np.r_[np.ones(len(split.val)),np.zeros(len(split.val_negatives))]
with torch.no_grad():
  emb=enc.encode(graph.features,A)[2]
  tang=den.manifold.logmap(proto,emb,c=1.0); m,s=tang.mean(),tang.std(); x=(tang-m)/s
  n,d=x.shape
  radial=x/x.norm(dim=1,keepdim=True)
  def run(kind, sigma=0.3, K=10):
    P=[]
    for k in range(K):
      g=torch.Generator().manual_seed(k)
      if kind=='isotropic': noise=sigma*torch.randn(n,d,generator=g,dtype=x.dtype)
      elif kind=='radial': noise=sigma*np.sqrt(d)*torch.randn(n,1,generator=g,dtype=x.dtype)*radial
      elif kind=='subspace8':
        B=torch.linalg.qr(torch.randn(d,8,generator=torch.Generator().manual_seed(123),dtype=x.dtype))[0]
        noise=sigma*np.sqrt(d/8)*torch.randn(n,8,generator=g,dtype=x.dtype)@B.T
      r=den.manifold.expmap((x+noise)*s+m,proto,c=1.0); P.append(enc.decode(r,pairs).numpy().astype(float))
    P=np.stack(P); tau=calibrate_tau(P[:,y==0].reshape(-1),0.05); pe=(P>=tau).mean(0)
    print(f'{kind:10s} same total noise energy: prob std across K {P.std(0).mean():.3f}; p_e in (0,1) {np.mean((pe>0)&(pe<1)):.2f}; hit {(P[:,y==1]>=tau).mean():.3f}')
  for kind in ('isotropic','subspace8','radial'): run(kind)
