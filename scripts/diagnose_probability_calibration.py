"""Diagnostic (2026-11-17): AUC and calibration of decoded edge PROBABILITIES (no threshold)
for no_diffusion, clamp_only and the mean over K samples of the pilot-1 conditional model
(residual, upstream noise, no clip, t*=30, eta=1). Validation protocol only; base-rate
weights = missing edges vs all non-edges. Needs the pilot-1 checkpoints.
Run from the repository root: python scripts/diagnose_probability_calibration.py
"""
import sys, tomllib, numpy as np, torch
sys.path.insert(0,'.')
from scripts.sweep_conditional import _setup, BASE_DENOISER
from stability.hypdiff_models import StabilityLPModel, StabilityDiffusion
from stability.model_training import build_model_args, normalized_adjacency
from stability.sample_diversity import calibration_table
from stability.validation_recovery import _auc
from stability.edge_recovery_experiment import recovery_seed
from stability.validation_recovery import VALIDATION_SERIES_INDEX
torch.set_num_threads(2)
raw=tomllib.load(open('configs/cora.toml','rb')); raw['denoiser'].update(BASE_DENOISER)
raw['denoiser'].update(architecture='conditional_residual_gcn', angular_noise='upstream', clip_x0=False)
raw['recovery'].update(sampler='stochastic', eta=1.0, steps=20)
config, graph, split = _setup(raw)
args,_=build_model_args(graph, config, torch.device('cpu'))
saved=torch.load('artifacts/diagnostics/sample_diversity/encoder_context.pt',weights_only=False)
enc=StabilityLPModel(args); enc.load_state_dict(saved['encoder']); enc.eval()
den=StabilityDiffusion(args,dim=args.dim,device='cpu')
den.load_state_dict(torch.load('artifacts/sweeps/conditional_pilot/denoiser_architecture=conditional_residual_gcn,angular_noise=upstream,clip_x0=False.pt')); den.eval()
lab,cen=saved['labels'],saved['centers']
A=normalized_adjacency(graph.node_count, split.train,'cpu')
pairs=torch.as_tensor(np.r_[split.val, split.val_negatives]); y=np.r_[np.ones(len(split.val)),np.zeros(len(split.val_negatives))].astype(int)
n=graph.node_count; w=np.where(y==1,(len(split.val)+len(split.pool))/len(split.val),(n*(n-1)//2-len(graph.edges))/len(split.val_negatives))
with torch.no_grad():
  emb=enc.encode(graph.features,A)[2]
  nod=enc.decode(emb,pairs).numpy().astype(float)
  clamp=enc.decode(den.shrink_to_prototype(emb,lab,cen),pairs).numpy().astype(float)
  P=np.stack([enc.decode(den.sample_from_tstar(emb,A,lab,cen,30,torch.Generator().manual_seed(recovery_seed(config.recovery.seed,VALIDATION_SERIES_INDEX,k))),pairs).numpy().astype(float) for k in range(10)])
mean=P.mean(0)
for name,sc in (('no_diffusion prob',nod),('clamp_only prob',clamp),('diffusion mean prob',mean)):
  print(f'{name:20s} AUC {_auc(sc[y==1],sc[y==0]):.3f}  ECE balanced {calibration_table(sc,y)["ece"]:.3f}  ECE base-rate {calibration_table(sc,y,w)["ece"]:.3f}  mean on non-edges {sc[y==0].mean():.3f}')
h,_=np.histogram(mean[y==1],bins=[0,.1,.3,.5,.7,.9,1.0001]); print('diffusion mean prob, validation edges, bins 0-.1/.1-.3/.3-.5/.5-.7/.7-.9/.9-1:', (h/h.sum()).round(2))
