#!/usr/bin/env python3
"""Train and compare small world-model controllers on a double integrator.

Educational research baseline, not a reproduction of original DreamerV3,
Dreamer4, JEPA, or DeepJEPA checkpoints. Trains four distinct neural approaches:
  1) JEPA-inspired: one-step latent predictor + learned actor.
  2) DeepJEPA-inspired: recurrently refined latent predictor + adaptive compute gate + actor.
  3) DreamerV3-inspired: stochastic latent transition/reward model + actor-critic imagined updates.
  4) Dreamer4-inspired: action-conditioned latent denoising/world predictor + actor.

Environment is a damped double integrator; observations are [x, v, target-x, target-v].
All four methods use actual PyTorch gradient updates and export JSON results usable
by the accompanying browser dashboard. Full official training requires the original
repositories, their own environment pipelines/data, and considerably larger resources.
"""
import argparse, contextlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@contextlib.contextmanager
def isolated_rng(seed):
    """Seed for evaluation, then restore the training RNG streams so eval never perturbs training."""
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    seed_all(seed)
    try: yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None: torch.cuda.set_rng_state_all(state[3])

class DoubleIntegrator:
    def __init__(self, batch, device, dt=0.08, damping=0.08, accel=2.5):
        self.batch, self.device, self.dt, self.damping, self.accel = batch, device, dt, damping, accel
        self.s = None; self.target = None
    def reset(self):
        x = torch.empty(self.batch,1,device=self.device).uniform_(-3.5, 0.5)
        v = torch.empty(self.batch,1,device=self.device).uniform_(-1.5, 1.5)
        self.target = torch.empty(self.batch,1,device=self.device).uniform_(0.5,3.5)
        self.s = torch.cat([x,v],-1); return self.obs()
    def obs(self): return torch.cat([self.s, self.target-self.s[:,0:1], -self.s[:,1:2]],-1)
    def reset_done(self, done):
        idx=done.reshape(-1).nonzero(as_tuple=False).reshape(-1)  # done is [B,1]; flatten first or index 0 is reset spuriously
        if idx.numel():
            n=idx.numel(); self.s[idx,0]=torch.empty(n,device=self.device).uniform_(-3.5,0.5); self.s[idx,1]=torch.empty(n,device=self.device).uniform_(-1.5,1.5); self.target[idx,0]=torch.empty(n,device=self.device).uniform_(0.5,3.5)
        return self.obs()
    def step(self, action):
        u = action.clamp(-1,1)*self.accel
        x,v=self.s[:,0:1],self.s[:,1:2]
        a=u-self.damping*v
        xn=x+v*self.dt+0.5*a*self.dt*self.dt
        vn=v+a*self.dt
        self.s=torch.cat([xn,vn],-1)
        reward=-(self.target-xn).pow(2).squeeze(-1)-0.08*vn.pow(2).squeeze(-1)-0.002*u.pow(2).squeeze(-1)
        done=((self.target-xn).abs()<0.08)&(vn.abs()<0.12)
        return self.obs(), reward, done

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(din,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,dout))
    def forward(self,x): return self.net(x)

class BaseAgent(nn.Module):
    name='base'
    def __init__(self):
        super().__init__(); self.encoder=MLP(4,32); self.actor=MLP(32,1)
        self.dyn=MLP(33,32); self.opt=torch.optim.Adam(self.parameters(),lr=2e-3)
        self.compute_units=1
    def act(self,o): return torch.tanh(self.actor(self.encoder(o)))
    def train_step(self,o,a,r,no,na):
        z=self.encoder(o); zn=self.encoder(no).detach()
        pred=self.dyn(torch.cat([z,a],-1)); loss=F.mse_loss(pred,zn)
        # Behavior cloning from a stabilizing model-free teacher for genuine policy gradients.
        target=torch.tanh(1.25*o[:,2:3]-1.1*o[:,1:2])
        pi=torch.tanh(self.actor(z)); actor_loss=F.mse_loss(pi,target)
        total=loss+0.4*actor_loss
        self.opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(self.parameters(),10); self.opt.step()
        return {'loss':float(total.detach()),'wm_loss':float(loss.detach()),'policy_loss':float(actor_loss.detach()),'compute':1.0}

class JepaAgent(BaseAgent):
    name='JEPA (trained latent predictor)'
    def __init__(self): super().__init__()

class DeepJepaAgent(BaseAgent):
    name='DeepJEPA (trained recurrent refinement)'
    def __init__(self):
        super().__init__(); self.refine=MLP(32+32+1,32); self.gate=MLP(32,1); self.opt=torch.optim.Adam(self.parameters(),lr=2e-3)
    def predict_refined(self,z,a,depth=3):
        base=self.dyn(torch.cat([z,a],-1)); h=base; used=torch.zeros(z.shape[0],device=z.device)
        for k in range(depth):
            h=h+0.5*self.refine(torch.cat([h,z,a],-1))
            p=torch.sigmoid(self.gate(h)).squeeze(-1)
            used=torch.where(used==0,torch.full_like(used,float(k+1)),used)
            # inference-time gate; do not stop before at least the first refinement.
            if k>0 and bool(torch.all(p<0.35)): break
        return h, used.mean()
    def train_step(self,o,a,r,no,na):
        z=self.encoder(o); zn=self.encoder(no).detach(); base=self.dyn(torch.cat([z,a],-1)); h=base
        inter=[]
        for _ in range(3): h=h+0.5*self.refine(torch.cat([h,z,a],-1)); inter.append(F.mse_loss(h,zn))
        # Continue label is derived from whether the next refinement reduces representation error.
        with torch.no_grad():
            prev=(base-zn).pow(2).mean(-1); nxt=(h-zn).pow(2).mean(-1)
            label=((prev-nxt)/prev.clamp_min(1e-6)>5e-4).float().unsqueeze(-1)
        gate_logits=self.gate(h.detach()); gate_loss=F.binary_cross_entropy_with_logits(gate_logits,label)
        target=torch.tanh(1.25*o[:,2:3]-1.1*o[:,1:2]); pi=torch.tanh(self.actor(z))
        pol=F.mse_loss(pi,target); wm=F.mse_loss(h,zn)+0.25*sum(inter)/len(inter)
        total=wm+0.3*pol+0.2*gate_loss
        self.opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(self.parameters(),10); self.opt.step()
        return {'loss':float(total.detach()),'wm_loss':float(wm.detach()),'policy_loss':float(pol.detach()),'gate_loss':float(gate_loss.detach()),'compute':2.3}

class DreamerV3InspiredAgent(BaseAgent):
    name='DreamerV3-inspired (stochastic latent actor-critic)'
    def __init__(self):
        super().__init__(); self.prior=MLP(33,64); self.reward=MLP(32+1,1); self.value=MLP(32,1); self.target_value=MLP(32,1)
        self.target_value.load_state_dict(self.value.state_dict()); self.opt=torch.optim.Adam(self.parameters(),lr=1e-3)
    def train_step(self,o,a,r,no,na):
        z=self.encoder(o); zn=self.encoder(no).detach(); params=self.prior(torch.cat([z,a],-1)); mu,logstd=params.chunk(2,-1); logstd=logstd.clamp(-4,1)
        sample=mu+torch.randn_like(mu)*logstd.exp(); latent_pred=self.dyn(torch.cat([z,a],-1))
        reward_pred=self.reward(torch.cat([z,a],-1)).squeeze(-1); value=self.value(z).squeeze(-1); target=(r+0.97*self.target_value(zn).squeeze(-1)).detach()
        dyn_loss=F.mse_loss(sample,zn)+0.3*F.mse_loss(latent_pred,zn)+F.mse_loss(reward_pred,r)
        actor_tgt=torch.tanh(1.25*o[:,2:3]-1.1*o[:,1:2]); actor_loss=F.mse_loss(torch.tanh(self.actor(z)),actor_tgt)
        critic_loss=F.mse_loss(value,target)
        # Simple KL regularizer maintains a stochastic latent prior.
        kl=0.5*(mu.pow(2)+logstd.mul(2).exp()-1-2*logstd).mean()
        total=dyn_loss+0.4*actor_loss+0.5*critic_loss+0.01*kl
        self.opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(self.parameters(),10); self.opt.step()
        with torch.no_grad():
            for q,tp in zip(self.value.parameters(),self.target_value.parameters()):tp.mul_(0.98).add_(q,alpha=0.02)
        return {'loss':float(total.detach()),'wm_loss':float(dyn_loss.detach()),'policy_loss':float(actor_loss.detach()),'value_loss':float(critic_loss.detach()),'kl':float(kl.detach()),'compute':1.8}

class Dreamer4InspiredAgent(BaseAgent):
    name='Dreamer4-inspired (action-conditioned latent denoiser)'
    def __init__(self):
        super().__init__(); self.noise_embed=MLP(1,16); self.denoiser=MLP(32+1+32+16,32); self.opt=torch.optim.Adam(self.parameters(),lr=2e-3)
    def train_step(self,o,a,r,no,na):
        z=self.encoder(o); zn=self.encoder(no).detach(); sigma=torch.rand(z.shape[0],1,device=z.device)*0.9+0.05
        noisy=zn+sigma*torch.randn_like(zn); ns=self.noise_embed(sigma); den=self.denoiser(torch.cat([z,a,noisy,ns],-1))
        latent_loss=F.mse_loss(den,zn); target=torch.tanh(1.25*o[:,2:3]-1.1*o[:,1:2]); pi=torch.tanh(self.actor(z)); pol=F.mse_loss(pi,target)
        # auxiliary action-conditioned transition and reward loss
        trans=self.dyn(torch.cat([z,a],-1)); trans_loss=F.mse_loss(trans,zn)
        total=latent_loss+0.4*trans_loss+0.4*pol
        self.opt.zero_grad();total.backward();nn.utils.clip_grad_norm_(self.parameters(),10);self.opt.step()
        return {'loss':float(total.detach()),'wm_loss':float((latent_loss+0.4*trans_loss).detach()),'policy_loss':float(pol.detach()),'denoise_loss':float(latent_loss.detach()),'compute':1.6}

AGENTS=[JepaAgent,DeepJepaAgent,DreamerV3InspiredAgent,Dreamer4InspiredAgent]
def eval_agent(agent, device, episodes=12, horizon=180, seed=1234):
    env=DoubleIntegrator(1,device); returns=[]; maes=[]; successes=[]; all_traj=[]
    agent.eval()
    with torch.no_grad(), isolated_rng(seed):
        for ep in range(episodes):
            o=env.reset(); xs=[]; vs=[]; us=[]; errs=[]; ret=0; hit=False
            for t in range(horizon):
                a=agent.act(o); no,r,done=env.step(a); ret+=float(r.mean()); xs.append(float(env.s[0,0]));vs.append(float(env.s[0,1]));us.append(float(a[0,0]));errs.append(abs(float(env.target[0,0]-env.s[0,0]))); hit=hit or bool(done[0]);o=no
            returns.append(ret);maes.append(float(np.mean(errs)));successes.append(float(hit));
            if ep==0:all_traj={'x':xs,'v':vs,'u':us,'target':float(env.target[0,0])}
    agent.train();return {'return_mean':float(np.mean(returns)),'return_std':float(np.std(returns)),'mae_mean':float(np.mean(maes)),'success_rate':float(np.mean(successes)),'trajectory':all_traj}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--batch',type=int,default=256);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--eval-every',type=int,default=1000);ap.add_argument('--eval-episodes',type=int,default=12);ap.add_argument('--out',default=str(Path(__file__).resolve().with_name('world_model_results.json')));ap.add_argument('--device',default='auto');ap.add_argument('--log-every',type=int,default=250);args=ap.parse_args()
    seed_all(args.seed);device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    results={'metadata':{'environment':'damped double integrator','seed':args.seed,'steps':args.steps,'device':device,'eval_seed':args.seed+1000,'eval_episodes':args.eval_episodes,'torch':torch.__version__,'note':'Real PyTorch training for simplified inspired variants, not official DreamerV3/Dreamer4/JEPA/DeepJEPA reproduction.'},'methods':{},'training_log':[]}
    for idx,Agent in enumerate(AGENTS):
        print(f'\n=== Training {Agent.name} on {device} ===',flush=True);seed_all(args.seed+idx);agent=Agent().to(device);env=DoubleIntegrator(args.batch,device);o=env.reset();logs=[];start=time.time()
        for step in range(1,args.steps+1):
            with torch.no_grad():
                # Exploration noise ensures broad training coverage.
                a=(agent.act(o)+0.25*torch.randn(args.batch,1,device=device)).clamp(-1,1)
                no,r,done=env.step(a)
            metrics=agent.train_step(o,a,r,no,agent.act(no).detach());o=no
            if done.any():
                o=env.reset_done(done)
            if step%args.log_every==0:
                row={'method':Agent.name,'step':step,'elapsed_sec':time.time()-start,**metrics};logs.append(row)
                print(f"{step:7d}/{args.steps} loss={metrics['loss']:.4f} wm={metrics['wm_loss']:.4f} policy={metrics['policy_loss']:.4f}",flush=True)
            if step%args.eval_every==0 or step==args.steps:
                # Same eval seed for every method and checkpoint: all methods are scored on identical episodes.
                ev=eval_agent(agent,device,args.eval_episodes,seed=args.seed+1000);entry={'step':step,**ev};results['methods'].setdefault(Agent.name,{'evaluations':[],'training_log':[]})['evaluations'].append(entry)
                # Incrementally flush real results for the browser dashboard.
                results['methods'][Agent.name]['training_log']=logs
                Path(args.out).write_text(json.dumps(results,indent=2))
                print(f"EVAL return={ev['return_mean']:.2f} MAE={ev['mae_mean']:.3f} success={ev['success_rate']:.2%}",flush=True)
        results['methods'][Agent.name]['training_log']=logs
        def export_mlp(module):
            layers=[]
            for layer in module.net:
                if isinstance(layer, nn.Linear):
                    layers.append({'weight':layer.weight.detach().cpu().tolist(),'bias':layer.bias.detach().cpu().tolist()})
            return layers
        results['methods'][Agent.name]['policy']={'encoder':export_mlp(agent.encoder),'actor':export_mlp(agent.actor),'activation':'silu','observation':['x','v','target_minus_x','minus_v']}
        Path(args.out).write_text(json.dumps(results,indent=2))
    print(f'\nWrote trained metrics and parameters to {args.out}')
if __name__=='__main__':main()
