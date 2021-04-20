#!/usr/bin/env python
# coding: utf-8

import torch
import os.path
import matplotlib.pyplot as plt
from matplotlib import animation,rc

import numpy as np
import random

nx = 5
ny = 5
nz = 5
nptl = nx*ny*nz
L = 22.28
nstep = 1000000
ds = 0.1
beta = 1.0/0.5962
eps = 0.1854
sigma = 3.542
Ra = torch.tensor(9.0)
Rb = torch.tensor(9.2)

repeat = torch.tensor([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
                      [0,0,1],[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,1],
                      [0,0,-1],[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]])*2*L

#print(repeat.shape)
def random_num(minimum,maximum,n):
    return minimum+torch.rand(n)*(maximum-minimum)

def vij(x):
    e = eps*torch.sum( (sigma/x)**12 - (sigma/x)**6 ) 
    return e

def init_energy(x):
    en0 = torch.tensor(0.0)
    for j in range(x.shape[0]):
        xdist = torch.sqrt( torch.sum(((x[j,:]-x[:,:]).unsqueeze(1)+repeat)**2,dim=2) )
        xdist = xdist.reshape(-1)    
        mask = (xdist < 10.0) & (xdist > 0.0)
        en0 = en0+vij(xdist[mask])
    return 0.5*en0

if os.path.isfile('position.dat'):
    x=[]
    f=open('position.dat','r')
    for lines in f:
         c=lines.split() 
         x.append(torch.tensor( [float(c[0]), float(c[1]), float(c[2]) ]) ) 
    x=torch.stack(x)
else:
    radius=random_num(Ra,L,nptl)
    phi=random_num(0.0,2.0*np.pi,nptl)
    theta=random_num(0.0,np.pi,nptl)
    x0 = radius*torch.cos(phi)*torch.sin(theta)
    y0 = radius*torch.sin(phi)*torch.sin(theta)
    z0 = radius*torch.cos(theta) 
    x = torch.stack([x0,y0,z0])
    x = torch.transpose(x,0,1)

def dF_eval(x):
    radii = torch.sqrt(torch.sum(x[:,:])**2,dim=1)
    mask = (radii > Ra) & (radii < Rb)
    if torch.sum(mask) == 0:
        return torch.tensor(1.0)
    else:
        return torch.tensor(0.0)

def move(x,energy):
    j = random.randint(0,nptl-1)

    disp=torch.rand(3)-torch.tensor([0.5,0.5,0.5])
    disp=ds*disp/torch.sqrt(torch.sum(disp**2))
    xtry = x[j,:] + disp

    if torch.sqrt(torch.sum( xtry[:])**2 ) < Ra:
        iaccept = 0
        return x, iaccept, energy

    for k in range(3):
        if xtry[k] > 2*L:
            xtry[k] = xtry[k]-2*L
        elif xtry[k]<=0:
            xtry[k] = xtry[k]+2*L
    
    xdist0 = torch.sqrt( torch.sum(((x[j,:]-x[:,:]).unsqueeze(1)+repeat)**2,dim=2) )
    xdist0 = xdist0.reshape(-1)    
    mask = (xdist0 < 10.0) & (xdist0 > 0.0)
#   print(xdist0[mask])
#   print('xdist0_min:',torch.min(xdist0[mask]))
    energy0 = vij(xdist0[mask])
        
        
    xdist = torch.sqrt( torch.sum(((xtry[:]-x[:,:]).unsqueeze(1)+repeat)**2,dim=2) )
    xdist = xdist.reshape(-1)
    mask = (xdist < 10.0) & (xdist > ds+0.011)
#   print(xdist[mask])
#   print('xdist_min:',torch.min(xdist[mask]))
    energy_trial = vij(xdist[mask])
#   print('energy0:',energy0.item(),'energy_trial:',energy_trial.item())
#   exit()
    de = energy_trial - energy0
#   print('de:',de)
    pratio = torch.exp(-beta*de)
#   print('pratio:',pratio)
#   exit()

    if(pratio >= 1.0):
        x[j,:] = xtry
        iaccept = 1
        energy = energy+de
    else:
        r = torch.rand(1)
        if(r < pratio):
            x[j,:]=xtry
            iaccept = 1
            energy = energy+de
        else:
            iaccept = 0
            
#   print('energy=',energy,'iaccept:',iaccept)
    return x, iaccept, energy

ntherm=10000
ii = 0
energy = init_energy(x)
while ii < ntherm:
    x,iacc,energy = move(x,energy)
    ii = ii + 1
print('Warming up done')
ii = 0
ia = 0
en_avg = torch.tensor(0.0)
energy = torch.tensor(0.0)
#energy = init_energy(x)
dF = 0.0
isample = 0
fen=open('energy.dat','a') 
f=open('dF.dat','a') 
while ii < nstep:
    x,iacc,energy = move(x,energy)
    ia = ia + iacc
    if ii%1000 == 99:
        isample = isample + 1
        en_avg = en_avg + energy
#       dF = dF + dF_eval(x)
#       print(isample,dF/float(isample),file=f)
        print(isample,en_avg.item()/isample,file=fen)
    ii = ii + 1
    
print('acceptance ratio:',float(ia)/nstep)

fn = open('position.dat','w')
for xa in x:
    print(xa[0].item(),xa[1].item(),xa[2].item(),file=fn)

