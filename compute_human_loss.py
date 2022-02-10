import sys, os
import numpy as np
import glob
import h5py
import glob
import torch

from models import ChamferLoss, EarthMoverLoss

human_dir = '/Users/huazhe_xu/Documents/postdoc_stuff/projects/deformable/VGPL-Dynamics-Prior/interactive/'
names = ['michelle', 'sifan', 'Yixing', 'yizhi']
emd = EarthMoverLoss()
cd = ChamferLoss()
all_persons = []
total_e = []
total_c = []
for name in names:
    this_person = {}
    name_dir = os.path.join(human_dir, name)
    letter_dirs = glob.glob(name_dir+'/*')
    for letter_dir in letter_dirs:

        if 'heart' in letter_dir:
            target = 'heart'
        else:
            target = os.path.basename(letter_dir)[0]
        try:
            particles = np.load(os.path.join(letter_dir, f'{target}_gtp.npy'))[None, ...]
        except:
            continue
        with h5py.File(os.path.join('./interactive/target/', target+'.h5'), "r") as f:
            target_particles = f['positions'][:300][None, ...]
        particles = torch.FloatTensor(particles)
        target_particles = torch.FloatTensor(target_particles)
        emd_loss = emd(particles, target_particles)
        chamfer_loss = cd(particles, target_particles)
        if target in this_person.keys():
            if emd_loss < this_person[target][0]:
                this_person[target] = [emd_loss, chamfer_loss]
        else:
            this_person[target] = [emd_loss, chamfer_loss]
    all_persons.append(this_person)
for i, p in enumerate(all_persons):
    print(names[i])
    print(p)# import pdb
            # pdb.set_trace()
            # print("Keys: %s" % f.keys())

