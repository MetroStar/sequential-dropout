# run through mnist experiments
import sys, os
sys.path.append('../models')
import argparse
import ExampleModels as em

import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim # TODO maybe switch this to torchmetrics to run as batch
from skimage.metrics import mean_squared_error
from sklearn.decomposition import PCA
import json
import pickle

data_dir = '../dataset'

def get_dataset():
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)

    #train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size=32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    #valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    
    return train_dataset, test_dataset, train_loader, test_loader

#TODO do we want epochs per window

# TODO scale output test


'''

config = {
    epochs=25,
    outdir="results",
    experiment_label="windowed_experiment"
    repeat=1,
    run_eval=True,
    save_models=True,
    windows=[
        (40,.1,"label"), 40 max embedding with 40 *.1 = 4 min embed
        (80,.5, "other_label")
        
    ]
}
'''
def embedding_windows(config):
    
    num_epochs = config["epochs"]
    repeat = config["repeat"] #TODO implement
    windows = config["windows"]
    outdir = config["outdir"]   
    run_eval = config["run_eval"]
    save_models = config["save_models"]
   
    train_dataset, test_dataset, train_loader, test_loader = get_dataset()
    
    results = {}
    results["config"] = config
    window_results = []
    for window in windows:
        window_result ={}
        max_embed_size=window[0]
        min_p = window[1]
        label = window[2]
        print(f"Training window {label},{max_embed_size},{min_p}")

        ae = em.MNISTAutoEncoder(max_embed_size,use_dr= "sequential",dr_min_p= min_p, scale_output=False)
                    
        diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss =ae.train_epoch(train_loader)
            val_loss = ae.test_epoch(test_loader)
            print('EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            diz_loss['train_loss'].append(float(train_loss))
            diz_loss['val_loss'].append(float(val_loss))
            
        window_result["loss"] = diz_loss
        
        
        #save models
        if save_models:
            pth = os.path.join(outdir, label+".pth")
            torch.save(ae,pth )
            window_result["model_path"]= pth
            print(f"Saved model to {pth}")
                        
        
        # run eval
        if run_eval:
            print("running eval")
            n = 10 # TODO parameterize this? - how many embedding sizes to use for eval
            r = np.rint(np.linspace(max_embed_size,int(max_embed_size* min_p),n))

            encoder = ae.encoder
            decoder = ae.decoder
            device = ae.device

            # enumerate embedding sizes
            avg_psnrs=[]
            avg_ssims=[]
            avg_mses=[]

            #TODO precision eval

            for i, embed_size in enumerate(r):
                #enumerate test dataset
                print(i, embed_size)
                psnrs = []
                ssims = []
                mses=[]
                for sample_idx in range(len(test_dataset)):
                    img = test_dataset[sample_idx][0].unsqueeze(0).to(device)
                    encoder.eval()
                    decoder.eval()
                    with torch.no_grad():
                        # ablate embedding
                        embedding = encoder(img)
                        #create mask

                        if embed_size < embedding.shape[1]:

                            embedding[0][-int(embedding.shape[1] - embed_size) :] = 0 

                        rec_img  = decoder(embedding)
                    ic= img.cpu().squeeze().numpy()

                    rc=rec_img.cpu().squeeze().numpy()
                    psnrs.append(float(peak_signal_noise_ratio(ic, rc)))
                    ssims.append(float(ssim(ic,rc, data_range=1.0)))# 1.0 data range for sigmoid
                    mses.append(float(mean_squared_error(ic,rc)))
                avg_psnrs.append(sum(psnrs)/len(psnrs))
                avg_ssims.append(sum(ssims)/len(ssims))
                avg_mses.append(sum(mses)/len(mses))
            window_result["psnr"] =avg_psnrs
            window_result["ssim"]= avg_ssims
            window_result["mse"] = avg_mses
            window_result["emb_dim"]= [float(x) for x in r]
            window_results.append(window_result)
            print("Completed Eval")
            
    results["window_results"] = window_results
        
    #write results

    with open(os.path.join(outdir, f'window_result.json'), 'w') as fp:
        #print(results)
        json.dump(results, fp)
        print("writing results")

            
#TODO precision experiments

'''
config = {
    epochs=25,
    outdir="results",
    experiment_label="fixedembed_experiment"
    repeat=1,
    run_eval=True,
    save_models=True,
    embeddings=[
        4,8,12,16,20,24,28,32,36,40
        
    ]
}
'''
def fixed_embedding(config):
    
    num_epochs = config["epochs"]
    repeat = config["repeat"] #TODO implement
    embedding_configs = config["embeddings"]
    outdir = config["outdir"]   
    run_eval = config["run_eval"]
    save_models = config["save_models"]
   
    train_dataset, test_dataset, train_loader, test_loader = get_dataset()
    
    results = {}
    results["config"] = config
    embed_results = []
    for embedding in embedding_configs:
        embed_result ={}

        label = f"embed_{embedding}"
        print(f"Training embedding {label},{embedding}")

        ae = em.MNISTAutoEncoder(embedding,use_dr= "none")
                    
        diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss =ae.train_epoch(train_loader)
            val_loss = ae.test_epoch(test_loader)
            print('EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            diz_loss['train_loss'].append(float(train_loss))
            diz_loss['val_loss'].append(float(val_loss))
            
        embed_result["loss"] = diz_loss
        
        #save models
        if save_models:
            pth = os.path.join(outdir, label+".pth")
            torch.save(ae,pth )
            embed_result["model_path"]= pth
            print(f"Saved model to {pth}")

        # run eval
        if run_eval:
            print(f"running eval on {embedding}")
             
            encoder = ae.encoder
            decoder = ae.decoder
            device = ae.device

            psnrs = []
            ssims = []
            mses=[] 
            
            #TODO use a built in way of doing this instead of iterating 1 by 1

           
            for sample_idx in range(len(test_dataset)):
                img = test_dataset[sample_idx][0].unsqueeze(0).to(device)
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    embedding = encoder(img)

                    rec_img  = decoder(embedding)
                ic= img.cpu().squeeze().numpy()

                rc=rec_img.cpu().squeeze().numpy()
                psnrs.append(float(peak_signal_noise_ratio(ic, rc)))
                ssims.append(float(ssim(ic,rc, data_range=1.0)))# 1.0 data range for sigmoid
                mses.append(float(mean_squared_error(ic,rc)))
            avg_psnr=sum(psnrs)/len(psnrs)
            avg_ssim=sum(ssims)/len(ssims)
            avg_mse=sum(mses)/len(mses)
            embed_result["psnr"] =avg_psnr
            embed_result["ssim"]= avg_ssim
            embed_result["mse"] = avg_mse
            embed_results.append(embed_result)
            
    results['embed_results'] = embed_results
    with open(os.path.join(outdir, f'embed_result.json'), 'w') as fp:
        #print(results)
        json.dump(results, fp)
        print("writing results")                             
# do pca at different embedding sizes and evaluate
def pca(config):
    
    repeat = config["repeat"] #TODO implement
    embedding_configs = config["embeddings"]
    outdir = config["outdir"]   
    run_eval = config["run_eval"]
    save_models = config["save_models"]
   
    train_dataset, test_dataset, train_loader, test_loader = get_dataset()
    
    results = {}
    results["config"] = config
    embed_results = []
   
    print(np.shape(train_dataset.data.numpy()))
    train_dataset = train_dataset.data.numpy()
    train_dataset = np.reshape(train_dataset, (60000, 28*28))
    print(np.shape(train_dataset))
    pca = PCA(n_components=np.max(embedding_configs))
    pca.fit(train_dataset)

    if save_models:
        with open(os.path.join(outdir, "pca.pkl"), 'wb') as pickle_file:
            pickle.dump(pca, pickle_file)

# TODO different embed sampling strategies

import argparse
if __name__ == '__main__':
    


    parser = argparse.ArgumentParser(description='Run MNIST Tests')
    parser.add_argument('--config_file', metavar='path', required=False,
                        help='path to config file')
    parser.add_argument('--test', required=True,
                        help='test mode', choices=["windows", "fixed", "pca"])

    args = parser.parse_args()
    
    test = args.test.upper()
    
    #TODO load config file
    
    if test == "WINDOWS":
        config = {
            "epochs":20,
            "outdir":"results",
            "experiment_label":"windowed_experiment",
            "repeat":1,
            "run_eval":True,
            "save_models":True,
            "windows":[
                (20,.2,"Max20Min4"), 
                (40,.1,"Max40Min4"), 
                (80,.05, "Max80Min4")

            ]
        }
        embedding_windows(config)
    elif test == "FIXED":
        config = {
            "epochs":20,
            "outdir":"results",
            "experiment_label":"fixedembed_experiment",
            "repeat":1,
            "run_eval":True,
            "save_models":True,
            "embeddings":[
                4,8,12,16,20,24,28, 32, 36,40
        
            ]
        }
        fixed_embedding(config)
        
    elif test =="PCA":
        config = {
            "outdir":"results",
            "experiment_label":"pca_experiment",
            "repeat":1,
            "run_eval":True,
            "save_models":True,
            "embeddings":[
                4,8,12,16,20,24,28, 32, 36,40
        
            ]
        }
        pca(config)
    