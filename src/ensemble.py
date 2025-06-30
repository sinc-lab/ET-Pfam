"""
Este archivo tiene la implementación de la ex-clase domCNNe, ahora llamada EnsembleModel.
"""
import os
import torch as tr
from torch import nn
from tqdm import tqdm
from src.model import BaseModel
from src.dataset import PFamDataset
from src.utils import load_config
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class EnsembleModel(nn.Module):
    def __init__(self, models_path, emb_path, data_path, cat_path, 
                 voting_strategy, model_weights_path=None):

        super(EnsembleModel, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        # Load categories
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        
        self.categories = categories
        self.emb_path = emb_path
        self.data_path = data_path
        self.voting_strategy = voting_strategy
        self.path = models_path

        # Initialize the ensemble of models
        self.models = nn.ModuleList()
        self.model_configs = []

        # Sort model directories to ensure consistent order
        model_dirs.sort()

        for model_dir in model_dirs:
            # Load the config.json to get the parameters
            config_path = os.path.join(model_dir, 'config.json')
            config = load_config(config_path)

            lr = config['lr']
            batch_size = config['batch_size']
            win_len = config['window_len']
            device = config['device']

            self.model_configs.append({
                'lr': lr,
                'batch_size': batch_size,
                'win_len': win_len
            })

            # Load the model weights
            weights_path = os.path.join(model_dir, 'weights.pk')
            print("loading weights from", model_dir)
            model = BaseModel(len(categories), lr=lr, device=device)
            model.load_state_dict(tr.load(weights_path))
            model.eval()
            self.models.append(model)
        
        if self.voting_strategy == 'weighted_model': # TODO: CHECK
            weights_ensemble = f"{model_weights_path}ensemble_model_weights.pt"
            if model_weights_path and os.path.exists(weights_ensemble):
                self.model_weights = nn.Parameter(tr.load(weights_ensemble))
                print(f"Loaded model weights from {model_weights_path}")
            else:
                self.model_weights = nn.Parameter(tr.rand(len(model_dirs)))
                if model_weights_path:
                    print(f"Warning: {model_weights_path} not found, using random init.")

        elif self.voting_strategy == 'weighted_families': # TODO: CHECK
            weights_ensemble = f"{model_weights_path}ensemble_family_weights.pt"
            if model_weights_path and os.path.exists(weights_ensemble):
                self.family_weights = nn.Parameter(tr.load(weights_ensemble))
                print(f"Loaded family weights from {model_weights_path}")
            else:
                self.family_weights = nn.Parameter(tr.rand(len(model_dirs), len(categories)))
                if model_weights_path:
                    print(f"Warning: {model_weights_path} not found, using random init.")

    def fit(self):
        if self.voting_strategy in ['weighted_model', 'weighted_families']:
            all_preds = []
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                dev_data = PFamDataset(
                    f"{self.data_path}dev.csv",
                    self.emb_path,
                    self.categories,
                    win_len=config['win_len'],
                    is_training=False
                )
                dev_loader = tr.utils.data.DataLoader(dev_data, batch_size=config['batch_size'], num_workers=config.get("nworkers", 1))

                with tr.no_grad():
                    _, _, pred, ref, *_ = net.pred(dev_loader)
                    all_preds.append(pred)
            stacked_preds = tr.stack(all_preds)

        if self.voting_strategy == 'weighted_model':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.model_weights], lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            
            tr.save(self.model_weights.detach().cpu(), f'{self.path}/ensemble_model_weights.pt')
            print("Saved model weights to ensemble_model_weights.pt")

        if self.voting_strategy == 'weighted_families':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.family_weights], lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            tr.save(self.family_weights.detach().cpu(), f'{self.path}/ensemble_family_weights.pt')
            print("Saved family weights to ensemble_family_weights.pt")

    def forward(self, batch):
        pred, _ = self.pred(batch)
        return pred

    def pred(self, batch=None, window_type='sliding'): # centered or sliding
        all_preds = []
        
        if window_type == 'sliding':
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                net_preds = []
                with tr.no_grad():
                    pred = net(batch).cpu().detach()
                    net_preds.append(pred)
                net_preds = tr.cat(net_preds)
                all_preds.append(net_preds)
        else:
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                test_data = PFamDataset(
                    f"{self.data_path}test.csv",
                    self.emb_path,
                    self.categories,
                    win_len=config['win_len'],
                    is_training=False
                )
                test_loader = tr.utils.data.DataLoader(test_data, batch_size=config['batch_size'], num_workers=config.get("nworkers", 1))

                net_preds = []
                with tr.no_grad():
                    test_loss, test_errate, pred, *_ = net.pred(test_loader)
                    net_preds.append(pred)
                print(f"win_len = {config['win_len']} - lr = {config['lr']} - test_loss {test_loss:.5f} - test_errate {test_errate:.5f}")
                net_preds = tr.cat(net_preds)
                all_preds.append(net_preds)

        stacked_preds = tr.stack(all_preds)

        if self.voting_strategy == 'score_voting':
            pred = tr.mean(stacked_preds, dim=0)
            pred_bin = tr.argmax(pred, dim=1)
            return pred, pred_bin
        elif self.voting_strategy == 'weighted_model':
            pred = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
            pred_bin = tr.argmax(pred, dim=1)
            return pred, pred_bin
        elif self.voting_strategy == 'weighted_families':
            pred = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
            pred_bin = tr.argmax(pred, dim=1)
            return pred, pred_bin
        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)
            return pred, pred_bin
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")