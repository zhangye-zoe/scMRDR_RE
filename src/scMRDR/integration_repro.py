import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
from .data import CombinedDataset
from .model import EmbeddingNet
from .train import train_model, inference_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import anndata as ad
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix,csr_matrix,issparse
import scanpy as sc
from sklearn.model_selection import train_test_split
import ot
from sklearn.neighbors import NearestNeighbors

def to_dense_array(x):
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

class Integration:
    def __init__(self, data, layer=None, modality_key="modality", batch_key=None, celltype_key=None,
                 distribution="ZINB", mask_key=None, feature_list=None):
        super(Integration,self).__init__()
        if isinstance(data, list) & isinstance(data[0], ad.AnnData):
            self.adata = ad.concat(data, axis='obs', join='inner', label="modality")
        elif isinstance(data, ad.AnnData):
            self.adata = data
        else:
            raise ValueError("Wrong type of data!")
        if layer is None:
            self.data = to_dense_array(self.adata.X)
        else:
            self.data = to_dense_array(self.adata.layers[layer])

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)

        self.modality_label = self.adata.obs[modality_key].to_numpy()
        self.modality = label_encoder.fit_transform(self.modality_label)
        self.modality_ordered = [label for label in label_encoder.classes_]
        self.modality = onehot_encoder.fit_transform(self.modality.reshape(-1, 1))

        if celltype_key is None:
            self.celltype = None
            self.celltype_ordered = None
        else:
            self.celltype_label = self.adata.obs[celltype_key].to_numpy()
            self.celltype = label_encoder.fit_transform(self.celltype_label)
            self.celltype_ordered = [label for label in label_encoder.classes_]
            self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))

        if batch_key is None:
            self.covariates = None
            self.covariates_ordered = None
        else:
            self.covariates_label = self.adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates_ordered = [label for label in label_encoder.classes_]
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))

        self.modality_num = self.modality.shape[1]

        if self.celltype is not None:
            self.celltype_num = self.celltype.shape[1]
        else:
            self.celltype_num = 0

        if self.covariates is not None:
            self.covariates_dim = self.covariates.shape[1]
        else:
            self.covariates_dim = 0

        if mask_key is None:
            self.mask = self.modality
            self.mask_num = self.modality_num
            self.mask_ordered = self.modality_ordered
        else:
            self.mask_label = self.adata.obs[mask_key].to_numpy()
            self.mask = label_encoder.fit_transform(self.mask_label)
            self.mask_ordered = [label for label in label_encoder.classes_]
            self.mask = onehot_encoder.fit_transform(self.mask.reshape(-1, 1))
            self.mask_num = self.mask.shape[1]

        if feature_list is not None:
            self.feat_mask = 0 * torch.ones(self.mask_num, self.data.shape[1])
            feature_list_ordered = [feature_list[label] for label in self.mask_ordered]
            for i, feat_idx in enumerate(feature_list_ordered):
                self.feat_mask[i, feat_idx] = 1
        else:
            self.feat_mask = torch.ones(self.mask_num, self.data.shape[1])

        self.distribution = distribution
        if self.distribution in ["ZINB", "NB"]:
            self.count_data = True
            self.positive_outputs = True
        elif self.distribution == "Normal":
            self.count_data = False
            self.positive_outputs = False
        elif self.distribution == "Normal_positive":
            self.count_data = False
            self.positive_outputs = True
        else:
            raise ValueError("Distribution not recognized!")

        self.prediction_results = {}

    def setup(self, hidden_layers=[100,50], latent_dim_shared=15, latent_dim_specific=15, dropout_rate=0.5,
              beta=2, gamma=1, lambda_adv=0.01, device=None):
        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.gamma = gamma
        self.lambda_adv = lambda_adv

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        print("using "+str(self.device))
        self.model = EmbeddingNet(
            self.device, self.input_dim, self.modality_num, self.covariates_dim,
            layer_dims=self.hidden_layers,
            latent_dim_shared=self.latent_dim_shared,
            latent_dim_specific=self.latent_dim_specific,
            dropout_rate=self.dropout_rate,
            beta=self.beta, gamma=self.gamma, lambda_adv=self.lambda_adv,
            feat_mask=self.feat_mask, distribution=self.distribution
        ).to(self.device)
        self.train_dataset = CombinedDataset(self.data,self.covariates,self.modality,self.mask, self.celltype)

    def train(self, epoch_num=200, batch_size=64, lr=1e-5, accumulation_steps=1,
              adaptlr=False, valid_prop=0.1, num_warmup=0, early_stopping=True, patience=10,
              weighted=False, tensorboard=False, savepath="./", random_state=42):
        if tensorboard:
            print("Using tensorboard!")
            self.writer = SummaryWriter(savepath)
        else:
            self.writer = None
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.adaptlr = adaptlr

        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset

        self.num_batch = len(train_dataset)//self.batch_size

        print("Training start!")
        if weighted:
            weights = 1.0 / np.bincount(self.modality.argmax(-1))
            sample_weights = weights[self.modality.argmax(-1)]
            sample_weights = sample_weights[train_indices]
            train_model(
                self.device, self.writer, train_dataset, valid_dataset,
                self.model, self.epoch_num, self.batch_size,
                self.num_batch, self.lr, accumulation_steps=self.accumulation_steps,
                adaptlr=self.adaptlr, num_warmup=num_warmup, early_stopping=early_stopping,
                patience=patience, sample_weights=sample_weights
            )
        else:
            train_model(
                self.device, self.writer, train_dataset, valid_dataset,
                self.model, self.epoch_num, self.batch_size,
                self.num_batch, self.lr, accumulation_steps=self.accumulation_steps,
                adaptlr=self.adaptlr, num_warmup=num_warmup, early_stopping=early_stopping,
                patience=patience
            )

        if tensorboard:
            self.writer.close()
        print("Training finished!")

    def generate_from_latent(
        self,
        z_concat,
        modality,
        covariates=None,
        library_size=None,
        n_samples=1,
        return_decoder_outputs=False,
    ):
        self.model.eval()

        z_t = torch.tensor(z_concat, dtype=torch.float32, device=self.device)
        m_t = torch.tensor(modality, dtype=torch.float32, device=self.device)

        if covariates is not None:
            b_t = torch.tensor(covariates, dtype=torch.float32, device=self.device)
        else:
            if self.covariates_dim > 0:
                b_t = torch.zeros((z_t.shape[0], self.covariates_dim), dtype=torch.float32, device=self.device)
            else:
                b_t = torch.zeros((z_t.shape[0], 0), dtype=torch.float32, device=self.device)

        if library_size is not None:
            s_t = torch.tensor(library_size, dtype=torch.float32, device=self.device)
            if s_t.ndim == 1:
                s_t = s_t.unsqueeze(1)
        else:
            s_t = None

        xs = []
        raw_outputs = []

        with torch.no_grad():
            for _ in range(n_samples):
                if self.distribution in ["ZINB", "NB"]:
                    rho, dispersion, pi = self.model.decoder(z_t, b_t, m_t)

                    if s_t is None:
                        x_hat = rho
                    else:
                        x_hat = rho * s_t

                    xs.append(x_hat.detach().cpu().numpy())
                    raw_outputs.append({
                        "rho": rho.detach().cpu().numpy(),
                        "dispersion": dispersion.detach().cpu().numpy(),
                        "pi": pi.detach().cpu().numpy(),
                    })
                else:
                    rho = self.model.decoder(z_t, b_t)
                    xs.append(rho.detach().cpu().numpy())
                    raw_outputs.append({"rho": rho.detach().cpu().numpy()})

        x_pred = np.mean(np.stack(xs, axis=0), axis=0)

        if return_decoder_outputs:
            return x_pred, raw_outputs
        return x_pred

    def inference(
        self,
        n_samples=1,
        dataset=None,
        batch_size=None,
        update=True,
        returns=False,
        predict_modalities=None,
        predict_strategy="latent",
        predict_method="knn",
        predict_k=10,
        prediction_library_size=None,
    ):
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size

        if n_samples > 1:
            z_shared, z_specific = zip(
                *[inference_model(self.device, dataset, self.model, batch_size) for _ in range(n_samples)]
            )
            self.z_shared = np.mean(np.stack(z_shared, axis=0), axis=0)
            self.z_specific = np.mean(np.stack(z_specific, axis=0), axis=0)
        else:
            self.z_shared, self.z_specific = inference_model(self.device, dataset, self.model, batch_size)

        if update:
            self.adata.obsm['latent_shared'] = self.z_shared
            self.adata.obsm['latent_specific'] = self.z_specific
            print('Latent results recorded in adata.')

        pred_results = {}
        if predict_modalities is not None:
            if isinstance(predict_modalities, str):
                predict_modalities = [predict_modalities]

            for pred_mod in predict_modalities:
                x_pred = self.predict(
                    predict_modality=pred_mod,
                    batch_size=batch_size,
                    strategy=predict_strategy,
                    library_size=prediction_library_size,
                    method=predict_method,
                    k=predict_k,
                )
                impt_index = self.modality_label != pred_mod
                pred_results[pred_mod] = {
                    "data": np.asarray(x_pred, dtype=np.float32),
                    "obs_names": np.asarray(self.adata.obs_names[impt_index].tolist(), dtype=object),
                    "var_names": np.asarray(self.adata.var_names.tolist(), dtype=object),
                    "source_modalities": np.asarray(self.modality_label[impt_index].tolist(), dtype=object),
                }

            self.prediction_results = pred_results
            print("Prediction results computed for:", list(pred_results.keys()))

        if returns:
            return {
                "z_shared": self.z_shared,
                "z_specific": self.z_specific,
                "predictions": pred_results,
            }

    def predict(self, predict_modality, batch_size=None, strategy="observed", library_size=None, method="ot", k=10):
        if batch_size is None:
            batch_size = self.batch_size

        z_shared, z_specific = self.z_shared, self.z_specific
        curr_index = self.modality_label == predict_modality
        impt_index = self.modality_label != predict_modality
        z_shared_curr = z_shared[curr_index,:]
        z_specific_curr = z_specific[curr_index,:]
        z_shared_impt = z_shared[impt_index,:]

        if strategy == "observed":
            x_curr = self.data[curr_index,:]

            if method == "ot":
                a = ot.unif(z_shared_impt.shape[0])
                b = ot.unif(z_shared_curr.shape[0])
                M = ot.dist(z_shared_impt, z_shared_curr, metric='euclidean')
                W = ot.emd(a, b, M)
                W = W/W.sum(axis=1,keepdims=True)
                x_pred = np.dot(W,x_curr)
            elif method == "knn":
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_shared_curr)
                distances, indices = nbrs.kneighbors(z_shared_impt)
                weights = 1 / (distances + 1e-5)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                x_pred = np.array([
                    np.sum(x_curr[indices[i]] * weights[i][:, np.newaxis], axis=0)
                    for i in range(indices.shape[0])
                ])
            else:
                raise ValueError("Unknown method!")

        elif strategy == "latent":
            z_concat_curr = np.concatenate((z_shared_curr, z_specific_curr), axis=1)

            if method == "ot":
                a = ot.unif(z_shared_impt.shape[0])
                b = ot.unif(z_shared_curr.shape[0])
                M = ot.dist(z_shared_impt, z_shared_curr, metric='euclidean')
                W = ot.emd(a, b, M)
                W = W / W.sum(axis=1, keepdims=True)
                z_concat = np.dot(W, z_concat_curr)

            elif method == "knn":
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_shared_curr)
                distances, indices = nbrs.kneighbors(z_shared_impt)
                weights = 1 / (distances + 1e-5)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                z_concat = np.array([
                    np.sum(z_concat_curr[indices[i]] * weights[i][:, np.newaxis], axis=0)
                    for i in range(indices.shape[0])
                ])
            else:
                raise ValueError("Unknown method!")

            if self.covariates is not None:
                covariates = self.covariates[impt_index, :]
            else:
                covariates = None

            modality = np.tile(self.modality[curr_index, :][0, :], (z_concat.shape[0], 1))

            # -------- library size: use cell-specific weighted RNA library size --------
            if library_size is None and self.count_data:
                curr_index_np = curr_index.astype(bool)
                lib_curr = self.data[curr_index_np, :].sum(axis=1)   # shape: n_curr

                if method == "knn":
                    library_size_use = np.array([
                        np.sum(lib_curr[indices[i]] * weights[i])
                        for i in range(indices.shape[0])
                    ], dtype=np.float32).reshape(-1, 1)

                elif method == "ot":
                    library_size_use = np.dot(W, lib_curr.reshape(-1, 1)).astype(np.float32)

                else:
                    library_size_use = None
            else:
                library_size_use = library_size

            x_pred = self.generate_from_latent(
                z_concat,
                modality,
                covariates=covariates,
                library_size=library_size_use,
                n_samples=1
            )
        else:
            raise ValueError("Unknown strategy!")

        return x_pred

    def get_prediction_adata(self, predict_modality, prediction_result=None):
        if prediction_result is None:
            if predict_modality not in self.prediction_results:
                raise KeyError(f"No stored prediction result for modality: {predict_modality}")
            prediction_result = self.prediction_results[predict_modality]

        pred_adata = ad.AnnData(
            X=np.asarray(prediction_result["data"], dtype=np.float32),
            obs=pd.DataFrame(index=list(prediction_result["obs_names"])),
            var=pd.DataFrame(index=list(prediction_result["var_names"])),
        )
        pred_adata.obs["predicted_to"] = predict_modality
        pred_adata.obs["source_modality"] = list(prediction_result["source_modalities"])
        pred_adata.layers["data"] = pred_adata.X.copy()
        return pred_adata

    def get_adata(self):
        return self.adata
