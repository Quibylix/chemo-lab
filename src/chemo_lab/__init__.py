import math
from typing import Tuple
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFingerprintGenerator, rdMolDescriptors
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tdc.single_pred import ADME
from matplotlib import pyplot as plt
import copy

class MoleculeModel(nn.Module):
    def __init__(self, depth: Tuple):
        super().__init__()
        layers = []
        for i in range(len(depth) - 1):
            layers.append(nn.Linear(depth[i], depth[i + 1]))
            if i < len(depth) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def main() -> None:
    data = ADME(name="Caco2_Wang", path="./tdc_data")
    split = data.get_split(method="scaffold", seed=42)

    fp_size = 512
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size, includeChirality=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    training_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['train']['Drug']], dtype=torch.float32).to(device)
    training_expected = torch.tensor(split['train']['Y'], dtype=torch.float32).to(device)

    validation_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['valid']['Drug']], dtype=torch.float32).to(device)
    validation_expected = torch.tensor(split['valid']['Y'], dtype=torch.float32).to(device)

    test_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['test']['Drug']], dtype=torch.float32).to(device)
    test_expected = torch.tensor(split['test']['Y'], dtype=torch.float32).to(device)

    baseline = (training_expected - training_expected.median()).abs().mean().item()
    print("Baseline test MAE:", baseline)

    # Training
    torch.manual_seed(42)

    depth = (training_input.shape[1], 100, 25, 1)
    epoch_count = 100
    learning_rate = 0.1
    train_batch_size = 25
    valid_batch_size = 25
    weight_decay = 0.005
    gamma = 0.95

    train_dataset = TensorDataset(training_input, training_expected.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    valid_dataset = TensorDataset(validation_input, validation_expected.unsqueeze(1))
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    model = MoleculeModel(depth).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_val_loss = float('inf')
    best_model_weights = None

    validation_errors = []

    for j in range(epoch_count):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in valid_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_predictions = model(val_x)
                val_loss += criterion(val_predictions, val_y).item() * val_x.size(0)

        val_loss /= len(valid_dataset)
        validation_errors.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    if best_model_weights is None:
        return

    model.eval()
    model.load_state_dict(best_model_weights)
    final_train_error = criterion(model(training_input), training_expected.unsqueeze(1)).item()
    print("Final training error:", final_train_error)
    print("Initial validation error:", validation_errors[0])
    print("Final validation error:", validation_errors[-1])
    min_valid_error = criterion(model(validation_input), validation_expected.unsqueeze(1)).item()
    print("Min validation error:", min_valid_error)

    # 🫣
    test_error = criterion(model(test_input), test_expected.unsqueeze(1)).item()
    print(f"Final test error: {test_error}")

    plt.plot(validation_errors[2:], color="blue")
    plt.title("Error over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

def get_mol_representation(smile: str, fpgen: rdFingerprintGenerator.FingerprintGenerator64) -> list[int]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    fingerprint = fpgen.GetCountFingerprint(mol)
    return fingerprint.ToList() + [rdMolDescriptors.CalcExactMolWt(mol) / 500,
                                   rdMolDescriptors.CalcNumHBD(mol) / 5,
                                   rdMolDescriptors.CalcNumHBA(mol) / 10,
                                   rdMolDescriptors.CalcTPSA(mol) / 200]
