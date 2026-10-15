import math
from typing import Tuple
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFingerprintGenerator, rdMolDescriptors
import torch
from tdc.single_pred import ADME
from matplotlib import pyplot as plt

def main() -> None:
    data = ADME(name="Caco2_Wang", path="./tdc_data")
    split = data.get_split(method="scaffold", seed=42)

    fp_size = 512
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    training_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['train']['Drug']], dtype=torch.float32)
    training_expected = torch.tensor(split['train']['Y'], dtype=torch.float32)

    validation_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['valid']['Drug']], dtype=torch.float32)
    validation_expected = torch.tensor(split['valid']['Y'], dtype=torch.float32)

    test_input = torch.tensor([get_mol_representation(smile, fpgen) for smile in split['test']['Drug']], dtype=torch.float32)
    test_expected = torch.tensor(split['test']['Y'], dtype=torch.float32)

    baseline = training_expected.median()
    baseline_test_mae = (test_expected - baseline).abs().mean()
    print("Baseline test MAE:", baseline_test_mae.item())

    # Training
    epoch_count = 100
    learning_rate = 0.1
    batch_size = 25

    torch.manual_seed(42)

    depth = (training_input.shape[1], 100, 25, 1)

    W = [torch.randn((depth[i], depth[i + 1])) / math.sqrt(depth[i]) for i in range(len(depth) - 1)]
    B = [torch.zeros((depth[i + 1])) for i in range(len(depth) - 1)]

    for i in range(len(depth) - 1):
        W[i].requires_grad_()
        B[i].requires_grad_()

    bestError = float('inf')
    best_W = W.copy()
    best_B = B.copy()

    training_errors = []
    validation_errors = []

    for j in range(epoch_count):
        indices = torch.randperm(training_input.shape[0])
        for k in range(0, training_input.shape[0], batch_size):
            if k + batch_size > training_input.shape[0]:
                break

            batch = training_input[indices[k:k+batch_size]]
            expected = training_expected[indices[k:k+batch_size]] 

            batch = run_model(W, B, batch, depth)
            
            error = calculate_mae(batch.reshape(batch_size), expected)
            error.backward()

            with torch.no_grad():
                for i in range(len(depth) - 1):
                    W[i] = W[i] - W[i].grad * learning_rate
                    B[i] = B[i] - B[i].grad * learning_rate

                    W[i] = W[i] - W[i] * 0.005 * learning_rate

                    W[i].requires_grad_()
                    B[i].requires_grad_()

        with torch.no_grad():
            training_result = run_model(W, B, training_input, depth)
            training_errors.append(calculate_mae(training_result.reshape(len(training_input)), training_expected).item())

            validation_result = run_model(W, B, validation_input, depth)
            validation_errors.append(calculate_mae(validation_result.reshape(len(validation_input)), validation_expected).item())

            if validation_errors[-1] < bestError:
                bestError = validation_errors[-1]
                best_W = [w.clone().detach() for w in W]
                best_B = [b.clone().detach() for b in B]

        learning_rate *= 0.95

    print("Initial training error:", training_errors[0])
    print("Final training error:", training_errors[-1])
    print("Initial validation error:", validation_errors[0])
    print("Final validation error:", validation_errors[-1])
    validation_result = run_model(best_W, best_B, validation_input, depth)
    print("Min validation error:", (calculate_mae(validation_result.reshape(len(validation_input)), validation_expected).item()))

    # 🫣
    test_result = run_model(best_W, best_B, test_input, depth)
    print(f"Final test error: {calculate_mae(test_result.reshape(len(test_input)), test_expected).item()}")

    plt.plot(training_errors[2:], color="blue")
    plt.plot(validation_errors[2:], color="red")
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
    
def run_model(W: list[torch.Tensor], B: list[torch.Tensor], input: torch.Tensor, depth: Tuple):
    batch = input.clone()
    for i in range(len(depth) - 1):
        batch @= W[i]
        batch += B[i]
        if i < len(depth) - 2:
            torch.relu_(batch)

    return batch

def calculate_mae(result: torch.Tensor, expected: torch.Tensor):
    return (result - expected).abs().mean()
