import numpy as np
from pysat.examples.rc2 import RC2 # RC2 MaxSat solver https://alexeyignatiev.github.io/assets/pdf/imms-jsat19-preprint.pdf
from pysat.formula import WCNF # for installation, see: https://pysathq.github.io/installation.html (pip install python-sat)


# MaxSAT formulation and solution
def MaxSAT(predictions, confidences, nli_matrix, return_flip_mask=False):
    wcnf = WCNF() # initialize CNF

    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0

    for i, (pred_row, conf_row) in enumerate(zip(predictions, confidences)): # iterate thru rows
        # j + i * B is the # assigned to constraint at index (i, j)
        constraints = [([j + i * B + 1] if pred_row[j] == 1 else [-(j + i * B + 1)]) for j in range(B)]
        # print (constraints, conf_row.tolist())
        wcnf.extend(constraints, weights=conf_row.tolist())

    contradiction_constraints = []
    weights = []
    for i, matr in enumerate(contra_matrix): # matr is BxB
        for j in range(B):
            for k in range(B):
                if k > j:
                    P = j + i * B + 1 if predictions[i][j] == 1 else -(j + i * B + 1)
                    Q = k + i * B + 1 if predictions[i][k] == 1 else -(k + i * B + 1)
                    contradiction_constraints.append([-P, -Q])      # P ∨ Q == ~ (~P ∧ ~Q)
                    weights.append((matr[j][k] - 0.5) * (-10)) # linearly scale from (0,1) to + weight if p < 0.5 / - weight if p > 0.5
    # print(contradiction_constraints, weights)
    wcnf.extend(contradiction_constraints, weights=weights)

    # print(wcnf.soft)

    # solving the MaxSAT problem
    with RC2(wcnf) as rc2:
        rc2.compute() 
        # print(rc2.cost, rc2.model) # cost + solution
        corrected = np.array(rc2.model).reshape(N, B)
        corrected = np.heaviside(corrected, 0).astype(int)
        flip = np.logical_xor(corrected, predictions)

        # print(corrected)
        # print(predictions)
        # print(flip)
    rc2.delete()
    
    if return_flip_mask:
        return corrected, flip
    return corrected

# Correction methods
# predictions: (N, B) bool
# confidences: (N, B) float
# nli_matrix: (N, B, B, 3) float, last dimension is [entailment, neutral, contradiction]
def C_0(predictions, confidences, nli_matrix, return_flip_mask=False):
    if return_flip_mask:
        return predictions, np.zeros_like(predictions)
    return predictions

def C_1(predictions, confidences, nli_matrix, return_flip_mask=False):
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    contra_conf = contra_matrix * np.expand_dims(confidences, axis=1) # P(j correct) * P(i contradicts j)
    contra_conf = np.sum(contra_conf, axis=2) / (B - 1) # Mean of non-diagonal elements
    flip = (contra_conf > 0.5 * confidences)
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        # print(corrected)
        # print(flip)
        return corrected, flip
    return corrected

def C_2(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Estimate probability of flipping i using probabilities conditioned on contradiction,
    with assumption of i & not j vs. not i & j ratios being same as if i and j were independent"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    Pi = confidences.reshape((N, B, 1))
    Pj = confidences.reshape((N, 1, B))
    P_notij = ((1 - Pi)*Pj / ((1 - Pi)*Pj + Pi*(1 - Pj) + 1e-8)) * contra_matrix
    P_notinotj = ((1 - Pi)*(1 - Pj) / ((1 - Pi)*(1 - Pj) + Pi*Pj + 1e-8)) * (1 - contra_matrix)
    P_noti = P_notij + P_notinotj
    P_noti = np.sum(P_noti, axis=2) / (B - 1)
    flip = (P_noti > 0.5)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_3(predictions, confidences, nli_matrix, return_flip_mask=False):
    """C_1, but filtering at 0.5 without confidences"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    contra_conf = contra_matrix * np.expand_dims(confidences, axis=1) # P(j correct) * P(i contradicts j)
    contra_conf = np.sum(contra_conf, axis=2) / (B - 1) # Mean of non-diagonal elements
    flip = (contra_conf > 0.5)
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_4(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Old C_2"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    contra_conf = (contra_matrix - 0.5) * np.expand_dims(confidences, axis=1) # (P(i contradicts j) - 0.5) * P(j correct)
    contra_conf = np.sum(contra_conf, axis=2) # Sum of all elements (including diagonal)
    flip = (contra_conf > 0)

    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_5(predictions, confidences, nli_matrix, return_flip_mask=False):
    """C_4, but subtracting 0.5 off of confidences as well"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    contra_conf = (contra_matrix - 0.5) * (np.expand_dims(confidences, axis=1) - 0.5) # (P(i contradicts j) - 0.5) * (P(j correct) - 0.5)
    contra_conf = np.sum(contra_conf, axis=2) # Sum of all elements (including diagonal)
    flip = (contra_conf > 0)

    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_6(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Flip everything with >0.5 contradiction probability compared to most confident in batch"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    most_confident = np.argmax(confidences, axis=1).reshape((N, 1))
    most_confident = np.repeat(most_confident, B, axis=1).reshape((N, 1, B))
    # Contradiction probability with most confident prediction in each batch
    contra_conf = np.take_along_axis(contra_matrix, most_confident, axis=1).reshape((N, B))
    flip = (contra_conf > 0.5)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_7(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Flip everything with >0.7 contradiction probability compared to most confident in batch"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    most_confident = np.argmax(confidences, axis=1).reshape((N, 1))
    most_confident = np.repeat(most_confident, B, axis=1).reshape((N, 1, B))
    # Contradiction probability with most confident prediction in each batch
    contra_conf = np.take_along_axis(contra_matrix, most_confident, axis=1).reshape((N, B))
    flip = (contra_conf > 0.7)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected

def C_8(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Generate contradictions as where contradiction is highest NLI probability, zero out the rest,
    then flip the lower probability prediction for each contradictory pair"""
    N, B = predictions.shape
    contra_matrix = 1 * (np.argmax(nli_matrix, axis=3) == 2)
    contra_matrix = contra_matrix + contra_matrix.transpose((0, 2, 1)) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    comp = (confidences.reshape((N, B, 1)) - confidences.reshape((N, 1, B))) < 0
    contra_lower = np.logical_and(comp, contra_matrix) # Contradictory positions ij where i has lower confidence
    flip = np.any(contra_lower, axis=2) # Flip position i if ij contradict with i having lower confidence
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, flip
    return corrected


def test_case():
    # N = 3, B = 4
    predictions = np.array([[1,0,0,1],
                            [0,1,1,0],
                            [1,1,0,1]])
    confidences = np.array([[1,1,1,1],
                            [0,0,0,0],
                            [1,1,1,1]])
    nli_matrix = np.random.uniform(0, 1, size = (3,4,4,3))

    MaxSAT(predictions, confidences, nli_matrix, return_flip_mask=True)
    C_1(predictions, confidences, nli_matrix, return_flip_mask=True)

if __name__ == '__main__':
    test_case()
