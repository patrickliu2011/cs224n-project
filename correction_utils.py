from xmlrpc.client import FastMarshaller
import numpy as np
from pysat.examples.rc2 import RC2 # RC2 MaxSat solver https://alexeyignatiev.github.io/assets/pdf/imms-jsat19-preprint.pdf
from pysat.formula import WCNF # for installation, see: https://pysathq.github.io/installation.html (pip install python-sat)

### MaxSAT formulation and solution
def MaxSAT(predictions, confidences, nli_matrix, return_flip_mask=False):
    print("MAXSAT Run")
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0
    corrected = np.array([])
    flip = np.array([])

    for i in range(N): # each batch individually has its own SAT
        wcnf = WCNF()

        pred_row = predictions[i]
        conf_row = confidences[i]
        # j + i * B + 1 is the # assigned to constraint at index (i, j)
        constraints = [ ([j + 1] if pred_row[j] == 1 else [-(j + 1)]) for j in range(B)]
        # print (constraints, conf_row.tolist())
        wcnf.extend(constraints, weights=conf_row.tolist())

        contradiction_constraints = []
        weights = []
        for j in range(B):
            for k in range(B):
                if k > j:
                    if contra_matrix[i][j][k] != 0: # weight != 0

                        P = j + 1 if predictions[i][j] == 1 else -(j + 1)
                        Q = k + 1 if predictions[i][k] == 1 else -(k + 1)
                        contradiction_constraints.append([-P, -Q])      # P ∨ Q == ~ (~P ∧ ~Q)
                        # TODO: play with scaling factor below (current best is +2)
                        weights.append((contra_matrix[i][j][k] - 0.5) * (5)) # linearly scale from (0,1) to - weight if p < 0.5 / + weight if p > 0.5
        # print(contradiction_constraints, weights)
        wcnf.extend(contradiction_constraints, weights=weights)

        # solving the MaxSAT problem
        rc2 = RC2(wcnf)
        rc2.compute() 
        # print("Time taken: " + str(rc2.oracle_time()))
        # print(rc2.cost, rc2.model) # cost + solution

        # print(rc2.cost, rc2.model) # cost + solution
        # print(np.heaviside(np.array(rc2.model), 0).astype(int))
        batch_pred = np.heaviside(np.array(rc2.model), 0).astype(int)
        corrected = np.vstack((corrected, batch_pred )) \
            if corrected.size else batch_pred 

        # print(corrected)
        rc2.delete()
        
    corrected.reshape(N, B)
    flip = np.logical_xor(corrected, predictions)

    print("Final MaxSAT Answers")
    print(corrected)
    print(predictions)
    print(flip)

    if return_flip_mask:
        return corrected, flip
    return corrected

### Correction Methods
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
    # (N, B) -- 1 val for each statement
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

def C_9(predictions, confidences, nli_matrix, return_flip_mask = False):
    """
    Entailment-based approach. Start with C_1.
    Then if statement B is flipped and A→B has a high entailment, then flip A.
    """
    entailment_threshold = 0.5
    flip_threshold = 0.75 # this threshold must be > 0.5

    N, B = predictions.shape
    entailment_matrix = nli_matrix[:, :, :, 0] # N x B x B

    corrected_preds, flip = C_1(predictions, confidences, nli_matrix, return_flip_mask = True) # flip: N x B

    # print(flip * entailment_matrix)
    flip = np.expand_dims(flip, axis = 1)
    N_idxs, A_idxs, B_idxs = np.where((flip * entailment_matrix) > entailment_threshold) # indices of statements where B is flipped and A->B has high entailment
    
    # Flip statement A if p < some threshold
    for N_idx, A_idx in zip(N_idxs, A_idxs):
        if(confidences[N_idx, A_idx] < flip_threshold):
            confidences[N_idx, A_idx] = 1 - confidences[N_idx, A_idx]
            corrected_preds[N_idx, A_idx] = np.logical_not(corrected_preds[N_idx, A_idx])
    return corrected_preds, flip

def C_10(predictions, confidences, nli_matrix, delta = 0.2, return_flip_mask = False): # delta = 0.0 is equivalent to running vanilla C_1

    '''
    Use entailment scores to augment contradiction scores. If entailment score of (A entails ~B) > threshold, 
    and A true but B false, decrease A's confidence.

    (Prob not theoretically sound)
    '''
    N, B = predictions.shape
    entailment_threshold = 0.5

    entailment_matrix = nli_matrix[:, :, :, 0] # N x B x B
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, range(B), range(B)] = 0 # Set diagonals to 0

    N_idxs, A_idxs, B_idxs = np.where(entailment_matrix > entailment_threshold)

    for N_idx, A_idx, B_idx in zip(N_idxs, A_idxs, B_idxs):
        # Prediction: A true, B false. P(A true entails B false) > 0.5. So decrease A conf.

        if(predictions[N_idx, A_idx] and not predictions[N_idx, B_idx]): # A true, B false
            confidences[N_idx, A_idx] -= delta # decrease A conf
        # if(not predictions[N_idx, A_idx] and predictions[N_idx, B_idx]): # A false, B true
        #     confidences[N_idx, A_idx] += delta # increase A conf
        # Below does NOT work:
        # if(predictions[N_idx, A_idx] and predictions[N_idx, B_idx]): 
        #     confidences[N_idx, A_idx] -= delta

    # Now apply existing correction func
    return C_1(predictions, confidences, nli_matrix, return_flip_mask=return_flip_mask)

def C_11(predictions, confidences, nli_matrix, return_flip_mask = False):
    '''
    Use entailment scores & predictions to requery NLI and generate new entailment scores.
    When prediction of (A true, B false), and A entails B (A->B), query NLI to generate new entailment score
    of (A, not B) and/or (B, not A). 
    If A entails not B (A-> ~B) is high, and B's conf is low, flip B.
    '''
    pass

def test_multiply():
    N = 2
    B = 3
    flip = np.array([[1, 0, 0],
                     [0, 1, 1]])
    flip = np.expand_dims(flip, axis = 1)
    entailment_matrix = np.random.uniform(0, 1, size = (N, B, B))
    test = np.nonzero(entailment_matrix > 0.9)
    N_idxs, A_idxs, B_idxs = test

    confidences = np.random.uniform(0, 1, size = (N, B))
    predictions = np.random.randint(0, 2, size= (N, B))

    # Flip statement A if p < some threshold
    print(confidences, predictions)
    print("Running flip")
    for N_idx, A_idx in zip(N_idxs, A_idxs):
        if(confidences[N_idx, A_idx] < 0.5):
            confidences[N_idx, A_idx] = 1 - confidences[N_idx, A_idx]
            predictions[N_idx, A_idx] = np.logical_not(predictions[N_idx, A_idx])
    print(confidences, predictions)
    # print(test)
    # print(entailment_matrix[N_idx, A_idx, B_idx])    
    # print(flip, entailment_matrix)

    # print(flip * entailment_matrix)

# def test_case():
#     # N = 1
#     # B = 400
#     # predictions = np.random.randint(0, 2, size= (N, B))
#     # confidences = np.random.uniform(0, 1, size = (N, B))
#     # nli_matrix = np.random.uniform(0, 1, size = (N, B, B, 3))


#     # N = 3 B = 4
#     predictions = np.array([[1,0,0,1],
#                             [0,1,1,0],
#                             [1,1,0,1]])
#     confidences = np.array([[1,1,1,1],
#                             [0,0,0,0],
#                             [1,1,1,1]])
#     nli_matrix = np.random.uniform(0, 1, size = (3,4,4,3))

#     MaxSAT(predictions, confidences, nli_matrix, return_flip_mask=True)
#     C_1(predictions, confidences, nli_matrix, return_flip_mask=True)

if __name__ == '__main__':
    # test_case()
    test_multiply()
