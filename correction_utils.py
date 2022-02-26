import numpy as np

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
    P_notij = ((1 - Pi)*Pj / ((1 - Pi)*Pj + Pi*(1 - Pj))) * contra_matrix
    P_notinotj = ((1 - Pi)*(1 - Pj) / ((1 - Pi)*(1 - Pj) + Pi*Pj)) * (1 - contra_matrix)
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