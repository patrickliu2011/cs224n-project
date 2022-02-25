import numpy as np

# Correction methods
# predictions: (N, B) bool
# confidences: (N, B) float
# nli_matrix: (N, B, B, 3) float, last dimension is [entailment, neutral, contradiction]
def do_nothing(predictions, confidences, nli_matrix, return_flip_mask=False):
    if return_flip_mask:
        return predictions, np.zeros_like(predictions)
    return predictions

def correction_1(predictions, confidences, nli_matrix, return_flip_mask=False):
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    contra_conf = contra_matrix * np.expand_dims(confidences, axis=1) # P(j correct) * P(i contradicts j)
    contra_conf = np.sum(contra_conf, axis=2) / (B - 1) # Mean of non-diagonal elements
    flip = (contra_conf > 0.5 * confidences)

    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected

def correction_2(predictions, confidences, nli_matrix, return_flip_mask=False):
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    contra_conf = (contra_matrix - 0.5) * np.expand_dims(confidences, axis=1) # (P(j correct) - 0.5) * P(i contradicts j)
    contra_conf = np.sum(contra_conf, axis=2) # Sum of all elements (including diagonal)
    flip = (contra_conf > 0)

    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected

def correction_2(predictions, confidences, nli_matrix, return_flip_mask=False):
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    contra_conf = (contra_matrix - 0.5) * np.expand_dims(confidences, axis=1) # (P(j correct) - 0.5) * P(i contradicts j)
    contra_conf = np.sum(contra_conf, axis=2) # Sum of all elements (including diagonal)
    flip = (contra_conf > 0)

    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected

def correction_3(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Flip everything with >0.5 contradiction probability compared to most confident in batch"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    most_confident = np.argmax(confidences, axis=1).reshape((N, 1))
    most_confident = np.repeat(most_confident, B, axis=1).reshape((N, 1, B))
    # Contradiction probability with most confident prediction in each batch
    contra_conf = np.take_along_axis(contra_matrix, most_confident, axis=1).reshape((N, B))
    flip = (contra_conf > 0.5)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected

def correction_4(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Flip everything with >0.3 contradiction probability compared to most confident in batch"""
    N, B = predictions.shape
    contra_matrix = nli_matrix[:, :, :, 2]
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    most_confident = np.argmax(confidences, axis=1).reshape((N, 1))
    most_confident = np.repeat(most_confident, B, axis=1).reshape((N, 1, B))
    # Contradiction probability with most confident prediction in each batch
    contra_conf = np.take_along_axis(contra_matrix, most_confident, axis=1).reshape((N, B))
    flip = (contra_conf > 0.3)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected

def correction_5(predictions, confidences, nli_matrix, return_flip_mask=False):
    """Generate contradictions as where contradiction is highest NLI probability, zero out the rest,
    then flip anywhere contradictory"""
    N, B = predictions.shape
    contra_matrix = 1 * (np.argmax(nli_matrix, axis=3) == 2)
    contra_matrix = (contra_matrix + contra_matrix.transpose((0, 2, 1))) / 2
    contra_matrix[:, np.diag_indices(B)] = 0 # Set diagonals to 0
    flip = np.any(contra_matrix > 0, dim=2)
    
    corrected = predictions.copy()
    corrected[flip] = np.logical_not(corrected[flip])
    if return_flip_mask:
        return corrected, np.zeros_like(flip)
    return corrected