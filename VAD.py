import math
import numpy as np

def VAD(signal, norma, frame_len, k_th = 1):
    
    def preprocess(signal, frame_len):    
        N_frames = math.floor(len(signal)/frame_len)
        signal = signal[:N_frames*frame_len]
        A = np.zeros((N_frames, frame_len))
        E = np.zeros(N_frames)
        
        for i in range(N_frames):
            A[i,:] = signal[i*frame_len : (i+1)*frame_len]
            s = sum(A[i,j]**2 for j in range(frame_len))
            E[i] = (1/frame_len) * s
        return signal, E, N_frames

    signal, E_s, N_fr_s = preprocess(signal, frame_len)
    norma, E_n, N_fr_n = preprocess(norma, frame_len)

    detection = np.zeros(len(signal))
    
    E_th = k_th * max(E_n)
    
    for i in range(N_fr_s):
        if E_s[i] > E_th:
            detection[i*frame_len : (i+1)*frame_len] = max(signal)*1.1
    
    return signal, detection