import librosa
import numpy as np
from numpy import linalg as LA
from scipy.signal import hilbert


class Evaluator(object):
    def __init__(self, norm=False):
        self.env_loss = []
        self.mag_loss = []
        self.snr_loss = []
        self.snr_norm_loss = []
        self.norm = norm

    def update(self, mag_prd, mag_gt, wav_prd, wav_gt):
        mag_loss = np.mean(np.power(mag_prd - mag_gt, 2)) * 2
        self.mag_loss.append(mag_loss)
        env_loss = self.Envelope_distance(wav_prd, wav_gt)
        self.env_loss.append(env_loss)
        snr_loss = self.SNR(wav_prd, wav_gt)
        self.snr_loss.append(snr_loss)

        wav_prd = normalize(wav_prd)
        wav_gt = normalize(wav_gt)
        snr_norm_loss = self.SNR(wav_prd, wav_gt)
        self.snr_norm_loss.append(snr_norm_loss)

        return [mag_loss, env_loss, snr_loss, snr_norm_loss]

    def report(self):
        item_len = len(self.mag_loss)
        return {
            "env": sum(self.env_loss) / item_len,
            "mag": sum(self.mag_loss) / item_len,
            "snr": sum(self.snr_loss) / item_len,
            "snr_norm": sum(self.snr_norm_loss) / item_len,
        }

    def Envelope_distance(self, predicted_binaural, gt_binaural):
        pred_env_channel1 = np.abs(hilbert(predicted_binaural[0, :]))
        gt_env_channel1 = np.abs(hilbert(gt_binaural[0, :]))
        channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1) ** 2))

        pred_env_channel2 = np.abs(hilbert(predicted_binaural[1, :]))
        gt_env_channel2 = np.abs(hilbert(gt_binaural[1, :]))
        channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2) ** 2))

        envelope_distance = channel1_distance + channel2_distance
        return float(envelope_distance)

    def SNR(self, predicted_binaural, gt_binaural):
        mse_distance = np.mean(np.power((predicted_binaural - gt_binaural), 2))
        snr = 10. * np.log10((np.mean(gt_binaural ** 2) + 1e-4) / (mse_distance + 1e-4))
        return float(snr)


def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))
