import torchaudio
import librosa
import torch
import numpy as np

class Predictor:
  def __init__(self, processor, model, config, device):
 
      self.processor = processor
      self.model = model
      self.config = config
      self.device=device

  def speech_file_to_array_fn(self,batch):
      speech_array, sampling_rate = torchaudio.load(batch["path"])
      speech_array = speech_array.squeeze().numpy()
      speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=self.processor.sampling_rate)

      batch["speech"] = speech_array
      return batch



  def predict(self,batch):
      features = self.processor(batch["speech"], sampling_rate=self.processor.sampling_rate, return_tensors="pt", padding=True)

      input_values = features.input_values.to(self.device)
      attention_mask = features.attention_mask.to(self.device)

      with torch.no_grad():
          logits = self.model(input_values, attention_mask=attention_mask).logits

      pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
      batch["predicted"] = pred_ids
      return batch
