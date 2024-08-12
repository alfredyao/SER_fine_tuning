import torchaudio
class Audiopreprocessor:
  def __init__(self, target_sampling_rate, processor, label_list, input_column, output_column):
      self.target_sampling_rate = target_sampling_rate
      self.processor = processor
      self.label_list = label_list
      self.input_column = input_column
      self.output_column = output_column

  def speech_file_to_array_fn(self,path):
      speech_array, sampling_rate = torchaudio.load(path)
      resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
      speech = resampler(speech_array).squeeze().numpy()
      return speech

  def label_to_id(self,label, label_list):

      if len(self.label_list) > 0:
          return self.label_list.index(label) if label in label_list else -1

      return label

  def preprocess_function(self, examples):
      speech_list = [self.speech_file_to_array_fn(path) for path in examples[self.input_column]]
      target_list = [self.label_to_id(label, self.label_list) for label in examples[self.output_column]]

      result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
      result["labels"] = list(target_list)

      return result