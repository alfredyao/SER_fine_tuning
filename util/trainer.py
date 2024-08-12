from transformers import Trainer
import matplotlib.pyplot as plt
from IPython.display import clear_output

class CustomTrainer(Trainer):
    def __init__(self, *args, scheduler=None, activate_after_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.activate_after_steps = activate_after_steps
        self.train_loss = []



    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer = self.optimizer


    def log(self, logs):

        if 'loss' in logs:
            self.train_loss.append(logs['loss'])
            self.plot_training_loss()
        super().log(logs)

    def plot_training_loss(self):

        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_loss, label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
