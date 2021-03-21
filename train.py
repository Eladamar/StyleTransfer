import os

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import *


class Trainer:
    def __init__(self,
                 image_transformer,
                 vgg,
                 optimizer,
                 criterion,
                 style,
                 device='cuda',
                 tensorboard_logger=None):

        self.vgg = vgg.eval()
        self.image_transformer = image_transformer.train()
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.style = style
        self.log_name = tensorboard_logger
        self.tensorboard_logger = SummaryWriter(log_dir=f'./logs/{tensorboard_logger}')

        if self.device:
            self.vgg.to(device)
            self.image_transformer.to(self.device)
            self.style = self.style.to(device)

    def load_checkpoint(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint: {checkpoint} does not exist")
        checkpoint = torch.load(checkpoint)
        self.image_transformer.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        return epoch, step

    def train(self, data_loader, num_epochs, checkpoint=None):
        if checkpoint is None:
            start_epoch = step = 0
        else:
            start_epoch, step = self.load_checkpoint(checkpoint)

        y_s_features = self.vgg(self.style)
        y_s_gram = [batch_gram(style_feature) for style_feature in y_s_features]

        gamma_c = 1e1
        gamma_s = 2e7
        min_loss = None

        num_styles = torch.tensor(list(range(self.style.shape[0]))).to(self.device)
        batch_size = torch.tensor(list(range(data_loader.batch_size))).to(self.device)

        for epoch in range(start_epoch, num_epochs):
            for y_c, _ in tqdm(data_loader):
                # Loss is 0 initially
                feature_loss = style_loss = 0

                style_idx = num_styles[batch_size % len(num_styles)]
                # case #style images > batch size
                num_styles = num_styles.roll(len(batch_size))

                y_c = y_c.to(self.device)
                y_c_features = self.vgg(y_c)

                y_hat = self.image_transformer(y_c, style_idx)
                y_hat_features = self.vgg(y_hat)

                feature_loss = batch_loss(y_hat_features[2], y_c_features[2], self.criterion)

                # iterate through all the features for the chosen layers
                for gen_feature, A in zip(y_hat_features, y_s_gram):
                    # Compute Gram Matrix
                    G = batch_gram(gen_feature)
                    style_loss += batch_loss(A[style_idx], G, self.criterion)

                total_loss = gamma_c * feature_loss + gamma_s * style_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if min_loss is None or total_loss < min_loss:
                    torch.save(self.image_transformer.state_dict(), f'{self.log_name}.model')
                    min_loss = total_loss

                if self.tensorboard_logger:
                    step += 1
                    self.tensorboard_logger.add_scalar('loss', total_loss, step)

            torch.save({
                'step': step,
                'epoch': epoch + 1,
                'model_state_dict': self.image_transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': total_loss}, f'{self.log_name}_{epoch}.ckpt')

        self.tensorboard_logger.close()
