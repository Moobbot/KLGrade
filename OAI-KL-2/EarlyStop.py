class EarlyStopping:
    """�־��� patience ���ķ� validation loss�� �������� ������ �н��� ���� ����"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss�� ������ �� ��ٸ��� �Ⱓ
                            Default: 7
            verbose (bool): True�� ��� �� validation loss�� ���� ���� �޼��� ���
                            Default: False
            delta (float): �����Ǿ��ٰ� �����Ǵ� monitered quantity�� �ּ� ��ȭ
                            Default: 0
            path (str): checkpoint���� ���
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss�� �����ϸ� ���� �����Ѵ�.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model_ft,'./models/kfold_CNN_{}fold_epoch{}.pt'.format(fold + 1, epoch + 1))
        self.val_loss_min = val_loss