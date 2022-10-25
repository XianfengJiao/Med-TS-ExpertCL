import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, size):
        diag = np.eye(2 * size)
        l1 = np.eye((2 * size), 2 * size, k=-size)
        l2 = np.eye((2 * size), 2 * size, k=size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        size = len(zis)
        representations = torch.cat([zjs, zis], dim=0)
        mask_samples_from_same_repr = self._get_correlated_mask(size).type(torch.bool)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        
        l_pos = torch.diag(similarity_matrix, size)
        r_pos = torch.diag(similarity_matrix, -size)
        positives = torch.cat([l_pos, r_pos]).view(2 * size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * size)

class ExpertCLLoos(torch.nn.Module):
    def __init__(self, device, temperature=1, maxdelta=1, similarity_name='l2'):
        super(ExpertCLLoos, self).__init__()
        self.temperature = temperature
        self.maxdelta = maxdelta
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(similarity_name)

    @staticmethod
    def _l2_dist(x, y):
        """
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    def _get_similarity_function(self, similarity_name='l2'):
        if similarity_name == 'l2':
            return self._l2_dist
        else:
            raise ValueError("Invalid Similarity Type!")
    
    def forward(self, feature, exp_feature):
        size = len(feature)
        exp_feature = exp_feature.flatten(start_dim=1)
        # calculate dist for expert feature
        dist_exp = self.similarity_function(exp_feature, exp_feature)
        dist_exp /= torch.max(dist_exp)
        dist_exp = (1 - dist_exp).square()
        # calculate dist for encoded feature
        dist_en = self.similarity_function(feature, feature)
        # # mask diagonal element
        # dist_exp -= torch.diag_embed(dist_exp)
        # dist_en -= torch.diag_embed(dist_en)
        # calculate loss for exp-dist and feature-dist
        loss = ((1-dist_exp)*self.maxdelta - dist_en).square() / self.temperature
        loss = loss.exp().mean()
        loss = loss.log() * self.temperature
        return loss

        



