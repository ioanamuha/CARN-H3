import torch
import torch.optim as optim


class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Newton-Schulz Orthogonalization
                if g.ndim > 1:
                    original_shape = g.shape
                    g_mat = g.view(g.size(0), -1)
                    X = g_mat
                    for _ in range(ns_steps):
                        norm = torch.norm(X) + 1e-8
                        X = X / norm
                        A = torch.mm(X.T, X)
                        eye = torch.eye(A.size(0), device=A.device)
                        X = torch.mm(X, 1.5 * eye - 0.5 * A)
                    g = X.view(original_shape)

                p.data.add_(g, alpha=-lr)
        return None
