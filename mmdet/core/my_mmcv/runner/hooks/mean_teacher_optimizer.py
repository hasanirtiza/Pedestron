from torch.nn.utils import clip_grad

from mmcv.runner.hooks.hook import Hook


class OptimizerHook(Hook):

    def __init__(self, grad_clip=None, mean_teacher=None):
        self.grad_clip = grad_clip
        self.mean_teacher = mean_teacher

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()

        #mean teacher
        if self.mean_teacher:
            for k, v in runner.model.module.state_dict().items():
                if k.find('num_batches_tracked') == -1:
                    runner.teacher_dict[k] = self.mean_teacher.alpha * runner.teacher_dict[k] + (1 - self.mean_teacher.alpha) * v
                else:
                    runner.teacher_dict[k] = 1 * v